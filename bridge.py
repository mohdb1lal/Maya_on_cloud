
import pjsua2 as pj
import asyncio
import websockets
import json
import numpy as np
import time
import threading
import queue
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
# import librosa


@dataclass
class BridgeConfig:
    """Configuration for the SIP-WebSocket bridge"""
    # SIP Configuration
    SIP_USER: str = "4qjpoz2h"
    SIP_PASSWORD: str = "Admin@123"
    SIP_DOMAIN: str = "pbx.voxbaysolutions.com"
    SIP_PORT: int = 5260
    SIP_TRANSPORT_PORT: int = 5060
    
    # WebSocket Configuration
    # WS_URI: str = "ws://13.233.41.221:8081" # AWS
    # WS_URI: str = "ws://34.93.85.137:8081" # GCP
    WS_URI: str = "ws://localhost:8081" # Local testing
    WS_RECONNECT_DELAY: int = 5
    WS_PING_INTERVAL: int = 30
    WS_PING_TIMEOUT: int = 10
    
    # Audio Configuration
    SIP_SAMPLE_RATE: int = 8000      # Phone uses 8kHz
    AI_INPUT_RATE: int = 16000       # Gemini expects 16kHz input
    AI_OUTPUT_RATE: int = 24000      # Gemini outputs 24kHz
    SAMPLES_PER_FRAME: int = 160     # 20ms at 8kHz
    BITS_PER_SAMPLE: int = 16
    CHANNELS: int = 1
    
    
    # Call Configuration
    AUTO_ANSWER: bool = True
    MAX_CALL_DURATION: int = 0  # seconds
    
    # Debug Configuration
    DEBUG_AUDIO: bool = True
    LOG_LEVEL: int = logging.INFO


def setup_logging(level: int = logging.INFO):
    """Configure logging with color support"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('sip_bridge.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('SIPBridge')

logger = setup_logging(BridgeConfig.LOG_LEVEL)

class AudioResampler:
    @staticmethod
    def resample(audio_data: bytes, from_rate: int, to_rate: int) -> bytes:
        try:
            if len(audio_data) % 2 != 0:
                audio_data = audio_data[:-1]
            
            if len(audio_data) == 0:
                return b''
            
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            if from_rate == to_rate:
                return audio_data
            
            # Use scipy for ALL resampling - your fallback methods are causing quality issues
            from scipy import signal
            
            # Use resample_poly for better quality
            resampled = signal.resample_poly(audio_array, to_rate, from_rate)
            
            # Proper clipping without distortion
            resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
            
            return resampled.tobytes()
            
        except Exception as e:
            logger.error(f"Resampling error: {e}")
            return audio_data

class AudioAnalyzer:
    """Analyze audio for debugging purposes"""
    
    @staticmethod
    def calculate_rms(audio_data: bytes) -> float:
        """Calculate RMS (volume level) of audio data"""
        try:
            if len(audio_data) % 2 != 0:
                audio_data = audio_data[:-1]
            if len(audio_data) == 0:
                return 0.0
            
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            rms = np.sqrt(np.mean(np.square(audio_array.astype(np.float32))))
            return float(rms)
        except:
            return 0.0


class WebSocketState(Enum):
    """WebSocket connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    READY = "ready"
    ERROR = "error"

class AIWebSocketClient:
    """WebSocket client for AI agent communication"""
    
    def __init__(self, config: BridgeConfig):
        self.config = config
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.state = WebSocketState.DISCONNECTED
        self.audio_send_queue = asyncio.Queue(maxsize=1000)
        self.audio_receive_queue = asyncio.Queue(maxsize=1000)
        self.call_id: Optional[str] = None
        self.caller_id: Optional[str] = None
        self.stats = {
            'packets_sent': 0,
            'packets_received': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'last_sent': None,
            'last_received': None
        }
        # Store the sample rates for easy access
        self.input_rate = config.AI_INPUT_RATE
        self.output_rate = config.AI_OUTPUT_RATE
    
    def _is_websocket_open(self) -> bool:
        """Check if websocket connection is open (handles different library versions)"""
        if not self.websocket:
            return False
        
        try:
            # Try different attributes for different websockets library versions
            if hasattr(self.websocket, 'open'):
                return self.websocket.open
            elif hasattr(self.websocket, 'closed'):
                return not self.websocket.closed
            else:
                # Fallback: assume open if we can access the websocket object
                return True
        except Exception:
            return False
        
    async def connect(self, call_id: str, caller_id: str) -> bool:
        """Connect to AI agent WebSocket server"""
        self.call_id = call_id
        self.caller_id = caller_id
        self.state = WebSocketState.CONNECTING
        
        try:
            logger.info(f"🔌 Connecting to AI agent at {self.config.WS_URI}")
            
            self.websocket = await websockets.connect(
                self.config.WS_URI,
                ping_interval=self.config.WS_PING_INTERVAL,
                ping_timeout=self.config.WS_PING_TIMEOUT,
                max_size=10 * 1024 * 1024  # 10MB max message size
            )
            
            self.state = WebSocketState.CONNECTED
            logger.info("✅ WebSocket connected successfully")
            
            # Send initial handshake
            await self._send_handshake()
            
            # Start message handler
            asyncio.create_task(self._handle_messages())
            
            # Start audio sender
            asyncio.create_task(self._send_audio_loop())
            
            return True
            
        except Exception as e:
            logger.error(f"❌ WebSocket connection failed: {e}")
            self.state = WebSocketState.ERROR
            return False
    
    async def _send_handshake(self):
        """Send initial handshake messages to AI agent"""
        try:
            # Send call start event
            start_msg = {
                "event": "start",
                "call_id": self.call_id,
                "caller_id": self.caller_id,
                "timestamp": datetime.now().isoformat()
            }
            await self.websocket.send(json.dumps(start_msg))
            logger.info(f"📤 Sent start event: {start_msg}")
            
            # Send media configuration - must match what Gemini expects
            media_msg = {
                "event": "media",
                "data": {
                    "sample_rate": self.config.AI_INPUT_RATE,  # 16000
                    "codec": "L16",
                    "channels": self.config.CHANNELS
                }
            }
            await self.websocket.send(json.dumps(media_msg))
            logger.info(f"📤 Sent media config: {media_msg}")
            
            self.state = WebSocketState.READY
            logger.info("✅ Handshake complete - ready for audio")
            
        except Exception as e:
            logger.error(f"❌ Handshake failed: {e}")
            self.state = WebSocketState.ERROR
 
    async def send_audio(self, audio_data: bytes):
        """Queue audio for sending to AI agent"""
        if self.state != WebSocketState.READY:
            logger.warning(f"⚠️ WebSocket not ready (state: {self.state}), dropping audio")
            return
        
        try:
            # Resample from 8kHz (SIP) to 16kHz (what Gemini expects)
            audio_16k = AudioResampler.resample(
                audio_data, 
                self.config.SIP_SAMPLE_RATE,     # 8000
                self.config.AI_INPUT_RATE         # 16000
            )
            
            if len(audio_16k) > 0:
                await self.audio_send_queue.put(audio_16k)
                
                # Enhanced debug logging
                if self.config.DEBUG_AUDIO:
                    rms = AudioAnalyzer.calculate_rms(audio_16k)
                    if self.stats['packets_sent'] < 10 or self.stats['packets_sent'] % 50 == 0:
                        logger.info(f"🎤 Queued audio: {len(audio_data)}→{len(audio_16k)} bytes (8kHz→16kHz), RMS: {rms:.1f}")
                        
        except asyncio.QueueFull:
            logger.warning("⚠️ Audio send queue full, dropping packet")
        except Exception as e:
            logger.error(f"❌ Error queueing audio: {e}")


    async def _send_audio_loop(self):
        """Continuously send queued audio to AI agent"""
        logger.info("🎤 Starting audio transmission loop")
        
        while self.state in [WebSocketState.CONNECTED, WebSocketState.READY]:
            try:
                # Get audio from queue with timeout
                audio_data = await asyncio.wait_for(
                    self.audio_send_queue.get(), 
                    timeout=0.1
                )
                
                # Send raw binary audio directly (like the working test client)
                if self.websocket and self._is_websocket_open():
                    await self.websocket.send(audio_data)
                    
                    # Update stats
                    self.stats['packets_sent'] += 1
                    self.stats['bytes_sent'] += len(audio_data)
                    self.stats['last_sent'] = time.time()
                    
                    # Debug logging
                    if self.stats['packets_sent'] % 100 == 0:
                        logger.info(f"📊 Sent {self.stats['packets_sent']} packets ({self.stats['bytes_sent']} bytes)")
                        
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"❌ Error in audio send loop: {e}")
                await asyncio.sleep(0.01)
        
        logger.info("🛑 Audio transmission loop ended")
    
    async def _handle_messages(self):
        """Handle incoming messages from AI agent"""
        logger.info("📥 Starting message reception loop")
        
        try:
            async for message in self.websocket:
                if isinstance(message, bytes):
                    # Audio response from AI
                    await self._handle_audio_response(message)
                else:
                    # JSON control message
                    await self._handle_control_message(message)
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("📵 WebSocket connection closed")
        except Exception as e:
            logger.error(f"❌ Error in message handler: {e}")
        finally:
            self.state = WebSocketState.DISCONNECTED
    
   
    async def _handle_audio_response(self, audio_data: bytes):
        try:
            self.stats['packets_received'] += 1
            
            # Just use numpy - no fancy libraries needed
            audio_24k = np.frombuffer(audio_data, dtype=np.int16)
            
            # Simple 3:1 decimation for 24kHz → 8kHz
            # This preserves the volume perfectly
            audio_8k = audio_24k[::2]
            
            # Optional: Apply slight amplification if needed
            # audio_8k = (audio_8k * 1.5).clip(-32768, 32767).astype(np.int16)
            
            # Debug log
            if self.stats['packets_received'] % 50 == 0:
                rms = np.sqrt(np.mean(np.square(audio_8k.astype(np.float32))))
                max_val = np.max(np.abs(audio_8k))
                logger.info(f"🔊 Audio: RMS={rms:.1f}, Max={max_val}, Samples={len(audio_8k)}")
            
            await self.audio_receive_queue.put(audio_8k.astype(np.int16).tobytes())
            
        except Exception as e:
            logger.error(f"Error: {e}")




    async def _handle_control_message(self, message: str):
        """Handle JSON control messages from AI agent"""
        try:
            data = json.loads(message)
            event = data.get('event', 'unknown')
            
            logger.info(f"📨 Received event: {event}")
            
            if event == 'ready':
                logger.info("✅ AI agent is ready")
            elif event == 'started':
                logger.info("✅ Call started confirmation")
            elif event == 'audio_complete':
                logger.info("🔊 AI finished speaking")
            elif event == 'transfer':
                logger.info(f"📞 Transfer requested to: {data.get('department')}")
            elif event == 'ended':
                logger.info(f"📵 Call ended: {data.get('reason')}")
                
        except json.JSONDecodeError:
            logger.error(f"❌ Invalid JSON: {message[:100]}")
        except Exception as e:
            logger.error(f"❌ Error handling control message: {e}")
    
    async def disconnect(self):
        """Disconnect from AI agent"""
        logger.info("🔌 Disconnecting from AI agent")
        
        if self.websocket and self._is_websocket_open():
            try:
                # Send hangup event
                hangup_msg = {
                    "event": "hangup",
                    "call_id": self.call_id,
                    "reason": "normal"
                }
                await self.websocket.send(json.dumps(hangup_msg))
                
                # Close connection
                await self.websocket.close()
                
            except Exception as e:
                logger.error(f"❌ Error during disconnect: {e}")
        
        self.state = WebSocketState.DISCONNECTED
        logger.info("✅ Disconnected from AI agent")
        
        # Print final stats
        logger.info(f"📊 Final stats: Sent {self.stats['packets_sent']} packets, Received {self.stats['packets_received']} packets")


class AudioBridge(pj.AudioMediaPort):
    """Custom audio port for capturing and injecting audio"""
    
    def __init__(self, config: BridgeConfig):
        pj.AudioMediaPort.__init__(self)
        self.config = config
        self.capture_queue = queue.Queue(maxsize=1000)  # Threading queue for PJSIP
        self.playback_queue = queue.Queue(maxsize=1000)  # Threading queue for PJSIP
        self.active = False
        self.test_mode = False
        self.stats = {
            'frames_captured': 0,
            'frames_played': 0,
            'capture_overruns': 0,
            'playback_underruns': 0
        }
        
        # Create media format using MediaFormatAudio
        fmt = pj.MediaFormatAudio()
        fmt.type = pj.PJMEDIA_TYPE_AUDIO
        fmt.clockRate = config.SIP_SAMPLE_RATE
        fmt.channelCount = config.CHANNELS
        fmt.bitsPerSample = config.BITS_PER_SAMPLE
        fmt.frameTimeUsec = (config.SAMPLES_PER_FRAME * 1000000) // config.SIP_SAMPLE_RATE
        
        logger.info(f"🎵 Audio format: {config.SIP_SAMPLE_RATE}Hz, {config.CHANNELS}ch, {config.BITS_PER_SAMPLE}bit")
        
        # Create the port with audio format
        self.createPort("ai_audio_bridge", fmt)
        
        logger.info(f"🎵 Audio bridge created: {config.SIP_SAMPLE_RATE}Hz, {config.SAMPLES_PER_FRAME} samples/frame")
    
    def _amplify_audio(self, audio_data: bytes, gain: float) -> bytes:
        """Amplify audio data by a gain factor"""
        try:
            # Convert to numpy array for processing
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Apply gain
            amplified_array = (audio_array * gain).astype(np.float32)
            
            # Clip to prevent overflow
            amplified_array = np.clip(amplified_array, -32768, 32767)
            
            # Convert back to int16
            amplified_array = amplified_array.astype(np.int16)
            
            return amplified_array.tobytes()
        except Exception as e:
            logger.error(f"❌ Audio amplification error: {e}")
            return audio_data
    
    def generate_test_tone(self, frequency=1000, duration_seconds=0.02):
        """Generate a test tone at 8kHz"""
        sample_rate = self.config.SIP_SAMPLE_RATE
        t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
        tone = (np.sin(2 * np.pi * frequency * t) * 16384).astype(np.int16)
        return tone.tobytes()
    
    def _fill_silence_frame(self, frame, size):
        """Fill frame with silence"""
        frame.size = size
        frame.type = pj.PJMEDIA_FRAME_TYPE_AUDIO
        frame.buf = pj.ByteVector()
        for _ in range(size):
            frame.buf.append(0)
    
    def onFrameRequested(self, frame):
        """Called when PJSIP needs audio to send to the phone - RUNS IN PJSIP THREAD"""
        if self.stats['frames_played'] == 0:
            logger.info("🎵 First onFrameRequested call!")
        
        self.stats['frames_played'] += 1
        
        try:
            expected_size = self.config.SAMPLES_PER_FRAME * 2
            
            if self.active and not self.playback_queue.empty():
                try:
                    # Get audio from queue (no asyncio here!)
                    ai_audio = self.playback_queue.get_nowait()
                    
                    # Ensure correct size
                    if len(ai_audio) < expected_size:
                        ai_audio = ai_audio + bytes(expected_size - len(ai_audio))
                    elif len(ai_audio) > expected_size:
                        ai_audio = ai_audio[:expected_size]
                    
                    # Set frame properties
                    frame.size = expected_size
                    frame.type = pj.PJMEDIA_FRAME_TYPE_AUDIO
                    
                    # Create buffer
                    frame.buf = pj.ByteVector()
                    for byte in ai_audio:
                        frame.buf.append(byte)
                    
                    # Minimal logging to avoid performance issues
                    if self.stats['frames_played'] % 100 == 0:
                        logger.debug(f"🎵 Frame #{self.stats['frames_played']}")
                        
                except queue.Empty:
                    self._fill_silence_frame(frame, expected_size)
                    self.stats['playback_underruns'] += 1
            else:
                self._fill_silence_frame(frame, expected_size)
                
        except Exception as e:
            logger.error(f"❌ Error in onFrameRequested: {e}")
            self._fill_silence_frame(frame, self.config.SAMPLES_PER_FRAME * 2)

            
   
    def onFrameReceived(self, frame):
        """Called when PJSIP receives audio from phone - RUNS IN PJSIP THREAD"""
        if not self.active:
            return
        
        try:
            if frame.size > 0:
                try:
                    if hasattr(frame, 'buf') and frame.buf is not None:
                        audio_data = bytes(frame.buf[:frame.size])
                    else:
                        audio_data = bytes(frame.size)
                except Exception as buffer_error:
                    logger.error(f"❌ Buffer conversion error: {buffer_error}")
                    audio_data = bytes(frame.size)
                
                try:
                    self.capture_queue.put_nowait(audio_data)
                    self.stats['frames_captured'] += 1
                    
                    if self.stats['frames_captured'] % 100 == 0:
                        logger.debug(f"📞 Frame #{self.stats['frames_captured']}")
                        
                except queue.Full:
                    try:
                        self.capture_queue.get_nowait()
                        self.capture_queue.put_nowait(audio_data)
                        self.stats['capture_overruns'] += 1
                    except:
                        pass
                        
        except Exception as e:
            logger.error(f"❌ Error in onFrameReceived: {e}")



    def start(self):
        """Start the audio bridge"""
        self.active = True
        logger.info("✅ Audio bridge started")
    
    def enable_test_mode(self):
        """Enable test mode to play test tones"""
        self.test_mode = True
        logger.info("🔊 Test mode enabled - will play test tones")
    
    def disable_test_mode(self):
        """Disable test mode"""
        self.test_mode = False
        logger.info("🔊 Test mode disabled")
    
    def stop(self):
        """Stop the audio bridge"""
        self.active = False
        logger.info(f"🛑 Audio bridge stopped - Stats: {self.stats}")

class AICall(pj.Call):
    """Call handler with AI integration"""
    
    def __init__(self, account, config: BridgeConfig, call_id=pj.PJSUA_INVALID_ID):
        pj.Call.__init__(self, account, call_id)
        self.config = config
        self.audio_bridge: Optional[AudioBridge] = None
        self.ai_client: Optional[AIWebSocketClient] = None
        self.call_active = False
        self.call_start_time: Optional[float] = None
        self.async_thread: Optional[threading.Thread] = None
        self.async_loop: Optional[asyncio.AbstractEventLoop] = None
        
    def onCallState(self, prm):
        """Handle call state changes"""
        try:
            ci = self.getInfo()
            logger.info(f"📞 Call state: {ci.stateText} ({ci.state})")
            
            if ci.state == pj.PJSIP_INV_STATE_CONFIRMED:
                # Call connected
                logger.info("✅ CALL CONNECTED - Starting AI integration")
                self.call_active = True
                self.call_start_time = time.time()
                
                # Start AI integration
                self._start_ai_integration(ci)
                
                # Set call duration timer
                if self.config.MAX_CALL_DURATION > 0:
                    timer = threading.Timer(self.config.MAX_CALL_DURATION, self._timeout_call)
                    timer.daemon = True
                    timer.start()
                    logger.info(f"⏰ Call will timeout in {self.config.MAX_CALL_DURATION} seconds")
                    
            elif ci.state == pj.PJSIP_INV_STATE_DISCONNECTED:
                # Call ended
                self.call_active = False
                if self.call_start_time:
                    duration = time.time() - self.call_start_time
                    logger.info(f"📵 CALL ENDED - Duration: {duration:.1f} seconds")
                
                # Stop AI integration
                self._stop_ai_integration()
                
        except Exception as e:
            logger.error(f"❌ Error in onCallState: {e}")
    
    def onCallMediaState(self, prm):
        """Handle media state changes"""
        try:
            ci = self.getInfo()
            
            for mi in ci.media:
                if mi.type == pj.PJMEDIA_TYPE_AUDIO and mi.status == pj.PJSUA_CALL_MEDIA_ACTIVE:
                    logger.info("🎵 Setting up audio bridge")
                    
                    # Get call audio media
                    aud_media = self.getAudioMedia(mi.index)
                    
                    # Create audio bridge
                    self.audio_bridge = AudioBridge(self.config)
                    
                    # Connect bidirectionally
                    aud_media.startTransmit(self.audio_bridge)  # Phone → Bridge
                    self.audio_bridge.startTransmit(aud_media)  # Bridge → Phone
                    
                    # Force audio bridge to be active
                    self.audio_bridge.active = True
                    
                    # Log the connection details
                    logger.info(f"🎵 Audio media index: {mi.index}")
                    logger.info(f"🎵 Audio bridge active: {self.audio_bridge.active}")
                    logger.info(f"🎵 Audio bridge test mode: {self.audio_bridge.test_mode}")
                    
                    # Log audio media details
                    try:
                        media_info = aud_media.getPortInfo()
                        logger.info(f"🎵 Audio media info: {media_info}")
                    except Exception as e:
                        logger.info(f"🎵 Could not get media info: {e}")
                    
                    # Test the connection
                    logger.info(f"🎵 Testing audio connection...")
                    try:
                        # Check what methods are available on audio media
                        methods = [method for method in dir(aud_media) if not method.startswith('_')]
                        logger.info(f"🎵 Available audio media methods: {methods[:10]}...")  # First 10 methods
                        
                        # Try to get port info
                        try:
                            port_info = aud_media.getPortInfo()
                            logger.info(f"🎵 Port info: {port_info}")
                        except Exception as e:
                            logger.info(f"🎵 Could not get port info: {e}")
                            
                        logger.info(f"🎵 Audio media test completed")
                    except Exception as e:
                        logger.warning(f"⚠️ Audio media test failed: {e}")
                    
                    # Start the audio bridge AFTER connecting
                    self.audio_bridge.start()
                    
                    # Enable test mode for debugging (comment out to disable)
                    # self.audio_bridge.enable_test_mode()
                    
                    # Test audio bridge immediately
                    self._test_audio_bridge()
                    
                    logger.info("✅ Audio bridge connected and started")
                    
        except Exception as e:
            logger.error(f"❌ Error in onCallMediaState: {e}")
            import traceback
            traceback.print_exc()

    def _test_audio_bridge(self):
        """Test if the audio bridge is working"""
        try:
            logger.info("🧪 Testing audio bridge...")
            
            # Create a test frame
            import pjsua2 as pj
            test_frame = pj.MediaFrame()
            test_frame.size = self.config.SAMPLES_PER_FRAME * 2
            
            # Call onFrameRequested manually
            if self.audio_bridge:
                self.audio_bridge.onFrameRequested(test_frame)
                logger.info(f"🧪 Test frame processed, size: {test_frame.size}")
            
        except Exception as e:
            logger.error(f"❌ Audio bridge test failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _start_ai_integration(self, call_info):
        """Start the AI integration in a separate thread"""
        def run_async_loop():
            try:
                # Extract caller information
                caller_id = "unknown"
                call_id = str(call_info.id) if call_info else "unknown"
                
                if call_info:
                    remote_uri = call_info.remoteUri
                    if "@" in remote_uri:
                        caller_part = remote_uri.split("@")[0]
                        if ":" in caller_part:
                            caller_id = caller_part.split(":")[-1].strip("<>\"")
                
                logger.info(f"🤖 Starting AI integration for call {call_id} from {caller_id}")
                
                # Create and run event loop
                self.async_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.async_loop)
                self.async_loop.run_until_complete(self._run_ai_client(call_id, caller_id))
                
            except Exception as e:
                logger.error(f"❌ AI integration error: {e}")
                import traceback
                traceback.print_exc()
        
        self.async_thread = threading.Thread(target=run_async_loop, daemon=True)
        self.async_thread.start()
    
    async def _run_ai_client(self, call_id: str, caller_id: str):
        """Run the AI client with audio bridging"""
        try:
            # Wait for audio bridge to be ready with a shorter timeout
            max_wait_ms = 2000  # 2 seconds total
            wait_interval_ms = 100
            total_waited_ms = 0
            while not self.audio_bridge or not self.audio_bridge.active:
                if total_waited_ms >= max_wait_ms:
                    logger.error(f"❌ Audio bridge not ready after {max_wait_ms / 1000} seconds")
                    return
                await asyncio.sleep(wait_interval_ms / 1000.0) # sleep for 100ms
                total_waited_ms += wait_interval_ms

            logger.info("✅ Audio bridge is ready")
            
            
            # Create AI client
            self.ai_client = AIWebSocketClient(self.config)
            
            # Connect to AI agent
            if not await self.ai_client.connect(call_id, caller_id):
                logger.error("❌ Failed to connect to AI agent")
                return
            
            logger.info("✅ AI agent connected and ready")
            
            # Create tasks for audio bridging
            tasks = [
                asyncio.create_task(self._bridge_captured_audio()),
                asyncio.create_task(self._bridge_playback_audio()),
                asyncio.create_task(self._monitor_call_state())
            ]
            
            # Run until call ends
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"❌ AI client error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Disconnect AI client
            if self.ai_client:
                await self.ai_client.disconnect()
    
    async def _bridge_captured_audio(self):
        """Bridge captured phone audio to AI agent"""
        logger.info("🎤 Starting capture bridge")
        packet_count = 0
        
        while self.call_active:
            try:
                if self.audio_bridge:
                    # Use asyncio.to_thread to safely access threading queue from async context
                    try:
                        audio_data = await asyncio.wait_for(
                            asyncio.to_thread(self.audio_bridge.capture_queue.get_nowait),
                            timeout=0.01
                        )
                        
                        # Send to AI agent
                        if self.ai_client:
                            await self.ai_client.send_audio(audio_data)
                            packet_count += 1
                            
                            if packet_count % 100 == 0:
                                logger.debug(f"📤 Bridged {packet_count} packets to AI")
                                
                    except (queue.Empty, asyncio.TimeoutError):
                        # No audio available, continue
                        await asyncio.sleep(0.001)
                        
                await asyncio.sleep(0.001)  # Small delay
                
            except Exception as e:
                logger.error(f"❌ Capture bridge error: {e}")
                await asyncio.sleep(0.1)
        
        logger.info(f"🛑 Capture bridge stopped ({packet_count} packets)")
    
    async def _bridge_playback_audio(self):
        """Bridge AI audio responses to phone"""
        logger.info("🔊 Starting playback bridge")
        packet_count = 0
        
        while self.call_active:
            try:
                if self.ai_client and self.audio_bridge:
                    # Get AI audio with timeout
                    try:
                        audio_data = await asyncio.wait_for(
                            self.ai_client.audio_receive_queue.get(),
                            timeout=0.1
                        )
                        
                        # Split into chunks for smooth playback
                        chunk_size = self.config.SAMPLES_PER_FRAME * 2  # 16-bit samples
                        for i in range(0, len(audio_data), chunk_size):
                            chunk = audio_data[i:i+chunk_size]
                            if len(chunk) < chunk_size:
                                # Pad with silence
                                chunk = chunk + bytes(chunk_size - len(chunk))
                            
                            try:
                                # Use asyncio.to_thread to safely access threading queue from async context
                                await asyncio.to_thread(self.audio_bridge.playback_queue.put_nowait, chunk)
                                packet_count += 1
                                
                                # Debug logging for first few packets
                                if packet_count <= 10 or packet_count % 50 == 0:
                                    rms = AudioAnalyzer.calculate_rms(chunk)
                                    logger.info(f"🔊 Queued playback chunk #{packet_count}, size: {len(chunk)}, RMS: {rms:.1f}")
                                    
                                # Log every 100th packet to track progress
                                if packet_count % 100 == 0:
                                    logger.info(f"🔊 Playback queue: {packet_count} chunks queued")
                                    
                            except queue.Full:
                                # Drop oldest if queue is full
                                try:
                                    await asyncio.to_thread(self.audio_bridge.playback_queue.get_nowait)
                                    await asyncio.to_thread(self.audio_bridge.playback_queue.put_nowait, chunk)
                                    logger.warning(f"⚠️ Playback queue full, dropped oldest chunk")
                                except:
                                    pass
                        
                        if packet_count % 100 == 0:
                            logger.debug(f"📥 Bridged {packet_count} packets from AI")
                            
                    except asyncio.TimeoutError:
                        pass
                        
                await asyncio.sleep(0.001)
                
            except Exception as e:
                logger.error(f"❌ Playback bridge error: {e}")
                await asyncio.sleep(0.1)
        
        logger.info(f"🛑 Playback bridge stopped ({packet_count} packets)")
    
    async def _monitor_call_state(self):
        """Monitor call state and handle disconnection"""
        while self.call_active:
            await asyncio.sleep(1)
        
        logger.info("📵 Call ended, stopping bridges")
    
    def _stop_ai_integration(self):
        """Stop the AI integration"""
        logger.info("🛑 Stopping AI integration")
        self.call_active = False
        
        # Stop audio bridge
        if self.audio_bridge:
            self.audio_bridge.stop()
        
        # Stop async loop
        if self.async_loop and self.async_loop.is_running():
            self.async_loop.call_soon_threadsafe(self.async_loop.stop)
    
    def _timeout_call(self):
        """Timeout the call after maximum duration"""
        logger.info("⏰ Call timeout reached - hanging up")
        try:
            call_prm = pj.CallOpParam()
            self.hangup(call_prm)
        except Exception as e:
            logger.error(f"❌ Error hanging up: {e}")

# ============================================================================
# SIP ACCOUNT
# ============================================================================

class AISipAccount(pj.Account):
    """SIP account with AI integration"""
    
    def __init__(self, config: BridgeConfig):
        pj.Account.__init__(self)
        self.config = config
        self.active_calls: Dict[int, AICall] = {}
    
    def onRegState(self, prm):
        """Handle registration state changes"""
        try:
            ai = self.getInfo()
            logger.info(f"📞 SIP Registration: {ai.regStatusText}")
            
            if ai.regIsActive:
                logger.info("✅ Successfully registered with SIP server")
            else:
                logger.warning("⚠️ SIP registration failed or expired")
                
        except Exception as e:
            logger.error(f"❌ Error in onRegState: {e}")
    
    def onIncomingCall(self, prm):
        """Handle incoming calls"""
        try:
            logger.info("📞 INCOMING CALL")
            
            # Create call handler
            call = AICall(self, self.config, prm.callId)
            self.active_calls[prm.callId] = call
            
            # Auto-answer if configured
            if self.config.AUTO_ANSWER:
                call_prm = pj.CallOpParam()
                call_prm.statusCode = 200
                call.answer(call_prm)
                logger.info("✅ Call auto-answered")
            else:
                logger.info("⏳ Call ringing - manual answer required")
            
            return call
            
        except Exception as e:
            logger.error(f"❌ Error in onIncomingCall: {e}")
            return None

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class SIPBridge:
    """Main SIP to WebSocket bridge application"""
    
    def __init__(self, config: BridgeConfig):
        self.config = config
        self.endpoint: Optional[pj.Endpoint] = None
        self.account: Optional[AISipAccount] = None
        self.running = False
    
    def initialize(self) -> bool:
        """Initialize PJSUA2 endpoint"""
        try:
            logger.info("🚀 Initializing SIP Bridge")
            
            # Create endpoint
            self.endpoint = pj.Endpoint()
            self.endpoint.libCreate()
            
            # Configure endpoint
            ep_cfg = pj.EpConfig()
            ep_cfg.logConfig.level = 3
            ep_cfg.logConfig.consoleLevel = 3
            
            # Audio configuration
            ep_cfg.medConfig.clockRate = self.config.SIP_SAMPLE_RATE
            ep_cfg.medConfig.sndClockRate = self.config.SIP_SAMPLE_RATE
            ep_cfg.medConfig.channelCount = self.config.CHANNELS
            ep_cfg.medConfig.audioFramePtime = 20  # 20ms frames
            ep_cfg.medConfig.sndRecLatency = 20
            ep_cfg.medConfig.sndPlayLatency = 100
            ep_cfg.medConfig.quality = 10  # Maximum quality
            
            # Initialize library
            self.endpoint.libInit(ep_cfg)
            logger.info(f"📚 PJSUA2 Version: {self.endpoint.libVersion().full}")
            
            # Create UDP transport
            transport_cfg = pj.TransportConfig()
            transport_cfg.port = self.config.SIP_TRANSPORT_PORT
            
            try:
                self.endpoint.transportCreate(pj.PJSIP_TRANSPORT_UDP, transport_cfg)
                logger.info(f"✅ SIP transport created on port {transport_cfg.port}")
            except:
                # Try alternative port if primary fails
                transport_cfg.port = self.config.SIP_TRANSPORT_PORT + 1
                self.endpoint.transportCreate(pj.PJSIP_TRANSPORT_UDP, transport_cfg)
                logger.info(f"✅ SIP transport created on alternative port {transport_cfg.port}")
            
            # Start the library
            self.endpoint.libStart()
            logger.info("✅ PJSUA2 library started")
            
            # Create SIP account
            self._create_account()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_account(self):
        """Create and configure SIP account"""
        try:
            # Account configuration
            acc_cfg = pj.AccountConfig()
            
            # SIP URI
            acc_cfg.idUri = f"sip:{self.config.SIP_USER}@{self.config.SIP_DOMAIN}"
            
            # Registration
            acc_cfg.regConfig.registrarUri = f"sip:{self.config.SIP_DOMAIN}:{self.config.SIP_PORT}"
            acc_cfg.regConfig.timeoutSec = 300
            acc_cfg.regConfig.retryIntervalSec = 30
            acc_cfg.regConfig.firstRetryIntervalSec = 5
            
            # Authentication
            cred = pj.AuthCredInfo(
                "digest",
                "*",  # Realm (use * for any)
                self.config.SIP_USER,
                0,  # Data type (0 = plaintext password)
                self.config.SIP_PASSWORD
            )
            acc_cfg.sipConfig.authCreds.append(cred)
            
            # Proxy
            acc_cfg.sipConfig.proxies.append(
                f"sip:{self.config.SIP_DOMAIN}:{self.config.SIP_PORT};transport=udp"
            )
            
            # Create account
            self.account = AISipAccount(self.config)
            self.account.create(acc_cfg)
            
            logger.info(f"✅ SIP account created: {self.config.SIP_USER}@{self.config.SIP_DOMAIN}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create SIP account: {e}")
            raise
    
    def run(self):
        """Run the main event loop"""
        try:
            self.running = True
            
            logger.info("\n" + "=" * 60)
            logger.info("🤖 AI-ENHANCED SIP BRIDGE READY")
            logger.info(f"📞 SIP: {self.config.SIP_USER}@{self.config.SIP_DOMAIN}")
            logger.info(f"🌐 WebSocket: {self.config.WS_URI}")
            # FIX: Use the correct attribute names
            logger.info(f"🎵 Audio: {self.config.SIP_SAMPLE_RATE}Hz ↔ {self.config.AI_INPUT_RATE}/{self.config.AI_OUTPUT_RATE}Hz")
            logger.info("⏳ Waiting for incoming calls...")
            logger.info("⌨️ Press Ctrl+C to stop")
            logger.info("=" * 60 + "\n")
            
            # Main event loop
            while self.running:
                self.endpoint.libHandleEvents(10)  # Handle events with 10ms timeout
                time.sleep(0.001)  # Small sleep to prevent CPU spinning
                
        except KeyboardInterrupt:
            logger.info("\n⌨️ Keyboard interrupt received")
        except Exception as e:
            logger.error(f"❌ Runtime error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown()

    
    def shutdown(self):
        """Shutdown the bridge gracefully"""
        logger.info("🛑 Shutting down SIP Bridge")
        self.running = False
        
        try:
            # Destroy account
            if self.account:
                self.account.shutdown()
                del self.account
                logger.info("✅ SIP account destroyed")
            
            # Destroy endpoint
            if self.endpoint:
                self.endpoint.libDestroy()
                del self.endpoint
                logger.info("✅ PJSUA2 library destroyed")
            
            logger.info("✅ Shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")

# ============================================================================
# ENTRY POINT
# ============================================================================

def print_banner():
    """Print application banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║          SIP TO WEBSOCKET AI AGENT BRIDGE v2.0              ║
║                    Complete Rewrite                          ║
╠══════════════════════════════════════════════════════════════╣
║  This bridge connects SIP phone calls to your AI agent      ║
║  via WebSocket with proper audio streaming                  ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """Main entry point"""
    print_banner()
    
    # Create configuration
    config = BridgeConfig()
    
    # Override with any command-line arguments or environment variables here
    # For example:
    # import os
    # config.WS_URI = os.getenv('AI_WEBSOCKET_URI', config.WS_URI)
    
    # Create and run bridge
    bridge = SIPBridge(config)
    
    if bridge.initialize():
        bridge.run()
    else:
        logger.error("❌ Failed to initialize bridge")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())