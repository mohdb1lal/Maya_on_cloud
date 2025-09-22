import asyncio
import json
import time
import websockets
from collections import deque
import pyaudio
import threading
import numpy as np
from datetime import datetime

# Test Configuration
TEST_SERVER = "ws://localhost:8081"
# TEST_SERVER = "ws://13.233.41.221:8081"
SAMPLE_RATE = 24000
INPUT_SAMPLE_RATE = 16000  # Microphone sample rate
CHUNK_SIZE = 640
SIMULATED_DTMF = "5"

class DebugLiveAudioClient:
    def __init__(self):
        self.audio_queue = deque()
        self.pt = pyaudio.PyAudio()
        self.running = False
        self.audio_sent_count = 0
        self.audio_received_count = 0
        self.last_audio_sent = None
        self.last_audio_received = None
        
        print("🎤 Initializing Audio Client with Debug Logging")
        print(f"📊 Input Sample Rate: {INPUT_SAMPLE_RATE}Hz, Output: {SAMPLE_RATE}Hz, Chunk Size: {CHUNK_SIZE}")
        
        # List available audio devices for debugging
        self.list_audio_devices()
        
        try:
            # Input stream (microphone)
            self.input_stream = self.pt.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=INPUT_SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=self._audio_input_callback
            )
            print("✅ Microphone input stream initialized")
            
            # Output stream (speakers)
            self.output_stream = self.pt.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                output=True,
                frames_per_buffer=CHUNK_SIZE
            )
            print("✅ Speaker output stream initialized")
            
        except Exception as e:
            print(f"❌ Audio initialization failed: {e}")
            raise

    def _audio_input_callback(self, in_data, frame_count, time_info, status):
        """Callback for microphone input with debugging"""
        if status:
            print(f"⚠️ Audio input status: {status}")
            
        if self.running:
            # Calculate RMS to see audio levels
            try:
                audio_np = np.frombuffer(in_data, dtype=np.int16)
                rms = np.sqrt(np.mean(np.square(audio_np.astype(np.float32))))
                
                # Only log if there's actual audio (above very low threshold)
                if rms > 10:
                    print(f"🎙️ MIC INPUT: {len(in_data)} bytes, RMS: {rms:.1f}, Frame Count: {frame_count}")
                
                self.audio_queue.append(in_data)
                self.last_audio_received = datetime.now()
                
            except Exception as e:
                print(f"❌ Error processing input audio: {e}")
                # Still queue the audio even if RMS calculation fails
                self.audio_queue.append(in_data)
                
        return (None, pyaudio.paContinue)

    async def connect_to_server(self):
        """Connect to the WebSocket server and start live audio with detailed debugging"""
        print(f"\n🌐 Connecting to {TEST_SERVER}...")
        
        try:
            async with websockets.connect(TEST_SERVER) as ws:
                print("✅ WebSocket connection established")
                await self._start_call(ws)
                
                # Start the session
                print("🚀 Starting audio streaming session...")
                self.running = True
                self.input_stream.start_stream()
                print("✅ Microphone stream started")
                
                # Start all tasks concurrently
                recv_task = asyncio.create_task(self._receive_messages(ws))
                send_task = asyncio.create_task(self._send_live_audio(ws))
                dtmf_task = asyncio.create_task(self._simulate_dtmf_delayed(ws))
                stats_task = asyncio.create_task(self._print_stats())
                
                print("\n" + "="*60)
                print("🎯 LIVE AUDIO SESSION ACTIVE")
                print("📢 Speak into your microphone now!")
                print("📊 Watch the logs below for audio flow")
                print("⌨️ Press Ctrl+C to stop")
                print("="*60 + "\n")
                
                # Run until interrupted
                try:
                    await asyncio.gather(recv_task, send_task, dtmf_task, stats_task)
                except KeyboardInterrupt:
                    print("\n🛑 Stopping audio session...")
                    
                await self._end_call(ws)
                
        except websockets.exceptions.ConnectionClosed:
            print("❌ Connection closed by server")
        except Exception as e:
            print(f"❌ Connection error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
            print("🔄 Session ended")

    async def _start_call(self, ws):
        """Send call initialization messages with debugging"""
        print("📞 Sending call start message...")
        
        start_msg = {
            "event": "start",
            "call_id": "debug_call_001",
            "caller_id": "+15551234567"
        }
        await ws.send(json.dumps(start_msg))
        print(f"📤 Sent: {start_msg}")
        
        media_msg = {
            "event": "media",
            "data": {
                "sample_rate": SAMPLE_RATE,
                "codec": "L16"
            }
        }
        await ws.send(json.dumps(media_msg))
        print(f"📤 Sent: {media_msg}")
        print("✅ Call initialization complete")

    async def _send_live_audio(self, ws):
        """Continuously send microphone audio to server with detailed debugging"""
        print("🎤 Starting audio transmission loop...")
        last_log_time = time.time()
        chunks_since_log = 0
        
        try:
            while self.running:
                if self.audio_queue:
                    audio_chunk = self.audio_queue.popleft()
                    
                    # Calculate RMS for debugging
                    try:
                        audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
                        rms = np.sqrt(np.mean(np.square(audio_np.astype(np.float32))))
                        
                        # Send the audio
                        await ws.send(audio_chunk)
                        self.audio_sent_count += 1
                        self.last_audio_sent = datetime.now()
                        chunks_since_log += 1
                        
                        # Log every chunk if RMS is significant, or every 10 chunks if quiet
                        current_time = time.time()
                        if rms > 100 or current_time - last_log_time > 2:
                            print(f"📤 SENT AUDIO #{self.audio_sent_count}: {len(audio_chunk)} bytes, RMS: {rms:.1f}")
                            if chunks_since_log > 1:
                                print(f"   📊 ({chunks_since_log} chunks sent in last {current_time - last_log_time:.1f}s)")
                            last_log_time = current_time
                            chunks_since_log = 0
                            
                    except Exception as e:
                        print(f"❌ Error processing outgoing audio: {e}")
                        # Send anyway
                        await ws.send(audio_chunk)
                        self.audio_sent_count += 1
                        
                else:
                    await asyncio.sleep(0.001)  # Small delay to prevent busy waiting
                    
        except websockets.exceptions.ConnectionClosed:
            print("❌ WebSocket closed during audio transmission")
        except Exception as e:
            print(f"❌ Error sending audio: {e}")
            import traceback
            traceback.print_exc()

    async def _simulate_dtmf_delayed(self, ws):
        """Send simulated DTMF event after delay"""
        print(f"⏰ DTMF '{SIMULATED_DTMF}' will be sent in 10 seconds...")
        await asyncio.sleep(10)
        
        if self.running:
            dtmf_msg = {
                "event": "dtmf",
                "digit": SIMULATED_DTMF
            }
            await ws.send(json.dumps(dtmf_msg))
            print(f"📞 Sent DTMF: {SIMULATED_DTMF}")

    async def _receive_messages(self, ws):
        """Handle incoming messages from server with detailed debugging"""
        print("📥 Starting message reception loop...")
        
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    # Audio response from server
                    self.audio_received_count += 1
                    
                    try:
                        # Calculate RMS of received audio
                        audio_np = np.frombuffer(message, dtype=np.int16)
                        rms = np.sqrt(np.mean(np.square(audio_np.astype(np.float32))))
                        
                        print(f"🔊 RECEIVED AUDIO #{self.audio_received_count}: {len(message)} bytes, RMS: {rms:.1f}")
                        
                        # Play through speakers
                        self.output_stream.write(message)
                        print("🔈 Audio played through speakers")
                        
                    except Exception as e:
                        print(f"❌ Error processing received audio: {e}")
                        # Try to play anyway
                        try:
                            self.output_stream.write(message)
                        except:
                            print("❌ Failed to play audio through speakers")
                else:
                    # JSON event from server
                    try:
                        event = json.loads(message)
                        print(f"📨 RECEIVED EVENT: {event}")
                        
                        if event.get("event") == "transfer":
                            print(f"📞 Call transfer requested to: {event.get('department', 'unknown')}")
                        elif event.get("event") == "ready":
                            print("✅ Server is ready for audio")
                        elif event.get("event") == "started":
                            print("✅ Call started confirmation")
                        elif event.get("event") == "audio_complete":
                            print("🔊 Server finished playing audio")
                            
                    except json.JSONDecodeError:
                        print(f"📨 Received non-JSON message: {message[:100]}...")
                        
        except websockets.exceptions.ConnectionClosed:
            print("❌ Connection closed by server during message reception")
        except Exception as e:
            print(f"❌ Error receiving messages: {e}")
            import traceback
            traceback.print_exc()

    async def _print_stats(self):
        """Print periodic statistics about audio flow"""
        while self.running:
            await asyncio.sleep(5)  # Print stats every 5 seconds
            
            current_time = datetime.now()
            print(f"\n📊 === AUDIO STATS ===")
            print(f"📤 Audio chunks sent: {self.audio_sent_count}")
            print(f"📥 Audio chunks received: {self.audio_received_count}")
            print(f"🎤 Queue size: {len(self.audio_queue)}")
            
            if self.last_audio_sent:
                seconds_since_sent = (current_time - self.last_audio_sent).total_seconds()
                print(f"⏰ Last audio sent: {seconds_since_sent:.1f}s ago")
                
            if self.last_audio_received:
                seconds_since_received = (current_time - self.last_audio_received).total_seconds()
                print(f"⏰ Last mic input: {seconds_since_received:.1f}s ago")
                
            print("========================\n")

    async def _end_call(self, ws):
        """Terminate the call"""
        try:
            end_msg = {
                "event": "hangup",
                "reason": "normal"
            }
            await ws.send(json.dumps(end_msg))
            print(f"📤 Sent hangup: {end_msg}")
        except Exception as e:
            print(f"❌ Error sending hangup: {e}")

    def cleanup(self):
        """Release audio resources"""
        print("🧹 Cleaning up audio resources...")
        self.running = False
        
        try:
            if self.input_stream.is_active():
                self.input_stream.stop_stream()
            self.input_stream.close()
            print("✅ Input stream closed")
        except Exception as e:
            print(f"❌ Error closing input stream: {e}")
        
        try:
            if self.output_stream.is_active():
                self.output_stream.stop_stream()
            self.output_stream.close()
            print("✅ Output stream closed")
        except Exception as e:
            print(f"❌ Error closing output stream: {e}")
        
        try:
            self.pt.terminate()
            print("✅ PyAudio terminated")
        except Exception as e:
            print(f"❌ Error terminating PyAudio: {e}")
            
        print(f"📊 Final stats: {self.audio_sent_count} chunks sent, {self.audio_received_count} chunks received")

    def list_audio_devices(self):
        """List available audio devices for debugging"""
        print("\n🎵 Available Audio Devices:")
        try:
            for i in range(self.pt.get_device_count()):
                info = self.pt.get_device_info_by_index(i)
                device_type = []
                if info['maxInputChannels'] > 0:
                    device_type.append(f"IN:{info['maxInputChannels']}")
                if info['maxOutputChannels'] > 0:
                    device_type.append(f"OUT:{info['maxOutputChannels']}")
                    
                print(f"  Device {i}: {info['name']} - {', '.join(device_type)}")
                print(f"           Sample Rate: {info['defaultSampleRate']}")
        except Exception as e:
            print(f"❌ Error listing audio devices: {e}")
        print()

async def main():
    client = DebugLiveAudioClient()
    
    try:
        await client.connect_to_server()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Main error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.cleanup()

if __name__ == "__main__":
    print("=" * 80)
    print("🔍 DEBUG LIVE AUDIO WEBSOCKET CLIENT")
    print("=" * 80)
    print("🎤 This client will show detailed logs of all audio operations")
    print("📊 Watch for:")
    print("   - MIC INPUT logs (showing microphone is picking up audio)")
    print("   - SENT AUDIO logs (showing audio is being transmitted)")
    print("   - RECEIVED AUDIO logs (showing server responses)")
    print("   - Periodic stats every 5 seconds")
    print()
    print("🔧 Make sure your microphone and speakers are working")
    print("🗣️ Speak clearly into your microphone during the test")
    print("=" * 80)
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Exiting gracefully...")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()