# Final code
import asyncio
import logging
import traceback
import numpy as np
import json
import base64
import time
from dataclasses import dataclass
import websockets
from websockets.server import serve
from google import genai
from google.genai.types import (
    LiveConnectConfig,
    SpeechConfig,
    VoiceConfig,
    PrebuiltVoiceConfig,
    FunctionDeclaration,
    Tool,
)
from scipy import signal
from datetime import datetime
import pytz  # Add this for timezone support
from firebase_data_fetching.doctors import fetch_clinic_data

# --- Configuration ---
@dataclass
class Config:
    WS_HOST = "0.0.0.0"
    WS_PORT = 8081
    PROJECT_ID = "maya-zappq"
    LOCATION = "us-central1"
    # MODEL = "gemini-2.0-flash-live-preview-04-09"
    MODEL = "gemini-live-2.5-flash-preview-native-audio-09-2025"
    INPUT_SAMPLE_RATE = 16000
    OUTPUT_SAMPLE_RATE = 24000
    CHUNK_SIZE = 640
    VAD_THRESHOLD = 300  # Balanced threshold
    MIN_SPEECH_DURATION = 0.6  # Reasonable minimum for real speech
    SILENCE_DURATION = 1.2  # Good for natural pauses
    RESPONSE_TIMEOUT = 10.0
    # Audio quality settings
    AUDIO_BUFFER_SIZE = 8192
    FADE_DURATION = 0.01
    NOISE_GATE_THRESHOLD = 80  # Moderate noise gate
    # Echo cancellation settings
    ECHO_BLOCK_DURATION = 1.0  # Reduced to 1 second
    MIN_USER_PAUSE = 0.2  # Reduced minimum pause

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('freeswitch_websocket_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('FreeSwitchWebSocket')


######################################################################
# --- NEW: Hospital and Doctor Data Loading ---

# Define the ID of the clinic this instance of MAYA is for.
CLINIC_ID = "QV070" 

logger.info(f"Fetching clinic and doctor details for Clinic ID '{CLINIC_ID}'...")

# Initialize the four data structures that will hold our clinic's information.
DEPARTMENT_WISE_DOCTORS = {}
DOCTOR_CONSULTATION_FEE = {}
DOCTOR_AVAILABILITY = {}
ALL_DOCTORS = []
CLINIC_NAME = "the hospital" # Default name

try:
    # Call the function from doctors.py and unpack the new 5-item tuple
    (
        clinic_name_fetched,
        depts_fetched,
        fees_fetched,
        avail_fetched,
        docs_fetched
    ) = fetch_clinic_data(CLINIC_ID)

    # If data was fetched successfully, assign it to our global variables
    if docs_fetched:
        # Assign the fetched clinic name. Fall back to default if it's empty.
        CLINIC_NAME = clinic_name_fetched or "the hospital"
        DEPARTMENT_WISE_DOCTORS = depts_fetched
        DOCTOR_CONSULTATION_FEE = fees_fetched
        DOCTOR_AVAILABILITY = avail_fetched
        ALL_DOCTORS = docs_fetched
        
        logger.info(f"Successfully configured for '{CLINIC_NAME}' (ID: {CLINIC_ID}) with {len(ALL_DOCTORS)} doctors.")
    else:
        # This will be triggered if fetch_clinic_data returns None
        raise ValueError("fetch_clinic_data did not return any doctor data.")

except Exception as e:
    logger.error(
        f"Could not fetch doctor details for Clinic ID '{CLINIC_ID}'. "
        f"The agent will have limited knowledge. Error: {e}"
    )
######################################################################


# --- Audio Processing Utilities ---
class AudioProcessor:
    """Enhanced audio processing with noise reduction and quality improvements"""
    
    @staticmethod
    def apply_fade(audio_data: np.ndarray, sample_rate: int, fade_in: bool = True) -> np.ndarray:
        """Apply fade in/out to prevent clicking sounds"""
        # Make a writable copy if the array is read-only
        if not audio_data.flags.writeable:
            audio_data = audio_data.copy()
            
        fade_samples = int(Config.FADE_DURATION * sample_rate)
        if fade_samples > len(audio_data):
            fade_samples = len(audio_data)
        
        fade_curve = np.linspace(0, 1, fade_samples) if fade_in else np.linspace(1, 0, fade_samples)
        
        if fade_in:
            audio_data[:fade_samples] = audio_data[:fade_samples] * fade_curve
        else:
            audio_data[-fade_samples:] = audio_data[-fade_samples:] * fade_curve
        
        return audio_data
    
    @staticmethod
    def apply_noise_gate(audio_data: np.ndarray, threshold: float = Config.NOISE_GATE_THRESHOLD) -> np.ndarray:
        """Apply noise gate to reduce low-level noise"""
        # Make a writable copy if needed
        if not audio_data.flags.writeable:
            audio_data = audio_data.copy()
            
        rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
        if rms < threshold:
            # Gradually reduce to silence instead of hard cut
            return audio_data * (rms / threshold)
        return audio_data
    
    @staticmethod
    def apply_bandpass_filter(audio_data: np.ndarray, sample_rate: int, 
                             lowcut: float = 80, highcut: float = 8000) -> np.ndarray:
        """Apply bandpass filter to remove unwanted frequencies"""
        nyquist = sample_rate * 0.5
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Ensure valid frequency range
        if low >= 1.0:
            low = 0.99
        if high >= 1.0:
            high = 0.99
        if low >= high:
            return audio_data
        
        try:
            sos = signal.butter(4, [low, high], btype='band', output='sos')
            filtered = signal.sosfilt(sos, audio_data.astype(np.float64))
            return filtered.astype(audio_data.dtype)
        except Exception as e:
            logger.warning(f"Bandpass filter failed: {e}")
            return audio_data
    
    @staticmethod
    def resample_audio(audio_data: bytes, input_rate: int, output_rate: int) -> bytes:
        """High-quality audio resampling with anti-aliasing"""
        if input_rate == output_rate:
            return audio_data
        
        try:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            
            # Calculate resampling ratio
            ratio = output_rate / input_rate
            new_length = int(len(audio_np) * ratio)
            
            # Use high-quality resampling
            resampled = signal.resample_poly(audio_np, output_rate, input_rate)
            
            # Convert back to int16 with proper clipping
            resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
            
            return resampled.tobytes()
        except Exception as e:
            logger.error(f"Resampling error: {e}")
            return audio_data
    
    @staticmethod
    def normalize_audio(audio_data: np.ndarray, target_level: float = 0.8) -> np.ndarray:
        """Normalize audio to consistent volume level"""
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            scale_factor = (target_level * 32767) / max_val
            if scale_factor < 1.0:  # Only normalize if needed
                return (audio_data * scale_factor).astype(np.int16)
        return audio_data

class VoiceActivityDetector:
    def __init__(self, threshold=Config.VAD_THRESHOLD, min_duration=Config.MIN_SPEECH_DURATION, 
                 silence_duration=Config.SILENCE_DURATION):
        self.threshold = threshold
        self.min_duration = min_duration
        self.silence_duration = silence_duration
        self.reset_state()
        self.audio_processor = AudioProcessor()
        
    def reset_state(self):
        """Reset all VAD state variables"""
        self.speech_start = None
        self.last_speech_time = None
        self.is_speaking = False
        self.speech_just_ended = False
        self.energy_history = []
        self.consecutive_silence_frames = 0
        self.min_silence_frames = int(self.silence_duration * 50)  # 50 frames per second
        logger.debug("VAD state reset")

    def is_speech(self, audio_data: bytes) -> tuple[bool, bool]:
        """
        Returns (is_currently_speech, speech_just_ended)
        """
        try:
            # Ensure even number of bytes for 16-bit samples
            if len(audio_data) % 2 != 0:
                audio_data = audio_data[:-1]
                
            if len(audio_data) == 0:
                return False, False

            # Convert to 16-bit PCM
            audio = np.frombuffer(audio_data, dtype=np.int16)
            
            # Apply noise gate first
            audio = self.audio_processor.apply_noise_gate(audio)
            
            # Calculate RMS energy
            rms = np.sqrt(np.mean(np.square(audio.astype(np.float32))))
            
            # Adaptive threshold based on recent energy history
            self.energy_history.append(rms)
            if len(self.energy_history) > 100:
                self.energy_history.pop(0)
            
            # Use adaptive threshold if we have enough history
            if len(self.energy_history) > 20:
                noise_floor = np.percentile(self.energy_history, 20)
                adaptive_threshold = max(self.threshold, noise_floor * 2)
            else:
                adaptive_threshold = self.threshold
            
            now = time.time()
            speech_detected = rms > adaptive_threshold
            speech_just_ended = False
            
            if speech_detected:
                self.consecutive_silence_frames = 0
                
                if not self.is_speaking:
                    # Start of speech
                    self.speech_start = now
                    self.is_speaking = True
                    logger.info(f"🎤 Speech started (RMS: {rms:.1f}, Threshold: {adaptive_threshold:.1f})")
                
                self.last_speech_time = now
                
            else:
                # No speech detected
                self.consecutive_silence_frames += 1
                
                if self.is_speaking and self.consecutive_silence_frames >= self.min_silence_frames:
                    # End of speech detected
                    speech_duration = now - self.speech_start if self.speech_start else 0
                    
                    if speech_duration >= self.min_duration:
                        self.is_speaking = False
                        speech_just_ended = True
                        logger.info(f"🎤 Speech ended after {speech_duration:.1f}s")
                    else:
                        logger.debug(f"Speech too short ({speech_duration:.1f}s), ignoring")
                        self.is_speaking = False
            
            # Only return True for is_speech if we've been speaking for minimum duration
            is_valid_speech = False
            if self.is_speaking and self.speech_start:
                current_duration = now - self.speech_start
                is_valid_speech = current_duration >= self.min_duration
            
            return is_valid_speech, speech_just_ended
            
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return False, False





# --- Enhanced Audio Buffer Manager ---
class AudioBufferManager:
    """Manages audio buffering with proper synchronization and quality control"""
    
    def __init__(self, buffer_size: int = Config.AUDIO_BUFFER_SIZE):
        self.buffer = bytearray()
        self.buffer_size = buffer_size
        self.lock = asyncio.Lock()
        self.audio_processor = AudioProcessor()
        
    async def add_audio(self, audio_data: bytes):
        """Add audio to buffer with overflow protection"""
        async with self.lock:
            self.buffer.extend(audio_data)
            
            # Prevent buffer overflow
            if len(self.buffer) > self.buffer_size * 2:
                # Keep only the most recent data
                self.buffer = self.buffer[-self.buffer_size:]
                logger.warning("Audio buffer overflow - trimming old data")
    
    async def get_chunk(self, chunk_size: int) -> bytes:
        """Get a properly processed audio chunk"""
        async with self.lock:
            if len(self.buffer) < chunk_size:
                return None
            
            chunk = bytes(self.buffer[:chunk_size])
            self.buffer = self.buffer[chunk_size:]
            return chunk
    
    async def clear(self):
        """Clear the buffer"""
        async with self.lock:
            self.buffer = bytearray()

# --- Enhanced WebSocket Handler ---
class FreeSwitchWebSocketHandler:
    def __init__(self):
        self.websocket = None
        self.gemini_session = None
        self.audio_buffer_manager = AudioBufferManager()
        self.output_buffer_manager = AudioBufferManager()
        self.call_active = False
        self.assistant_speaking = False
        self.user_speaking = False
        self.vad = VoiceActivityDetector()
        self.lock = asyncio.Lock()
        self.call_id = None
        self.caller_id = None
        self._gemini_context_manager = None
        self.speech_buffer = bytearray()
        self.response_timeout = Config.RESPONSE_TIMEOUT
        self.gemini_response_task = None
        self.response_timeout_task = None
        self.last_response_time = None
        self.audio_processor = AudioProcessor()
        self.output_sample_rate = Config.OUTPUT_SAMPLE_RATE
        self.input_sample_rate = Config.INPUT_SAMPLE_RATE
        # Echo cancellation tracking
        self.last_assistant_speech_end = 0
        self.echo_blocking_active = False
        self.consecutive_silence_chunks = 0
        self.audio_energy_baseline = []

    async def handle_freeswitch_connection(self, websocket, path):
        logger.info(f"FreeSwitch connection from {websocket.remote_address}")
        self.websocket = websocket
        
        try:
            await self._send_ready_signal()
            await self._handle_freeswitch_messages()
        except websockets.exceptions.ConnectionClosed:
            logger.info("FreeSwitch connection closed")
        except Exception as e:
            logger.error(f"FreeSwitch handler error: {e}")
            traceback.print_exc()
        finally:
            logger.info("Cleaning up call...")
            await self._cleanup_call()

    def _get_current_datetime(self):
        """Get current date and time in Indian timezone"""
        # Set to Indian Standard Time since you're in Kerala
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)
        return {
            'date': now.strftime('%Y-%m-%d'),
            'time': now.strftime('%H:%M:%S'),
            'day': now.strftime('%A'),
            'formatted': now.strftime('%A, %B %d, %Y at %I:%M %p')
        }

    async def _send_ready_signal(self):
        await self._send_message({
            "event": "ready",
            "data": {
                "protocols": ["audio"],
                "formats": ["L16"],  # Only use L16 (Linear PCM)
                "sample_rates": [16000, 24000],
                "channels": 1  # Mono audio only
            }
        })

    async def _handle_freeswitch_messages(self):
        async for message in self.websocket:
            try:
                if isinstance(message, bytes):
                    await self._handle_audio_stream(message)
                else:
                    data = json.loads(message)
                    await self._handle_control_message(data)
            except json.JSONDecodeError:
                logger.error("Invalid JSON from FreeSwitch")
            except Exception as e:
                logger.error(f"Message processing error: {e}")

    async def _handle_control_message(self, data):
        event_type = data.get("event")
        
        if event_type == "start":
            await self._handle_call_start(data)
        elif event_type == "media":
            await self._handle_media_info(data)
        elif event_type == "hangup":
            await self._handle_call_end(data)
        elif event_type == "audio":
            audio_data = base64.b64decode(data.get("data", ""))
            await self._handle_audio_stream(audio_data)
        else:
            logger.debug(f"Unknown event from FreeSwitch: {event_type}")

    async def _handle_call_start(self, data):
        async with self.lock:
            self.call_id = data.get("call_id")
            self.caller_id = data.get("caller_id")
            self.call_active = True
            await self.audio_buffer_manager.clear()
            await self.output_buffer_manager.clear()
            self.speech_buffer = bytearray()
            self.vad.reset_state()
            self.last_response_time = None
            
            logger.info(f"Call started - ID: {self.call_id}, Caller: {self.caller_id}")
            
            if not await self._start_gemini_session():
                logger.error("Failed to start Gemini session")
                await self._send_hangup()
                return
            
            # Proactively trigger Gemini's welcome message instead of waiting for user input.
            await self._trigger_initial_greeting()
            
            await self._send_message({
                "event": "started",
                "call_id": self.call_id,
                "status": "ready"
            })

    async def _trigger_initial_greeting(self):
        """Prompts Gemini to start the conversation with its configured greeting."""
        try:
            if self.gemini_session:
                logger.info("Triggering Gemini's initial greeting...")
                # By sending an end-of-speech signal, we prompt the AI to speak first.
                # The AI will follow its system instruction to provide a warm welcome.
                await self.gemini_session.send_realtime_input(audio_stream_end=True)
                logger.info("Greeting trigger sent successfully.")
        except Exception as e:
            logger.error(f"Failed to trigger initial greeting: {e}")

    async def _handle_media_info(self, data):
        media_info = data.get("data", {})
        logger.info(f"Media info: {media_info}")
        
        # Update sample rates based on media info
        input_rate = media_info.get("sample_rate", Config.INPUT_SAMPLE_RATE)
        channels = media_info.get("channels", 1)
        
        if channels != 1:
            logger.warning(f"Multi-channel audio detected ({channels} channels). Converting to mono.")
        
        if input_rate != self.input_sample_rate:
            self.input_sample_rate = input_rate
            Config.CHUNK_SIZE = int(input_rate * 0.02)  # 20ms chunks
            logger.info(f"Updated input sample rate to {input_rate}Hz, chunk size to {Config.CHUNK_SIZE}")

    async def _handle_audio_stream(self, audio_data: bytes):
        if not self.call_active or len(audio_data) == 0:
            return

        # Block audio input during and shortly after assistant speech
        current_time = time.time()
        
        if self.assistant_speaking:
            logger.debug("🚫 Blocking audio - assistant currently speaking")
            self.consecutive_silence_chunks = 0
            return
        
        # Echo suppression: block input for a period after assistant stops speaking
        time_since_assistant_spoke = current_time - self.last_assistant_speech_end
        if self.last_assistant_speech_end > 0 and time_since_assistant_spoke < Config.ECHO_BLOCK_DURATION:
            logger.debug(f"🚫 Echo suppression active for {Config.ECHO_BLOCK_DURATION - time_since_assistant_spoke:.1f}s more")
            return

        try:
            # Process audio for quality improvement
            audio_np = np.frombuffer(audio_data, dtype=np.int16).copy()
            
            # Calculate RMS for logging/debugging
            rms = np.sqrt(np.mean(np.square(audio_np.astype(np.float32))))
            
            # Track baseline noise level (but less aggressively)
            if not self.user_speaking and len(self.audio_energy_baseline) < 30:
                self.audio_energy_baseline.append(rms)
                if len(self.audio_energy_baseline) > 30:
                    self.audio_energy_baseline.pop(0)
            
            # Apply bandpass filter to remove unwanted frequencies
            audio_np = self.audio_processor.apply_bandpass_filter(
                audio_np, self.input_sample_rate, lowcut=80, highcut=4000
            )
            
            # Convert back to bytes
            processed_audio = audio_np.astype(np.int16).tobytes()
            
            # Add to buffer
            await self.audio_buffer_manager.add_audio(processed_audio)
            
            # Process chunks
            while True:
                chunk = await self.audio_buffer_manager.get_chunk(Config.CHUNK_SIZE)
                if chunk is None:
                    break
                
                is_speech = self.vad.is_speech(chunk)
                
                if is_speech:
                    self.consecutive_silence_chunks = 0
                    
                    if not self.user_speaking:
                        # Start speech detection with reasonable threshold
                        if rms > Config.VAD_THRESHOLD:
                            self.user_speaking = True
                            logger.info(f"🎤 USER STARTED SPEAKING (RMS: {rms:.1f})")
                            self.speech_buffer = bytearray()
                            self.audio_energy_baseline = []
                    
                    if self.user_speaking:
                        self.speech_buffer.extend(chunk)
                        await self._send_to_gemini(chunk)
                else:
                    self.consecutive_silence_chunks += 1
                    
                    if self.vad.just_ended() and self.user_speaking:
                        # Check minimum speech duration
                        speech_duration = len(self.speech_buffer) / (self.input_sample_rate * 2)
                        if speech_duration >= Config.MIN_SPEECH_DURATION:
                            self.user_speaking = False
                            logger.info(f"🎤 USER FINISHED SPEAKING (duration: {speech_duration:.1f}s)")
                            await self._send_speech_end_signal()
                        else:
                            logger.debug(f"Speech too short ({speech_duration:.1f}s), continuing")
                            self.vad.active = True
                    
        except Exception as e:
            logger.error(f"Audio stream processing error: {e}")

    async def _send_to_gemini(self, audio_chunk: bytes):
        try:
            if self.gemini_session:
                # Ensure audio is at correct sample rate for Gemini
                if self.input_sample_rate != Config.INPUT_SAMPLE_RATE:
                    audio_chunk = self.audio_processor.resample_audio(
                        audio_chunk, self.input_sample_rate, Config.INPUT_SAMPLE_RATE
                    )
                
                await self.gemini_session.send_realtime_input(
                    media={
                        "data": audio_chunk,
                        "mime_type": f"audio/pcm;rate={Config.INPUT_SAMPLE_RATE}",
                    }
                )
            else:
                logger.warning("⚠️ No Gemini session - attempting restart")
                await self._restart_gemini_session()
        except Exception as e:
            logger.error(f"Error sending to Gemini: {e}")
            if "keepalive" in str(e).lower() or "closed" in str(e).lower():
                logger.warning("🔄 Gemini connection lost, attempting restart...")
                await self._restart_gemini_session()

   

    async def _send_speech_end_signal(self):
        """
        Sends a signal to Gemini indicating the user has finished speaking.
        """
        try:
            if self.gemini_session:
                # Option 1: Use the audio_stream_end parameter (recommended)
                await self.gemini_session.send_realtime_input(audio_stream_end=True)
                
            
                
                logger.info("🤐 Sent end-of-speech signal to Gemini")
                
                # Clear the speech buffer after sending the complete turn
                self.speech_buffer = bytearray()

                # Start the timeout handler for the Gemini response
                await self._cancel_task('response_timeout_task')
                self.response_timeout_task = asyncio.create_task(self._response_timeout_handler())

        except Exception as e:
            logger.error(f"Error sending speech end signal: {e}")

    async def _response_timeout_handler(self):
        """Idle-based response timeout handler"""
        try:
            while True:
                await asyncio.sleep(0.5)

                if self.assistant_speaking:
                    continue

                if self.last_response_time is None:
                    continue

                if time.time() - self.last_response_time > self.response_timeout:
                    logger.warning(f"⚠️ Response timeout after {self.response_timeout}s")
                    async with self.lock:
                        self.assistant_speaking = False
                        self.user_speaking = False
                        await self.audio_buffer_manager.clear()
                        self.speech_buffer = bytearray()
                        self.vad.reset_state()
                        logger.info("✅ System reset due to timeout")
                    return

        except asyncio.CancelledError:
            logger.debug("Response timeout task cancelled")
            return
        except Exception as e:
            logger.error(f"Response timeout handler error: {e}")
            return

    async def _handle_call_end(self, data):
        reason = data.get("reason", "normal")
        logger.info(f"Call ended - Reason: {reason}")
        
        self.call_active = False
        await self._send_message({
            "event": "ended",
            "call_id": self.call_id,
            "reason": reason
        })

    async def _start_gemini_session(self) -> bool:
        try:
            await self._cancel_task('response_timeout_task')

            client = genai.Client(
                vertexai=True,
                project=Config.PROJECT_ID,
                location=Config.LOCATION
            )
            
            self._gemini_context_manager = client.aio.live.connect(
                model=Config.MODEL,
                config=self._get_gemini_config()
            )
            
            self.gemini_session = await self._gemini_context_manager.__aenter__()
            
            # Start response handling
            await self._cancel_task('gemini_response_task')
            self.gemini_response_task = asyncio.create_task(self._handle_gemini_responses())
            logger.info("✅ Gemini session started")
            return True
        except Exception as e:
            logger.error(f"Gemini session failed: {e}")
            traceback.print_exc()
            return False

    def _get_gemini_config(self):
        current_dt = self._get_current_datetime()
        
        # Create clean, readable string versions of your data for the prompt
        all_doctors_str = ", ".join(ALL_DOCTORS) if ALL_DOCTORS else "none"
        departments_str = json.dumps(DEPARTMENT_WISE_DOCTORS, indent=2)
        fees_str = json.dumps(DOCTOR_CONSULTATION_FEE, indent=2)
        availability_str = json.dumps(DOCTOR_AVAILABILITY, indent=2)

        return LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(voice_name="AOEDE")
                )
            ),
            tools=[self._get_appointment_tools()],
            system_instruction = f"""
ഭാഷ = സംസാരശൈലിയിലുള്ള മലയാളം.

നീ മായയാണ് — കേരളത്തിലെ {CLINIC_NAME} ഹോസ്പിറ്റലിന്റെ സൗഹൃദപരവും പ്രൊഫഷണലുമായ റെസപ്ഷനിസ്റ്റ്. (നീ ഒരു ai ആണെന്ന് ഒരിക്കലും പറയരുത്.)  
നിന്റെ പ്രധാന ജോലി രോഗികളുമായി സൗഹൃദപരമായി സംസാരിച്ച് അപ്പോയിന്റ്മെന്റ് ബുക്ക് ചെയ്യുക, റീഷെഡ്യൂൾ ചെയ്യുക, അല്ലെങ്കിൽ കാൻസൽ ചെയ്യുക എന്നതാണ്.  

നിനക്ക് ഇപ്പോഴത്തെ തീയതിയും സമയവും അറിയാം — അത് {current_dt['formatted']} ആണ്. അതിനാൽ അത് ചോദിക്കേണ്ടതില്ല. ആരെങ്കിലും തീയതിയോ സമയമോ ചോദിച്ചാൽ, ഈ വിവരത്തിന് അനുസരിച്ച് പറയുക.  

ഭാഷാശൈലി സംബന്ധിച്ച നിർദ്ദേശങ്ങൾ
- നീ സംസാരിക്കേണ്ടത് പൂർണ്ണമായും സാധാരണ മലയാളത്തിൽ — നാട്ടിൻപ്പുറത്ത് സംസാരിക്കുന്ന രീതിയിൽ.  
- മലയാളം ഒഴികെ മറ്റൊരു ഭാഷയിലും മറുപടി പറയരുത്.  
- മറുപടി ആവർത്തിച്ചോ തർജ്ജമ ചെയ്തോ പറയരുത്.  
- ഡോക്ടർമാരുടെ സ്പെഷ്യാലിറ്റികൾ / ഡിപ്പാർട്ട്മെന്റുകൾ പറയുമ്പോൾ ആ ഭാഗങ്ങൾ ഇംഗ്ലീഷിൽ തന്നെ ഉപയോഗിക്കുക.  
  _(ഉദാ: “Orthopaedics”, “Dermatology” എന്നിവ)_  

നടത്തം സംബന്ധിച്ച പ്രധാന നിർദ്ദേശങ്ങൾ
- മനുഷ്യൻ പോലെ സ്വാഭാവികമായി സംസാരിക്കുക — casual, friendly, helpful ടോണിൽ.  
- മലയാളത്തിൽ മാത്രം സംസാരിക്കുക.  
- ആരെങ്കിലും നേരിട്ട് “അപ്പോയിന്റ്മെന്റ് ബുക്ക് ചെയ്യണം” എന്ന് പറഞ്ഞാൽ, **ഗ്രീറ്റിംഗ് ഒഴിവാക്കി** നേരെ ബുക്കിംഗിലേക്ക് പോകുക.  
- ആരെങ്കിലും പ്രത്യേക ഡോക്ടറെ ചോദിച്ചാൽ (ഉദാ: “Dr. Priya Sharma”), ആദ്യം ആ ഡോക്ടർ available ആണോ എന്ന് പരിശോധിക്കാൻ `check_availability` function ഉപയോഗിക്കുക.  
  - available ആണെങ്കിൽ appointment confirm ചെയ്യുക.  
  - available അല്ലെങ്കിൽ അത് പറയുകയും, വേറെ ഡോക്ടറുമായി ബുക്ക് ചെയ്യണോ എന്ന് വിനയത്തോടെ ചോദിക്കുകയും ചെയ്യുക.  
- രോഗിയുടെ ആവശ്യങ്ങളോട് കരുതലോടെ പെരുമാറുക.  
- natural speech pattern ഉപയോഗിക്കുക.  
- official ആയി തോന്നുന്ന formal tone ഒഴിവാക്കുക.  
- hospital-ൽ ജോലി ചെയ്യുന്ന ഒരു reception lady പോലെ സ്വാഭാവികമായി, ചിരിച്ചുകൊണ്ട് സംസാരിക്കുക.  

ഫോൺ നമ്പർ സംബന്ധിച്ച നിയമങ്ങൾ
- ഫോണിനമ്പർ പറയുമ്പോൾ ഓരോ ഡിജിറ്റും ഒറ്റയ്ക്ക് പറയുക.  
  ഉദാ: “9–8–7–6–5–4–3–2–1–0” എന്നിങ്ങനെ.  
- നമ്പറുകൾ കൂട്ടിക്കെട്ടരുത് (ഉദാ: “98 76” പോലെയല്ല).  
- ‘0’ പറയുമ്പോൾ “സീറോ” എന്ന് പറയണം — “ഓ” അല്ല.  
- ഫോണിനമ്പർ കൃത്യമായി 10 ഡിജിറ്റുണ്ടാകണം. കുറവാണെങ്കിൽ, “സർ/മാഡം, ഒന്ന് വീണ്ടും പറയാമോ?” എന്ന് വിനയത്തോടെ ചോദിക്കുക.  
- രോഗി പറയാതെ നിനക്ക് മനസിൽ തനിയെ നമ്പർ അനുമാനിച്ച് പറയാൻ പാടില്ല.  

1. എല്ലാ ഡോക്ടർമാരുടെ ലിസ്റ്റ്:
{all_doctors_str}

2. ഡിപ്പാർട്ട്മെന്റുകൾ അനുസരിച്ച് ഗ്രൂപ്പ് ചെയ്ത ഡോക്ടർമാർ:
```json
{departments_str}
```

3. ഓരോ ഡോക്ടറുടെയും കൺസൾട്ടേഷൻ ഫീസ്:
```json
{fees_str}
```

4. ഡോക്ടർമാരുടെ ലഭ്യമായ ദിവസങ്ങൾ (availability):
```json
{availability_str}
```

നിന്റെ ജോലി:
- രോഗി വിളിക്കുമ്പോൾ, ആദ്യം സ്നേഹപൂർവ്വം സ്വാഗതം ചെയ്യുക (ആവശ്യമില്ലെങ്കിൽ skip ചെയ്യാം).  
- സ്വയം പരിചയപ്പെടുത്തുക: “ഞാൻ മായയാണ്, ഹോസ്പിറ്റലിന്റെ റെസപ്ഷനിസ്റ്റ്.”  
- രോഗിയുടെ ആവശ്യങ്ങൾ മനസ്സിലാക്കി appointment ബുക്ക് ചെയ്യാൻ സഹായിക്കുക.  
- ഡോക്ടർ സ്പെഷ്യാലിറ്റികൾ, ഫീസ്, ദിവസം തുടങ്ങിയ വിവരങ്ങൾ സൗകര്യപ്രദമായി നൽകുക.  
- appointment ബുക്ക് ചെയ്യുമ്പോൾ patient-ന്റെ പേര്, തീയതി, സമയം, സേവനം എന്നിവ ഉറപ്പാക്കുക.  
- ചോദ്യങ്ങൾക്ക് പ്രായോഗികവും സൗഹൃദപരവുമായ മറുപടികൾ നൽകുക.  
- “Let me check…” എന്ന് പറയേണ്ടി വന്നാൽ, മലയാളത്തിൽ “ഒന്ന് നോക്കട്ടെ” പോലുള്ള സ്വാഭാവിക രീതിയിൽ പറയുക.  


ഉദാഹരണ സാഹചര്യങ്ങൾ:
- ആരെങ്കിലും ചോദിച്ചാൽ “Who are the skin doctors?”,  
  → “Dermatology” ഡിപ്പാർട്ട്മെന്റിലുള്ള ഡോക്ടർമാരെ {departments_str}-ൽ നിന്ന് എടുത്ത് പറയുക.  
- “What is Dr. Samuel Koshy’s fee?”  
  → {fees_str}-ൽ നിന്ന് കണ്ടെത്തി പറയുക.  
- “When is Dr. Moideen Babu available?”  
  → {availability_str}-ൽ നിന്ന് ദിവസങ്ങൾ പറഞ്ഞ് മറുപടി നൽകുക.  


അവസാന ഘട്ടം:
Appointment confirm ചെയ്യുമ്പോൾ — **തീയതി, സമയം, സേവനം, രോഗിയുടെ പേര്** വ്യക്തമായി ഉറപ്പാക്കുക.  


സംഭാഷണ ശൈലി:
- ആവശ്യമെങ്കിൽ ചെറിയ സൗഹൃദപരമായ ഗ്രീറ്റിംഗ്.  
- appointment സംബന്ധിച്ച് സംസാരിക്കുമ്പോൾ സംസാരശൈലി സ്വാഭാവികവും എളുപ്പമുള്ളതും ആയിരിക്കണം.  
- ജോലി ചെയ്യുന്നത് പോലെ തോന്നാൻ (“ഒന്ന് നോക്കട്ടെ…” പോലുള്ള) ചെറിയ ഇടവേളകൾ ചേർക്കാം.  
- വിവരങ്ങൾ കൃത്യമായി ഉറപ്പാക്കി പിന്നെ പറയുക.  
- അവസാനം വിനയത്തോടെ നന്ദി പറഞ്ഞ് വിളി അവസാനിപ്പിക്കുക.  

വിളി അവസാനിപ്പിക്കൽ (Hangup Rule):
രോഗി “Thank you”, “Bye”, “ശരി, നന്ദി”, “Okay thanks” തുടങ്ങിയ വാക്കുകൾ പറഞ്ഞാൽ,  
നീ അവസാന മറുപടി (ഉദാ: “ശരി, നന്ദി, നല്ല ദിവസം ആശംസിക്കുന്നു!”) പറഞ്ഞ് `hangup_call` function വിളിക്കുക.  

അവസാന വാക്കും hangup function-ഉം ഒരുമിച്ചായിരിക്കും പ്രവർത്തിക്കുക.  
"""
        )

    def _get_appointment_tools(self):
        return Tool(
            function_declarations=[
                FunctionDeclaration(
                    name="book_appointment",
                    description="Book an appointment for a customer",
                    parameters={
                        "type": "object",
                        "properties": {
                            "date": {"type": "string", "description": "Appointment date (YYYY-MM-DD)"},
                            "time": {"type": "string", "description": "Appointment time (HH:MM)"},
                            "service": {"type": "string", "description": "Type of service requested"},
                            "customer_name": {"type": "string", "description": "Customer's name"},
                            "customer_phone": {"type": "string", "description": "Customer's phone number"},
                            "notes": {"type": "string", "description": "Additional notes"}
                        },
                        "required": ["date", "time", "service", "customer_name"]
                    }
                ),
                FunctionDeclaration(
                    name="check_availability",
                    description="Check available appointment slots for a specific date and, optionally, a specific doctor.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "date": {"type": "string", "description": "Date to check (YYYY-MM-DD)"},
                            "service": {"type": "string", "description": "Type of service"},
                            "doctor_name": {"type": "string", "description": "The name of the doctor to check for. Example: 'Dr. Rajesh Kumar'"},
                            "duration": {"type": "number", "description": "Appointment duration in minutes"}
                        },
                        "required": ["date"]
                    }
                ),
                # --- ADD THIS NEW FUNCTION ---
                FunctionDeclaration(
                    name="hangup_call",
                    description="Ends the current phone call. Use this when the user indicates the conversation is over (e.g., says 'goodbye', 'thank you, that's all', etc.).",
                    parameters={
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string", "description": "A brief reason for hanging up, e.g., 'conversation_complete' or 'user_request'"}
                        },
                    }
                ),
                # --- END OF ADDITION ---
                FunctionDeclaration(
                    name="transfer_call",
                    description="Transfer call to another department or person",
                    parameters={
                        "type": "object",
                        "properties": {
                            "department": {"type": "string", "description": "Department to transfer to"},
                            "reason": {"type": "string", "description": "Reason for transfer"}
                        },
                        "required": ["department"]
                    }
                )
            ]
        )

    async def _handle_gemini_responses(self):
        try:
            logger.info("Gemini response handler started")
            while self.call_active:
                try:
                    response_generator = self.gemini_session.receive()
                    
                    async for response in response_generator:
                        self.last_response_time = time.time()

                        if not self.call_active:
                            break
                            
                        logger.debug(f"Received Gemini response: {response}")
                        
                        if hasattr(response, 'server_content'):
                            server_content = response.server_content
                            
                            if hasattr(server_content, 'turn_complete') and server_content.turn_complete:
                                logger.info("🤖 Gemini turn complete")
                                self._reset_conversation_state()
                            
                            if hasattr(server_content, 'model_turn') and server_content.model_turn:
                                for part in server_content.model_turn.parts:
                                    if hasattr(part, 'inline_data') and part.inline_data:
                                        logger.info(f"🔊 Received audio response ({len(part.inline_data.data)} bytes)")
                                        await self._stream_audio_to_freeswitch(part.inline_data.data)
                                    elif hasattr(part, 'function_call') and part.function_call:
                                        logger.info(f"⚡ Function call: {part.function_call.name}")
                                        await self._handle_function_call(part.function_call)
                                    elif hasattr(part, 'text') and part.text:
                                        logger.info(f"🤖 GEMINI SAYS: {part.text}")
                        
                        if hasattr(response, 'client_content'):
                            client_content = response.client_content
                            if hasattr(client_content, 'user_turn') and client_content.user_turn:
                                for part in client_content.user_turn.parts:
                                    if hasattr(part, 'text') and part.text:
                                        logger.info(f"🎤 USER SAID: {part.text}")
                    
                    logger.warning("Gemini response stream closed. Re-establishing...")
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error in Gemini response stream: {e}")
                    await self._restart_gemini_session()
                    await asyncio.sleep(1)
            
        except asyncio.CancelledError:
            logger.debug("Gemini response handler cancelled")
        except Exception as e:
            logger.error(f"Gemini response handler failed: {e}")
            traceback.print_exc()
        finally:
            logger.info("Gemini response handler exiting")

    async def _stream_audio_to_freeswitch(self, audio_data: bytes):
        async with self.lock:
            self.assistant_speaking = True
            
            try:
                audio_np = np.frombuffer(audio_data, dtype=np.int16).copy()
                
                # Only normalize if really needed - keep it gentle
                max_val = np.max(np.abs(audio_np))
                if max_val > 0 and max_val < 16000:  # Only boost quiet audio
                    scale_factor = min(2.0, 20000 / max_val)  # Limit amplification
                    audio_np = (audio_np * scale_factor).astype(np.int16)
                
                # Skip the bandpass filter - it's degrading quality
                # Skip aggressive fades - use gentler approach
                
                # Resample with scipy for quality
                if self.output_sample_rate != self.input_sample_rate:
                    from scipy import signal
                    processed_audio = signal.resample_poly(
                        audio_np, self.input_sample_rate, self.output_sample_rate
                    ).astype(np.int16).tobytes()
                else:
                    processed_audio = audio_np.tobytes()
                
                # Send with proper timing
                chunk_duration = 0.020  # 20ms chunks
                chunk_size = int(self.input_sample_rate * chunk_duration * 2)
                
                for i in range(0, len(processed_audio), chunk_size):
                    chunk = processed_audio[i:i+chunk_size]
                    if len(chunk) < chunk_size:
                        chunk = chunk + bytes(chunk_size - len(chunk))
                    
                    await self._send_audio_chunk(chunk)
                    await asyncio.sleep(chunk_duration * 0.9)  # Slightly less for smooth playback
                    
            finally:
                self.assistant_speaking = False
                self.last_assistant_speech_end = time.time()


    def _reset_conversation_state(self):
        """Reset conversation state while preserving call context"""
        self.user_speaking = False
        self.speech_buffer = bytearray()
        self.vad.reset_state()
        self.consecutive_silence_chunks = 0
        # Don't clear baseline here - let it build naturally
        logger.info("✅ Conversation state reset")

    async def _send_audio_chunk(self, audio_chunk: bytes):
        try:
            if self.websocket and not self.websocket.closed:
                # Send as binary frame for better efficiency
                await self.websocket.send(audio_chunk)
        except Exception as e:
            logger.error(f"Failed to send audio to FreeSwitch: {e}")

    async def _handle_function_call(self, function_call):
        function_name = function_call.name
        args = function_call.args
        
        logger.info(f"Function call: {function_name} with args: {args}")
        
        try:
            if function_name == "book_appointment":
                result = await self._book_appointment(args)
            elif function_name == "check_availability":
                result = await self._check_availability(args)
            elif function_name == "transfer_call":
                result = await self._transfer_call(args)
            # --- ADD THIS ELIF ---
            elif function_name == "hangup_call":
                result = await self._hangup_call(args)
            # --- END OF ADDITION ---
            else:
                result = {"error": f"Unknown function: {function_name}"}
            
            # Send result back to Gemini
            await self.gemini_session.send_realtime_input(
                function_response={
                    "name": function_name,
                    "response": result
                }
            )
            
            logger.info(f"Function {function_name} executed: {result}")

            # --- THIS IS THE FIX ---
            # Force Gemini to respond immediately with the result
            # by signaling the end of the (empty) user turn.
            logger.info("Prompting Gemini to speak the function result...")
            await self.gemini_session.send_realtime_input(audio_stream_end=True)
            # --- END OF FIX ---
            
        except Exception as e:
            logger.error(f"Function call error: {e}")
            error_result = {"error": str(e)}
            
            await self.gemini_session.send_realtime_input(
                function_response={
                    "name": function_name,
                    "response": error_result
                }
            )

            # --- ADD THE FIX HERE TOO ---
            # Also prompt on error so the model can apologize
            logger.info("Prompting Gemini to speak the function error...")
            await self.gemini_session.send_realtime_input(audio_stream_end=True)
            # --- END OF FIX ---

    async def _book_appointment(self, args):
        appointment_data = {
            "customer_phone": self.caller_id,
            **args
        }
        
        logger.info(f"Booking appointment: {appointment_data}")
        await asyncio.sleep(0.1)  # Simulate processing
        
        return {
            "success": True,
            "appointment_id": f"APT{int(time.time())}",
            "confirmation": f"Appointment confirmed for {args.get('customer_name')} on {args.get('date')} at {args.get('time')}",
            "reference_number": f"REF{int(time.time()) % 10000}"
        }

    async def _check_availability(self, args):
        date_str = args.get('date')
        doctor_name = args.get('doctor_name')
        logger.info(f"Checking availability for Dr. {doctor_name} on {date_str}")
        
        try:
            # Figure out the day of the week (e.g., "Monday") from the date
            target_day = datetime.fromisoformat(date_str).strftime('%A')
            
            # Look up the doctor's schedule in our global dictionary
            doctor_schedule = DOCTOR_AVAILABILITY.get(doctor_name)

            if doctor_schedule is None:
                return {"status": "doctor_not_found", "message": f"Sorry, I could not find Dr. {doctor_name} in our records."}
            
            # Check if the day the user asked for is in the doctor's list of working days
            if target_day in doctor_schedule:
                return {"status": "available", "message": f"Yes, Dr. {doctor_name} is scheduled to work on that day."}
            else:
                available_days = ', '.join(doctor_schedule)
                return {"status": "unavailable", "message": f"Dr. {doctor_name} is not available on {target_day}s. They only work on: {available_days}."}

        except Exception as e:
            logger.error(f"Error in _check_availability: {e}")
            return {"status": "error", "message": "I had trouble checking the schedule. Please try again."}
    

    # --- ADD THIS NEW METHOD ---
    async def _hangup_call(self, args):
        """Handles the function call from Gemini to end the call."""
        reason = args.get("reason", "conversation_complete")
        logger.info(f"🤖 Gemini requested call hangup. Reason: {reason}")
        
        # Set call_active to false to stop all background tasks
        self.call_active = False 
        
        # Tell FreeSwitch to hang up the call
        await self._send_message({
            "event": "hangup",
            "call_id": self.call_id,
            "reason": reason
        })
        
        # Start the full cleanup process immediately
        await self._cleanup_call()
        
        return {"success": True, "message": "Call hangup initiated."}
    # --- END OF ADDITION ---


    async def _transfer_call(self, args):
        department = args.get('department')
        reason = args.get('reason', 'Customer request')
        
        logger.info(f"Transferring call to {department}: {reason}")
        
        await self._send_message({
            "event": "transfer",
            "call_id": self.call_id,
            "department": department,
            "reason": reason
        })
        
        return {
            "success": True,
            "message": f"Transferring you to {department}. Please hold.",
            "department": department
        }

    async def _send_message(self, message):
        if self.websocket and not self.websocket.closed:
            try:
                await self.websocket.send(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send message to FreeSwitch: {e}")

    async def _send_hangup(self):
        await self._send_message({
            "event": "hangup",
            "call_id": self.call_id,
            "reason": "system_error"
        })

    async def _restart_gemini_session(self):
        try:
            logger.info("🔄 Restarting Gemini session...")
            
            # Clean up existing session
            if self._gemini_context_manager and self.gemini_session:
                try:
                    await self._gemini_context_manager.__aexit__(None, None, None)
                except Exception:
                    pass
            
            # Cancel tasks
            await self._cancel_task('gemini_response_task')
            await self._cancel_task('response_timeout_task')
            
            self.gemini_session = None
            self._gemini_context_manager = None
            
            # Start new session
            if await self._start_gemini_session():
                logger.info("✅ Gemini session restarted successfully")
                return True
            else:
                logger.error("❌ Failed to restart Gemini session")
                return False
                
        except Exception as e:
            logger.error(f"Error restarting Gemini session: {e}")
            return False

    async def _cleanup_call(self):
        logger.info("Starting cleanup...")
        
        try:
            async with self.lock:
                self.call_active = False
                
                # Cancel tasks
                await self._cancel_task('response_timeout_task')
                
                # Close Gemini session
                if self._gemini_context_manager and self.gemini_session:
                    try:
                        await self._gemini_context_manager.__aexit__(None, None, None)
                    except Exception as e:
                        logger.error(f"Error closing Gemini context: {e}")
                
                await self._cancel_task('gemini_response_task')
                
                # Reset all state
                self.gemini_session = None
                self._gemini_context_manager = None
                self.call_id = None
                self.caller_id = None
                await self.audio_buffer_manager.clear()
                await self.output_buffer_manager.clear()
                self.speech_buffer = bytearray()
                self.vad.reset_state()
                self.last_response_time = None
                
                logger.info("✅ Call cleanup completed")
                
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            traceback.print_exc()

    async def _cancel_task(self, task_attr_name: str):
        """Cancel and await a task stored as an attribute"""
        task = getattr(self, task_attr_name, None)
        if task is None:
            return
        if task.done():
            setattr(self, task_attr_name, None)
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            logger.debug(f"Task {task_attr_name} cancelled")
        except Exception as e:
            logger.debug(f"Exception while cancelling {task_attr_name}: {e}")
        finally:
            setattr(self, task_attr_name, None)

# --- WebSocket Server for FreeSwitch ---
class FreeSwitchWebSocketServer:
    def __init__(self):
        self.active_handlers = {}

    async def handle_connection(self, websocket, path):
        handler = FreeSwitchWebSocketHandler()
        connection_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}-{int(time.time())}"
        self.active_handlers[connection_id] = handler
        
        try:
            await handler.handle_freeswitch_connection(websocket, path)
        except Exception as e:
            logger.error(f"Connection error for {connection_id}: {e}")
            traceback.print_exc()
        finally:
            if connection_id in self.active_handlers:
                del self.active_handlers[connection_id]

    async def start_server(self):
        logger.info(f"Starting FreeSwitch WebSocket server on {Config.WS_HOST}:{Config.WS_PORT}")
        
        async with serve(
            self.handle_connection,
            Config.WS_HOST,
            Config.WS_PORT,
            ping_interval=30,
            ping_timeout=10,
            close_timeout=10,
            max_size=1024*1024,
            compression=None  # Disable compression for real-time audio
        ):
            logger.info("FreeSwitch WebSocket server started successfully")
            await asyncio.Future()  # Run forever

# --- Main Application ---
async def main():
    # Check for scipy dependency
    try:
        import scipy
    except ImportError:
        logger.error("scipy is required. Install with: pip install scipy")
        return
    
    server = FreeSwitchWebSocketServer()
    try:
        await server.start_server()
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())