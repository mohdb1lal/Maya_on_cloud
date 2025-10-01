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

# --- Configuration ---
@dataclass
class Config:
    WS_HOST = "0.0.0.0"
    WS_PORT = 8081
    PROJECT_ID = "maya-zappq"
    LOCATION = "us-central1"
    MODEL = "gemini-2.0-flash-live-preview-04-09"
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
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("freeswitch_websocket_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FreeSwitchWebSocket")

# --- ENHANCED DOCTOR AVAILABILITY SYSTEM ---
# Dedicated variable containing doctor names and their real-time availability
DOCTOR_AVAILABILITY = {
    "Dr. Rajesh Kumar": {
        "available": True,
        "specialty": "General Medicine",
        "next_available_slot": "09:00"
    },
    "Dr. Priya Sharma": {
        "available": False,
        "specialty": "Cardiology", 
        "next_available_slot": "14:00"
    },
    "Dr. Arjun Nair": {
        "available": True,
        "specialty": "Dermatology",
        "next_available_slot": "10:30"
    },
    "Dr. Meera Pillai": {
        "available": True,
        "specialty": "Pediatrics",
        "next_available_slot": "11:00"
    },
    "Dr. Anand Krishnan": {
        "available": False,
        "specialty": "Orthopedics",
        "next_available_slot": "15:30"
    },
    "Dr. Sushma Menon": {
        "available": True,
        "specialty": "General Medicine",
        "next_available_slot": "09:30"
    },
    "Dr. Deepak Varma": {
        "available": True,
        "specialty": "Cardiology",
        "next_available_slot": "13:00"
    },
    "Dr. Kavitha Reddy": {
        "available": False,
        "specialty": "Dermatology",
        "next_available_slot": "16:00"
    },
    "Dr. Sunil Thomas": {
        "available": True,
        "specialty": "Orthopedics",
        "next_available_slot": "12:00"
    },
    "Dr. Lakshmi Devi": {
        "available": True,
        "specialty": "Pediatrics",
        "next_available_slot": "10:00"
    }
}

# For backward compatibility - maintaining the original AVAILABLE_DOCTORS list
AVAILABLE_DOCTORS = list(DOCTOR_AVAILABILITY.keys())

# --- Enhanced Functions for Doctor Availability Management ---
def check_doctor_availability(doctor_name: str) -> dict:
    """
    Check if a specific doctor is available for appointments
    Returns detailed availability information
    """
    # Normalize doctor name for case-insensitive matching
    doctor_name_normalized = doctor_name.strip()

    # Direct match
    if doctor_name_normalized in DOCTOR_AVAILABILITY:
        return DOCTOR_AVAILABILITY[doctor_name_normalized]

    # Fuzzy matching for partial names
    for full_name, details in DOCTOR_AVAILABILITY.items():
        if doctor_name_normalized.lower() in full_name.lower():
            return {**details, "matched_name": full_name}

    return {"available": False, "error": "Doctor not found"}

def get_available_doctors_by_specialty(specialty: str = None) -> list:
    """
    Get list of available doctors, optionally filtered by specialty
    """
    available = []
    for doctor, details in DOCTOR_AVAILABILITY.items():
        if details["available"]:
            if specialty is None or specialty.lower() in details["specialty"].lower():
                available.append({
                    "name": doctor,
                    "specialty": details["specialty"],
                    "next_slot": details["next_available_slot"]
                })
    return available

def update_doctor_availability(doctor_name: str, available: bool, next_slot: str = None):
    """
    Update doctor availability status (for dynamic updates)
    """
    if doctor_name in DOCTOR_AVAILABILITY:
        DOCTOR_AVAILABILITY[doctor_name]["available"] = available
        if next_slot:
            DOCTOR_AVAILABILITY[doctor_name]["next_available_slot"] = next_slot
        logger.info(f"Updated {doctor_name} availability: {available}")

# --- Audio Processing Utilities ---
class AudioProcessor:
    """Enhanced audio processing with noise reduction and quality improvements"""

    @staticmethod
    def apply_fade(audio_data: np.ndarray, sample_rate: int, fade_in: bool = True) -> np.ndarray:
        """Apply fade in/out to prevent clicking sounds"""
        if not audio_data.flags.writeable:
            audio_data = audio_data.copy()

        fade_samples = int(Config.FADE_DURATION * sample_rate)
        if fade_samples >= len(audio_data):
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
        # Make a writable copy if the array is read-only
        if not audio_data.flags.writeable:
            audio_data = audio_data.copy()

        rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
        if rms < threshold:
            # Make a writable copy if needed
            return audio_data * (rms / threshold)
        return audio_data

    @staticmethod
    def apply_bandpass_filter(audio_data: np.ndarray, sample_rate: int, lowcut: float = 80, highcut: float = 8000) -> np.ndarray:
        """Apply bandpass filter to remove unwanted frequencies"""
        nyquist = sample_rate * 0.5
        low = lowcut / nyquist
        high = highcut / nyquist

        # Gradually reduce to silence instead of hard cut
        if low >= 1.0:
            low = 0.99
        if high >= 1.0:
            high = 0.99
        if low >= high:
            return audio_data

        try:
            # Ensure valid frequency range
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
    def __init__(self, threshold=Config.VAD_THRESHOLD, min_duration=Config.MIN_SPEECH_DURATION, silence_duration=Config.SILENCE_DURATION):
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
        """Returns (is_currently_speech, speech_just_ended)"""
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

            self.energy_history.append(rms)
            if len(self.energy_history) > 100:
                self.energy_history.pop(0)

            # Adaptive threshold based on recent energy history
            if len(self.energy_history) >= 20:
                # Use adaptive threshold if we have enough history
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
                    logger.info(f"Speech started (RMS: {rms:.1f}, Threshold: {adaptive_threshold:.1f})")
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
                        logger.info(f"Speech ended after {speech_duration:.1f}s")
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
        self.gemini_context_manager = None
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
            await self.send_ready_signal()
            await self.handle_freeswitch_messages()
        except websockets.exceptions.ConnectionClosed:
            logger.info("FreeSwitch connection closed")
        except Exception as e:
            logger.error(f"FreeSwitch handler error: {e}")
            traceback.print_exc()
        finally:
            logger.info("Cleaning up call...")
            await self.cleanup_call()

    def get_current_datetime(self):
        """Get current date and time in Indian timezone"""
        # Set to Indian Standard Time since you're in Kerala
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)
        return {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "day": now.strftime("%A"),
            "formatted": now.strftime("%A, %B %d, %Y at %I:%M %p")
        }

    async def send_ready_signal(self):
        await self.send_message({
            "event": "ready",
            "data": {
                "protocols": ["audio"],
                "formats": ["L16"],  # Only use L16 Linear PCM
                "sample_rates": [16000, 24000],
                "channels": 1  # Mono audio only
            }
        })

    async def handle_freeswitch_messages(self):
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
                await self.send_hangup()
                return

            # Set to Indian Standard Time since you're in Kerala
            await self.trigger_initial_greeting()
            await self.send_message({
                "event": "started",
                "call_id": self.call_id,
                "status": "ready"
            })

    async def trigger_initial_greeting(self):
        """Prompts Gemini to start the conversation with its configured greeting."""
        try:
            if self.gemini_session:
                logger.info("Triggering Gemini's initial greeting...")
                # Proactively trigger Gemini's welcome message instead of waiting for user input.
                await self.gemini_session.send(realtime_input={"audio_stream_end": True})
                logger.info("Greeting trigger sent successfully.")
        except Exception as e:
            logger.error(f"Failed to trigger initial greeting: {e}")

    async def _handle_media_info(self, data):
        media_info = data.get("data", {})
        logger.info(f"Media info: {media_info}")

        # The AI will follow its system instruction to provide a warm welcome.
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

        # Update sample rates based on media info
        current_time = time.time()

        # Block audio input during and shortly after assistant speech
        if self.assistant_speaking:
            logger.debug("Blocking audio - assistant currently speaking")
            self.consecutive_silence_chunks = 0
            return

        # Echo suppression: block input for a period after assistant stops speaking
        time_since_assistant_spoke = current_time - self.last_assistant_speech_end
        if self.last_assistant_speech_end > 0 and time_since_assistant_spoke < Config.ECHO_BLOCK_DURATION:
            logger.debug(f"Echo suppression active for {Config.ECHO_BLOCK_DURATION - time_since_assistant_spoke:.1f}s more")
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

            processed_audio = audio_np.tobytes()

            # Voice Activity Detection
            is_speech, speech_just_ended = self.vad.is_speech(processed_audio)

            if is_speech:
                self.consecutive_silence_chunks = 0
                if not self.user_speaking:
                    # Start speech detection with reasonable threshold
                    if rms > Config.VAD_THRESHOLD:
                        self.user_speaking = True
                        logger.info(f"USER STARTED SPEAKING (RMS: {rms:.1f})")
                        self.speech_buffer = bytearray()
                        self.audio_energy_baseline = []

                if self.user_speaking:
                    self.speech_buffer.extend(processed_audio)
                    await self.send_to_gemini(processed_audio)
            else:
                self.consecutive_silence_chunks += 1
                if speech_just_ended and self.user_speaking:
                    # Check minimum speech duration
                    speech_duration = len(self.speech_buffer) / (self.input_sample_rate * 2)
                    if speech_duration >= Config.MIN_SPEECH_DURATION:
                        self.user_speaking = False
                        logger.info(f"USER FINISHED SPEAKING (duration: {speech_duration:.1f}s)")
                        await self.send_speech_end_signal()
                    else:
                        logger.debug(f"Speech too short ({speech_duration:.1f}s), continuing")
                        self.vad.active = True

            # Add to buffer
            await self.audio_buffer_manager.add_audio(processed_audio)

        except Exception as e:
            logger.error(f"Audio stream processing error: {e}")

    async def send_to_gemini(self, audio_chunk: bytes):
        try:
            if self.gemini_session:
                await self.gemini_session.send(realtime_input={"media_chunks": [{"data": audio_chunk, "mime_type": "audio/pcm"}]})
        except Exception as e:
            logger.error(f"Error sending audio to Gemini: {e}")

    async def send_speech_end_signal(self):
        """Send end-of-speech signal to Gemini"""
        try:
            if self.gemini_session:
                await self.gemini_session.send(realtime_input={"audio_stream_end": True})
                logger.info("Sent end-of-speech signal to Gemini")
                # Option 1: Use the "audio_stream_end" parameter (recommended)
        except Exception as e:
            logger.error(f"Error sending speech end signal: {e}")

    def get_appointment_tools(self):
        return [
            Tool(
                function_declarations=[
                    FunctionDeclaration(
                        name="book_appointment",
                        description="Book an appointment for a customer with enhanced doctor availability checking",
                        parameters={
                            "type": "object",
                            "properties": {
                                "date": {"type": "string", "description": "Appointment date (YYYY-MM-DD)"},
                                "time": {"type": "string", "description": "Appointment time (HH:MM)"},
                                "doctor_name": {"type": "string", "description": "Name of the requested doctor"},
                                "service": {"type": "string", "description": "Type of service requested"},
                                "customer_name": {"type": "string", "description": "Customer's name"},
                                "customer_phone": {"type": "string", "description": "Customer's phone number"},
                                "notes": {"type": "string", "description": "Additional notes"}
                            },
                            "required": ["date", "time", "doctor_name", "service", "customer_name"]
                        }
                    ),
                    FunctionDeclaration(
                        name="check_doctor_availability",
                        description="Check if a specific doctor is available for appointments",
                        parameters={
                            "type": "object",
                            "properties": {
                                "doctor_name": {"type": "string", "description": "Name of the doctor to check"},
                                "date": {"type": "string", "description": "Date to check availability (YYYY-MM-DD)"},
                                "specialty": {"type": "string", "description": "Medical specialty if searching by specialty"}
                            },
                            "required": ["doctor_name"]
                        }
                    ),
                    FunctionDeclaration(
                        name="get_available_doctors",
                        description="Get list of currently available doctors, optionally filtered by specialty",
                        parameters={
                            "type": "object",
                            "properties": {
                                "specialty": {"type": "string", "description": "Filter by medical specialty (optional)"},
                                "date": {"type": "string", "description": "Date to check availability (YYYY-MM-DD)"}
                            },
                            "required": []
                        }
                    ),
                    FunctionDeclaration(
                        name="check_availability",
                        description="Check available appointment slots for any doctor",
                        parameters={
                            "type": "object",
                            "properties": {
                                "date": {"type": "string", "description": "Date to check (YYYY-MM-DD)"},
                                "service": {"type": "string", "description": "Type of service"},
                                "duration": {"type": "number", "description": "Appointment duration in minutes"}
                            },
                            "required": ["date"]
                        }
                    ),
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
        ]

    async def _start_gemini_session(self) -> bool:
        try:
            # Clean up existing session
            if self.gemini_context_manager and self.gemini_session:
                try:
                    await self.gemini_context_manager.__aexit__(None, None, None)
                except Exception:
                    pass

            # Cancel tasks
            await self.cancel_task("gemini_response_task")
            await self.cancel_task("response_timeout_task")
            self.gemini_session = None
            self.gemini_context_manager = None

            # Start new session
            client = genai.Client(
                api_key=os.environ.get("GOOGLE_API_KEY"),
                http_options={"api_version": "v1alpha"}
            )

            # Cancel tasks
            await self.cancel_task("response_timeout_task")
            self.response_timeout_task = asyncio.create_task(self.response_timeout_handler())

            config = self.get_gemini_config()
            self.gemini_context_manager = client.live.connect(config=config)
            self.gemini_session = await self.gemini_context_manager.__aenter__()

            # Start response handling
            await self.cancel_task("gemini_response_task")
            self.gemini_response_task = asyncio.create_task(self.handle_gemini_responses())

            logger.info("Gemini session started")
            return True
        except Exception as e:
            logger.error(f"Gemini session failed: {e}")
            traceback.print_exc()
            return False

    def get_gemini_config(self):
        current_dt = self.get_current_datetime()
        return LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(voice_name="Aoede")
                )
            ),
            tools=self.get_appointment_tools(),
            system_instruction=f"""
LANGUAGE: COLLOQUIAL MALAYALAM. You are MAYA, a friendly and professional ai assistant for zappque medical hospital in KERALA. So always talk in malayalam but you can use english terms when needed, specializing in appointment scheduling. Your task is to assist callers in booking, rescheduling, or canceling medical appointments. To get the list of doctors available use list. You knows the current date and time is {current_dt['formatted']}. so don't ask for it. If the caller asks for the date or time, provide it based on this information. Don't spell RECEPTIONIST. As it is a position, YOU are the RECEPTIONIST MAYA.

CRITICAL LINGUISTIC RULE: You MUST speak in the natural, conversational style of a person from KERALA (Colloquial Malayalam).

CULTURAL VOCABULARY RULE: When referring to doctor specialties or hospital departments, use ENGLISH terms

YOUR ROLE:
- Greet every patient when they are connected (skip this or make it short if the user is asking to book directly)
- Introduce yourself as Maya, the hospital RECEPTIONIST
- Help patients book appointments with doctors
- **ALWAYS check doctor availability using the check_doctor_availability function BEFORE booking**
- Provide information about doctor specialties available
- Ask for necessary details in a conversational way
- Confirm appointment details clearly
- Also you can give details disease information
- Don't be stubborn, reply to the user on what they need.
- You can provide any sort of information to the user, but your main task is booking appointments

ENHANCED DOCTOR AVAILABILITY CHECKING:
- **MANDATORY**: When a patient requests an appointment with a specific doctor, you MUST first use the check_doctor_availability function
- **DO NOT** say "let me check if appointment is available" and leave the user waiting
- **INSTEAD**: Actually check using the function and immediately respond with:
  - If available: "Dr. [Name] is available! I can book your appointment for [time]"
  - If not available: "Dr. [Name] is currently not available, but their next available slot is [time]. Would you like me to book that or suggest another available doctor?"
- Use get_available_doctors function to suggest alternative doctors if the requested one is unavailable

AVAILABLE SPECIALTIES: Use the Malayalam/Manglish terms in conversation:
- General medicine (ജനറൽ മെഡിസിൻ, സാധാരണ ഡോക്ടർ) - General Doctor
- Cardiology (ഹൃദയരോഗവിദ്യ) - Heart Doctor  
- Dermatology (ത്വക്രോഗവിദ്യ, സ്കിൻ) - Skin Doctor
- Orthopedic (അസ്ഥിരോഗം, ബോൺ) - Bone Doctor
- Pediatric (കുട്ടികളുടെ ഡോക്ടർ) - Children's Doctor

AFTER GETTING THE DETAILS: Confirm the DATE, TIME, DOCTOR NAME, SERVICE, and CUSTOMER NAME with the caller before finalizing the appointment.

CONVERSATION STYLE:
- Start with a warm greeting (You can skip this and go to next step if the user is asking to book instead of saying hello)
- When booking appointments, ask for details naturally and conversationally
- **Show you're working**: Actually use the availability checking functions and respond immediately with results
- Confirm details warmly
- End calls by saying thank you and wishing well.

IMPORTANT BEHAVIORAL GUIDELINES:
- Speak naturally like a real human receptionist - use casual, conversational language
- You MUST respond ONLY in the MALAYALAM to the user
- DO NOT provide translations or repeat your answer in another language.
- Never provide a response in one language followed by the same response in another. Stick to ONLY MALAYALAM
- If the user is directly asking to book appointment, then don't say the greetings and all, just go directly into bookings.
- Show empathy and concern for patients' needs
- Use contractions and natural speech patterns appropriate in MALAYALAM
- Add small personal touches and warmth to your responses
- Never mention you're an AI or assistant - you're Maya, the receptionist
- Sound genuinely helpful and caring, not robotic or overly formal
- Whenever repeating or confirming a phone number, ALWAYS say it digit by digit for example 9876543210. NEVER group numbers into thousands, lakhs, crores, or treat them like money or quantities. Phone numbers are NOT amounts of money - they must be spoken ONLY as individual digits, one by one. Don't repeat the dictation unless the user asks for it.

Phone Number confirmation guidelines:
- While confirming phone number, the digit 0 should be spelled as "Zero" not "Ooo" okay.
- Be very attentive when noting the phone number, don't make any mistake and also without the user dictating you the phone number,
- don't assume any random number, be very cautious about it
"""
        )

    async def handle_function_call(self, function_call):
        function_name = function_call.name
        args = function_call.args
        logger.info(f"Function call: {function_name} with args: {args}")

        try:
            if function_name == "book_appointment":
                result = await self.book_appointment(args)
            elif function_name == "check_doctor_availability":
                result = await self.check_doctor_availability_async(args)
            elif function_name == "get_available_doctors":
                result = await self.get_available_doctors_async(args)
            elif function_name == "check_availability":
                result = await self.check_availability(args)
            elif function_name == "transfer_call":
                result = await self.transfer_call(args)
            else:
                result = {"error": f"Unknown function: {function_name}"}

            # Send result back to Gemini
            await self.gemini_session.send(
                realtime_input={"function_response": {"name": function_name, "response": result}}
            )
            logger.info(f"Function {function_name} executed: {result}")
        except Exception as e:
            logger.error(f"Function call error: {e}")
            error_result = {"error": str(e)}
            await self.gemini_session.send(
                realtime_input={"function_response": {"name": function_name, "response": error_result}}
            )

    async def book_appointment(self, args):
        """Enhanced appointment booking with doctor availability verification"""
        # First check if the requested doctor is available
        doctor_name = args.get("doctor_name", "")
        availability_info = check_doctor_availability(doctor_name)

        if not availability_info.get("available", False):
            if "error" in availability_info:
                return {
                    "success": False,
                    "error": f"Doctor {doctor_name} not found in our system",
                    "message": "Please check the doctor name and try again"
                }
            else:
                return {
                    "success": False,
                    "error": f"Dr. {doctor_name} is currently not available",
                    "next_available": availability_info.get("next_available_slot"),
                    "message": f"Dr. {doctor_name} is not available right now. Next available slot: {availability_info.get('next_available_slot', 'Unknown')}"
                }

        # If doctor is available, proceed with booking
        appointment_data = {
            "customer_phone": self.caller_id,
            **args
        }
        logger.info(f"Booking appointment: {appointment_data}")

        # Simulate processing time
        await asyncio.sleep(0.1)

        return {
            "success": True,
            "appointment_id": f"APT{int(time.time())}",
            "confirmation": f"Appointment confirmed for {args.get('customer_name')} with {doctor_name} on {args.get('date')} at {args.get('time')}",
            "doctor_info": {
                "name": doctor_name,
                "specialty": availability_info.get("specialty", "General"),
                "next_available": availability_info.get("next_available_slot")
            },
            "reference_number": f"REF{int(time.time()) % 10000}"
        }

    async def check_doctor_availability_async(self, args):
        """Async wrapper for doctor availability checking"""
        doctor_name = args.get("doctor_name", "")
        logger.info(f"Checking availability for: {doctor_name}")

        # Simulate checking time (very brief)
        await asyncio.sleep(0.05)

        availability_info = check_doctor_availability(doctor_name)

        if "error" in availability_info:
            return {
                "doctor_found": False,
                "error": availability_info["error"],
                "available_doctors": get_available_doctors_by_specialty()[:3]  # Suggest 3 alternatives
            }

        return {
            "doctor_found": True,
            "doctor_name": availability_info.get("matched_name", doctor_name),
            "available": availability_info["available"],
            "specialty": availability_info["specialty"],
            "next_available_slot": availability_info["next_available_slot"],
            "message": f"Dr. {doctor_name} is {'available' if availability_info['available'] else 'not available right now'}"
        }

    async def get_available_doctors_async(self, args):
        """Get list of available doctors with optional specialty filtering"""
        specialty = args.get("specialty")
        logger.info(f"Getting available doctors for specialty: {specialty}")

        await asyncio.sleep(0.05)  # Brief processing simulation

        available_doctors = get_available_doctors_by_specialty(specialty)

        return {
            "available_doctors": available_doctors,
            "total_available": len(available_doctors),
            "specialty_filter": specialty,
            "message": f"Found {len(available_doctors)} available doctors" + (f" in {specialty}" if specialty else "")
        }

    async def check_availability(self, args):
        """Original availability checking function for backward compatibility"""
        logger.info(f"Checking availability: {args}")
        available_slots = ["09:00", "09:30", "10:00", "10:30", "11:00", "14:00", "14:30", "15:00", "15:30", "16:00"]
        return {
            "date": args.get("date"),
            "available_slots": available_slots,
            "service": args.get("service", "general"),
            "next_available": available_slots[0] if available_slots else None
        }

    async def transfer_call(self, args):
        department = args.get("department")
        reason = args.get("reason", "Customer request")
        logger.info(f"Transferring call to {department}: {reason}")

        await self.send_message({
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

    async def handle_gemini_responses(self):
        try:
            logger.info("Gemini response handler started")
            while self.call_active:
                try:
                    response_generator = self.gemini_session.receive()
                    async for response in response_generator:
                        self.last_response_time = time.time()

                        # Start response handling
                        if not self.call_active:
                            break

                        logger.debug(f"Received Gemini response: {response}")

                        if hasattr(response, 'server_content'):
                            server_content = response.server_content

                            if hasattr(server_content, 'turn_complete') and server_content.turn_complete:
                                logger.info("Gemini turn complete")
                                self.reset_conversation_state()

                            if hasattr(server_content, 'model_turn') and server_content.model_turn:
                                for part in server_content.model_turn.parts:
                                    if hasattr(part, 'inline_data') and part.inline_data:
                                        logger.info(f"Received audio response: {len(part.inline_data.data)} bytes")
                                        await self.stream_audio_to_freeswitch(part.inline_data.data)
                                    elif hasattr(part, 'function_call') and part.function_call:
                                        logger.info(f"Function call: {part.function_call.name}")
                                        await self.handle_function_call(part.function_call)
                                    elif hasattr(part, 'text') and part.text:
                                        logger.info(f"GEMINI SAYS: {part.text}")

                        if hasattr(response, 'client_content'):
                            client_content = response.client_content
                            if hasattr(client_content, 'user_turn') and client_content.user_turn:
                                for part in client_content.user_turn.parts:
                                    if hasattr(part, 'text') and part.text:
                                        logger.info(f"USER SAID: {part.text}")

                except Exception as e:
                    if self.call_active:
                        logger.error(f"Error in Gemini response handling: {e}")
                        break

        except Exception as e:
            logger.error(f"Gemini response handler error: {e}")
        finally:
            logger.info("Gemini response handler stopped")

    def reset_conversation_state(self):
        """Reset conversation state after turn complete"""
        self.assistant_speaking = False
        self.last_assistant_speech_end = time.time()
        logger.debug("Conversation state reset")

    async def stream_audio_to_freeswitch(self, audio_data: bytes):
        """Stream audio response to FreeSwitch"""
        try:
            self.assistant_speaking = True

            # Process audio data for output
            if self.output_sample_rate != Config.OUTPUT_SAMPLE_RATE:
                audio_data = self.audio_processor.resample_audio(
                    audio_data, Config.OUTPUT_SAMPLE_RATE, self.output_sample_rate
                )

            # Send audio to FreeSwitch
            chunk_size = Config.AUDIO_BUFFER_SIZE
            for i in range(0, len(audio_data), chunk_size):
                if not self.call_active:
                    break

                audio_chunk = audio_data[i:i + chunk_size]
                # Send as binary frame for better efficiency
                await self.websocket.send(audio_chunk)

        except Exception as e:
            logger.error(f"Failed to send audio to FreeSwitch: {e}")

    async def response_timeout_handler(self):
        """Idle-based response timeout handler"""
        try:
            while True:
                await asyncio.sleep(0.5)

                if self.assistant_speaking:
                    continue

                if self.last_response_time is None:
                    continue

                if time.time() - self.last_response_time > self.response_timeout:
                    logger.warning(f"Response timeout after {self.response_timeout}s")

                    async with self.lock:
                        self.assistant_speaking = False
                        self.user_speaking = False
                        await self.audio_buffer_manager.clear()
                        self.speech_buffer = bytearray()
                        self.vad.reset_state()

                    logger.info("System reset due to timeout")
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

        await self.send_message({
            "event": "ended",
            "call_id": self.call_id,
            "reason": reason
        })

    async def restart_gemini_session(self):
        try:
            logger.info("Restarting Gemini session...")

            # Cancel tasks
            await self.cancel_task("response_timeout_task")
            await self.cancel_task("gemini_response_task")
            await self.cancel_task("response_timeout_task")
            self.gemini_session = None
            self.gemini_context_manager = None

            # Start new session
            if await self._start_gemini_session():
                logger.info("Gemini session restarted successfully")
                return True
            else:
                logger.error("Failed to restart Gemini session")
                return False
        except Exception as e:
            logger.error(f"Error restarting Gemini session: {e}")
            return False

    async def cleanup_call(self):
        logger.info("Starting cleanup...")
        try:
            async with self.lock:
                self.call_active = False
                # Reset all state
                self.gemini_session = None
                self.gemini_context_manager = None
                self.call_id = None
                self.caller_id = None
                await self.audio_buffer_manager.clear()
                await self.output_buffer_manager.clear()
                self.speech_buffer = bytearray()
                self.vad.reset_state()
                self.last_response_time = None

            logger.info("Call cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            traceback.print_exc()

    async def cancel_task(self, task_attr_name: str):
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

    async def send_message(self, message):
        if self.websocket and not self.websocket.closed:
            try:
                await self.websocket.send(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send message to FreeSwitch: {e}")

    async def send_hangup(self):
        await self.send_message({
            "event": "hangup",
            "call_id": self.call_id,
            "reason": "system_error"
        })

class FreeSwitchWebSocketServer:
    def __init__(self, host=Config.WS_HOST, port=Config.WS_PORT):
        self.host = host
        self.port = port
        self.handlers = {}

    async def handle_connection(self, websocket, path):
        handler = FreeSwitchWebSocketHandler()
        self.handlers[websocket] = handler

        try:
            await handler.handle_freeswitch_connection(websocket, path)
        finally:
            del self.handlers[websocket]

    async def start_server(self):
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")

        # Process chunks
        while True:
            chunk = await self.audio_buffer_manager.get_chunk(Config.CHUNK_SIZE)
            if chunk is None:
                break

            is_speech, _ = self.vad.is_speech(chunk)
            if is_speech:
                self.consecutive_silence_chunks = 0
                if not self.user_speaking:
                    # Check minimum speech duration
                    # Start speech detection with reasonable threshold
                    if rms > Config.VAD_THRESHOLD:
                        self.user_speaking = True
                        logger.info(f"USER STARTED SPEAKING (RMS: {rms:.1f})")
                        self.speech_buffer = bytearray()
                        self.audio_energy_baseline = []

                if self.user_speaking:
                    self.speech_buffer.extend(chunk)
                    await self.send_to_gemini(chunk)
            else:
                self.consecutive_silence_chunks += 1
                if self.vad.just_ended and self.user_speaking:
                    speech_duration = len(self.speech_buffer) / (self.input_sample_rate * 2)
                    if speech_duration >= Config.MIN_SPEECH_DURATION:
                        self.user_speaking = False
                        logger.info(f"USER FINISHED SPEAKING (duration: {speech_duration:.1f}s)")
                        await self.send_speech_end_signal()
                    else:
                        logger.debug(f"Speech too short ({speech_duration:.1f}s), continuing")
                        self.vad.active = True

        async with serve(self.handle_connection, self.host, self.port):
            logger.info("WebSocket server started successfully")
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
