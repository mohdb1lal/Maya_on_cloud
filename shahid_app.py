# Complete Multi-Call FreeSWITCH WebSocket Server with Multiplexing
import asyncio
import logging
import traceback
import numpy as np
import json
import base64
import time
import uuid
import collections
from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime # <-- ADDED
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
    VAD_THRESHOLD = 300
    MIN_SPEECH_DURATION = 0.7      # Tuned for better noise rejection
    SILENCE_DURATION = 1.5         # Tuned to allow for natural pauses
    RESPONSE_TIMEOUT = 10.0
    AUDIO_BUFFER_SIZE = 8192
    FADE_DURATION = 0.01
    NOISE_GATE_THRESHOLD = 80
    ECHO_BLOCK_DURATION = 1.0
    MAX_CONCURRENT_CALLS = 10

# --- Call Manager (No changes needed) ---
class CallManager:
    """Manages multiple concurrent calls with proper resource isolation"""
    
    def __init__(self):
        self.active_calls: Dict[str, 'FreeSwitchWebSocketHandler'] = {}
        self.call_lock = asyncio.Lock()
        self.stats = {
            'total_calls': 0,
            'current_calls': 0,
            'max_concurrent': 0,
            'rejected_calls': 0,
            'start_time': time.time()
        }
    
    async def can_accept_call(self) -> bool:
        """Check if we can accept another call"""
        async with self.call_lock:
            return len(self.active_calls) < Config.MAX_CONCURRENT_CALLS
    
    async def add_call(self, call_id: str, handler: 'FreeSwitchWebSocketHandler') -> bool:
        """Add a new call to the manager"""
        async with self.call_lock:
            if len(self.active_calls) >= Config.MAX_CONCURRENT_CALLS:
                self.stats['rejected_calls'] += 1
                logger.warning(f"Call {call_id} rejected - max concurrent calls ({Config.MAX_CONCURRENT_CALLS}) reached")
                return False
            
            self.active_calls[call_id] = handler
            self.stats['total_calls'] += 1
            self.stats['current_calls'] = len(self.active_calls)
            
            if self.stats['current_calls'] > self.stats['max_concurrent']:
                self.stats['max_concurrent'] = self.stats['current_calls']
            
            logger.info(f"Call {call_id} added - Active calls: {self.stats['current_calls']}/{Config.MAX_CONCURRENT_CALLS}")
            return True
    
    async def remove_call(self, call_id: str):
        """Remove a call from the manager"""
        async with self.call_lock:
            if call_id in self.active_calls:
                del self.active_calls[call_id]
                self.stats['current_calls'] = len(self.active_calls)
                logger.info(f"Call {call_id} removed - Active calls: {self.stats['current_calls']}/{Config.MAX_CONCURRENT_CALLS}")
    
    async def get_call(self, call_id: str) -> Optional['FreeSwitchWebSocketHandler']:
        """Get a specific call handler"""
        async with self.call_lock:
            return self.active_calls.get(call_id)
    
    async def get_stats(self) -> Dict:
        """Get call statistics"""
        async with self.call_lock:
            uptime = time.time() - self.stats['start_time']
            return {
                **self.stats,
                'uptime_hours': uptime / 3600,
                'calls_per_hour': self.stats['total_calls'] / (uptime / 3600) if uptime > 0 else 0
            }
    
    async def cleanup_all_calls(self):
        """Cleanup all active calls"""
        async with self.call_lock:
            for call_id, handler in list(self.active_calls.items()):
                try:
                    await handler._cleanup_call()
                except Exception as e:
                    logger.error(f"Error cleaning up call {call_id}: {e}")
            self.active_calls.clear()
            self.stats['current_calls'] = 0


# --- Logging Setup (No changes needed) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multiplexed_freeswitch_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MultiplexedFreeSWITCH')

# --- Audio Processing Utilities (No changes needed) ---
class AudioProcessor:
    """Enhanced audio processing with noise reduction and quality improvements"""
    
    @staticmethod
    def apply_fade(audio_data: np.ndarray, sample_rate: int, fade_in: bool = True) -> np.ndarray:
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
        if not audio_data.flags.writeable:
            audio_data = audio_data.copy()
        rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
        if rms < threshold:
            return audio_data * (rms / threshold)
        return audio_data
    
    @staticmethod
    def apply_bandpass_filter(audio_data: np.ndarray, sample_rate: int, 
                             lowcut: float = 80, highcut: float = 8000) -> np.ndarray:
        nyquist = sample_rate * 0.5
        low = lowcut / nyquist
        high = highcut / nyquist
        if low >= 1.0: low = 0.99
        if high >= 1.0: high = 0.99
        if low >= high: return audio_data
        try:
            sos = signal.butter(4, [low, high], btype='band', output='sos')
            filtered = signal.sosfilt(sos, audio_data.astype(np.float64))
            return filtered.astype(audio_data.dtype)
        except Exception as e:
            logger.warning(f"Bandpass filter failed: {e}")
            return audio_data
    
    @staticmethod
    def resample_audio(audio_data: bytes, input_rate: int, output_rate: int) -> bytes:
        if input_rate == output_rate: return audio_data
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            resampled = signal.resample_poly(audio_np, output_rate, input_rate)
            resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
            return resampled.tobytes()
        except Exception as e:
            logger.error(f"Resampling error: {e}")
            return audio_data

    @staticmethod
    def normalize_audio(audio_data: np.ndarray, target_level: float = 0.8) -> np.ndarray:
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            scale_factor = (target_level * 32767) / max_val
            if scale_factor < 1.0:
                return (audio_data * scale_factor).astype(np.int16)
        return audio_data

# --- IMPROVED Voice Activity Detector ---
class VoiceActivityDetector:
    def __init__(self, call_id: str, threshold=Config.VAD_THRESHOLD, 
                 min_duration=Config.MIN_SPEECH_DURATION, 
                 silence_duration=Config.SILENCE_DURATION):
        self.call_id = call_id
        self.threshold = threshold
        self.min_duration = min_duration
        self.silence_duration = silence_duration
        self.audio_processor = AudioProcessor()
        # Buffer about 240ms of audio (12 chunks * 20ms/chunk)
        self.pre_speech_buffer = collections.deque(maxlen=12)
        self.reset_state()
        
    def reset_state(self):
        """Reset all VAD state variables"""
        self.speech_start = None
        self.is_speaking = False
        self.energy_history = []
        self.consecutive_silence_frames = 0
        self.min_silence_frames = int(self.silence_duration * 50) # 50 frames/sec for 20ms chunks
        if hasattr(self, 'pre_speech_buffer'):
            self.pre_speech_buffer.clear()
        logger.debug(f"Call {self.call_id}: VAD state reset")

    def process_chunk(self, audio_data: bytes) -> tuple[bool, bool]:
        """
        Processes a single audio chunk.
        Returns (is_currently_speech, speech_just_ended)
        """
        try:
            if len(audio_data) % 2 != 0: audio_data = audio_data[:-1]
            if len(audio_data) == 0: return False, False

            audio = np.frombuffer(audio_data, dtype=np.int16)
            audio = self.audio_processor.apply_noise_gate(audio)
            rms = np.sqrt(np.mean(np.square(audio.astype(np.float32))))
            
            self.energy_history.append(rms)
            if len(self.energy_history) > 100: self.energy_history.pop(0)
            
            adaptive_threshold = self.threshold
            if len(self.energy_history) > 20:
                noise_floor = np.percentile(self.energy_history, 20)
                adaptive_threshold = max(self.threshold, noise_floor * 2.5)
            
            now = time.time()
            speech_detected = rms > adaptive_threshold
            speech_just_ended = False
            
            if speech_detected:
                self.consecutive_silence_frames = 0
                if not self.is_speaking:
                    # This is the moment speech starts
                    self.speech_start = now
                    self.is_speaking = True
                    logger.info(f"Call {self.call_id}: Speech started (RMS: {rms:.1f} > Threshold: {adaptive_threshold:.1f})")
            else:
                # Buffer non-speech chunks for context
                self.pre_speech_buffer.append(audio_data)
                self.consecutive_silence_frames += 1
                
                if self.is_speaking and self.consecutive_silence_frames >= self.min_silence_frames:
                    speech_duration = now - self.speech_start if self.speech_start else 0
                    if speech_duration >= self.min_duration:
                        self.is_speaking = False
                        speech_just_ended = True
                        logger.info(f"Call {self.call_id}: Speech ended after {speech_duration:.1f}s of silence")
                    else:
                        # Speech was too short, discard it as noise
                        self.is_speaking = False
            
            return self.is_speaking, speech_just_ended
            
        except Exception as e:
            logger.error(f"Call {self.call_id}: VAD error: {e}")
            return False, False

    def get_buffered_audio(self) -> bytes:
        """Returns and clears the pre-speech buffer."""
        buffered_audio = b"".join(self.pre_speech_buffer)
        self.pre_speech_buffer.clear()
        return buffered_audio

# --- Audio Buffer Manager (No changes needed) ---
class AudioBufferManager:
    """Manages audio buffering with proper synchronization and quality control"""
    
    def __init__(self, call_id: str, buffer_size: int = Config.AUDIO_BUFFER_SIZE):
        self.call_id = call_id
        self.buffer = bytearray()
        self.buffer_size = buffer_size
        self.lock = asyncio.Lock()
        
    async def add_audio(self, audio_data: bytes):
        async with self.lock:
            self.buffer.extend(audio_data)
            if len(self.buffer) > self.buffer_size * 2:
                self.buffer = self.buffer[-self.buffer_size:]
                logger.warning(f"Call {self.call_id}: Audio buffer overflow - trimming old data")
    
    async def get_chunk(self, chunk_size: int) -> bytes:
        async with self.lock:
            if len(self.buffer) < chunk_size: return None
            chunk = bytes(self.buffer[:chunk_size])
            self.buffer = self.buffer[chunk_size:]
            return chunk
    
    async def clear(self):
        async with self.lock:
            self.buffer = bytearray()

# --- Enhanced WebSocket Handler ---
class FreeSwitchWebSocketHandler:
    def __init__(self, call_manager: CallManager):
        self.call_manager = call_manager
        self.websocket = None
        self.gemini_session = None
        self.call_active = False
        self.assistant_speaking = False
        self.user_speaking = False
        self.lock = asyncio.Lock()
        self.call_id = None
        self.caller_id = None
        self.session_id = str(uuid.uuid4())[:8]
        self._gemini_context_manager = None
        self.speech_buffer = bytearray()
        self.response_timeout = Config.RESPONSE_TIMEOUT
        self.gemini_response_task = None
        self.response_timeout_task = None
        self.last_response_time = None
        self.audio_processor = AudioProcessor()
        self.output_sample_rate = Config.OUTPUT_SAMPLE_RATE
        self.input_sample_rate = Config.INPUT_SAMPLE_RATE
        self.last_assistant_speech_end = 0
        self.vad = None
        self.audio_buffer_manager = None

    async def handle_freeswitch_connection(self, websocket, path):
        logger.info(f"Session {self.session_id}: FreeSWITCH connection from {websocket.remote_address}")
        self.websocket = websocket
        try:
            await self._send_ready_signal()
            await self._handle_freeswitch_messages()
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Session {self.session_id}: FreeSWITCH connection closed")
        except Exception as e:
            logger.error(f"Session {self.session_id}: FreeSWITCH handler error: {e}", exc_info=True)
        finally:
            logger.info(f"Session {self.session_id}: Cleaning up call...")
            await self._cleanup_call()

    async def _send_ready_signal(self):
        await self._send_message({"event": "ready", "session_id": self.session_id})

    async def _handle_freeswitch_messages(self):
        async for message in self.websocket:
            try:
                if isinstance(message, bytes):
                    await self._handle_audio_stream(message)
                else:
                    data = json.loads(message)
                    await self._handle_control_message(data)
            except Exception as e:
                logger.error(f"Session {self.session_id}: Message processing error: {e}")

    async def _handle_control_message(self, data):
        event_type = data.get("event")
        if event_type == "start": await self._handle_call_start(data)
        elif event_type == "media": await self._handle_media_info(data)
        elif event_type == "hangup": await self._handle_call_end(data)

    async def _handle_call_start(self, data):
        async with self.lock:
            self.call_id = data.get("call_id", f"call_{self.session_id}")
            self.caller_id = data.get("caller_id")
            
            if not await self.call_manager.add_call(self.call_id, self):
                await self._send_message({"event": "rejected", "call_id": self.call_id, "reason": "call_manager_full"})
                return
            
            self.call_active = True
            self.vad = VoiceActivityDetector(self.call_id)
            self.audio_buffer_manager = AudioBufferManager(self.call_id)
            self.speech_buffer = bytearray()
            
            logger.info(f"Call {self.call_id}: Started - Caller: {self.caller_id}")
            
            if not await self._start_gemini_session():
                logger.error(f"Call {self.call_id}: Failed to start Gemini session")
                await self._send_hangup()
                return
            
            await self._trigger_initial_greeting()
            await self._send_message({"event": "started", "call_id": self.call_id})

    async def _trigger_initial_greeting(self):
        try:
            if self.gemini_session:
                logger.info(f"Call {self.call_id}: Triggering Gemini's initial greeting...")
                await self.gemini_session.send_realtime_input(audio_stream_end=True)
        except Exception as e:
            logger.error(f"Call {self.call_id}: Failed to trigger initial greeting: {e}")

    async def _handle_media_info(self, data):
        media_info = data.get("data", {})
        input_rate = media_info.get("sample_rate", Config.INPUT_SAMPLE_RATE)
        if input_rate != self.input_sample_rate:
            self.input_sample_rate = input_rate
            logger.info(f"Call {self.call_id}: Updated input sample rate to {input_rate}Hz")

    async def _handle_audio_stream(self, audio_data: bytes):
        if not self.call_active or not audio_data: return

        if self.assistant_speaking: return
        
        time_since_assistant_spoke = time.time() - self.last_assistant_speech_end
        if self.last_assistant_speech_end > 0 and time_since_assistant_spoke < Config.ECHO_BLOCK_DURATION:
            return

        try:
            processed_audio = self.audio_processor.apply_bandpass_filter(
                np.frombuffer(audio_data, dtype=np.int16), self.input_sample_rate
            ).astype(np.int16).tobytes()
            
            await self.audio_buffer_manager.add_audio(processed_audio)
            
            while True:
                chunk = await self.audio_buffer_manager.get_chunk(Config.CHUNK_SIZE)
                if chunk is None: break
                
                is_speech, speech_ended = self.vad.process_chunk(chunk)
                
                if is_speech:
                    if not self.user_speaking:
                        self.user_speaking = True
                        # Prepend buffered audio to capture the start of speech
                        buffered_start = self.vad.get_buffered_audio()
                        self.speech_buffer.extend(buffered_start)
                        await self._send_to_gemini(buffered_start)

                    self.speech_buffer.extend(chunk)
                    await self._send_to_gemini(chunk)
                
                if speech_ended:
                    self.user_speaking = False
                    logger.info(f"Call {self.call_id}: USER FINISHED SPEAKING")
                    await self._send_speech_end_signal()
                    
        except Exception as e:
            logger.error(f"Call {self.call_id}: Audio stream processing error: {e}", exc_info=True)

    async def _send_to_gemini(self, audio_chunk: bytes):
        if not self.gemini_session or not audio_chunk: return
        try:
            if self.input_sample_rate != Config.INPUT_SAMPLE_RATE:
                audio_chunk = self.audio_processor.resample_audio(audio_chunk, self.input_sample_rate, Config.INPUT_SAMPLE_RATE)
            await self.gemini_session.send_realtime_input(media={"data": audio_chunk})
        except Exception as e:
            logger.error(f"Call {self.call_id}: Error sending to Gemini: {e}")
            if "closed" in str(e).lower():
                await self._restart_gemini_session()

    async def _send_speech_end_signal(self):
        if not self.gemini_session: return
        try:
            await self.gemini_session.send_realtime_input(audio_stream_end=True)
            logger.info(f"Call {self.call_id}: Sent end-of-speech signal to Gemini")
            self.speech_buffer.clear()
            await self._cancel_task('response_timeout_task')
            self.response_timeout_task = asyncio.create_task(self._response_timeout_handler())
        except Exception as e:
            logger.error(f"Call {self.call_id}: Error sending speech end signal: {e}")

    async def _response_timeout_handler(self):
        try:
            await asyncio.sleep(self.response_timeout)
            if not self.assistant_speaking:
                logger.warning(f"Call {self.call_id}: Response timeout after {self.response_timeout}s")
                self.vad.reset_state()
        except asyncio.CancelledError:
            pass

    async def _handle_call_end(self, data):
        logger.info(f"Call {self.call_id}: Call ended - Reason: {data.get('reason', 'normal')}")
        self.call_active = False
        await self._send_message({"event": "ended", "call_id": self.call_id})

    async def _start_gemini_session(self) -> bool:
        try:
            client = genai.Client(vertexai=True, project=Config.PROJECT_ID, location=Config.LOCATION)
            self._gemini_context_manager = client.aio.live.connect(model=Config.MODEL, config=self._get_gemini_config())
            self.gemini_session = await self._gemini_context_manager.__aenter__()
            self.gemini_response_task = asyncio.create_task(self._handle_gemini_responses())
            logger.info(f"Call {self.call_id}: Gemini session started")
            return True
        except Exception as e:
            logger.error(f"Call {self.call_id}: Gemini session failed: {e}", exc_info=True)
            return False

    def _get_gemini_config(self):
        # --- ADDED: Get the current date and format it ---
        current_date = datetime.now().strftime("%A, %B %d, %Y")

        return LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(prebuilt_voice_config=PrebuiltVoiceConfig(voice_name="AOEDE"))
            ),
            tools=[self._get_appointment_tools()],
            system_instruction=f"""
You are MAYA, a friendly and professional ai assistant for zappque medical hospital in KERALA(So always talk in malayalam but you can use ), specializing in appointment scheduling.
Your task is to assist callers in booking, rescheduling, or canceling medical appointments. To get the list of doctors available use list.
You knows the current date and time is {current_dt['formatted']}. so don't ask for it. If the caller asks for the date or time, provide it based on this information. 
Don't spell RECEPTIONIST. As it is a position, YOU are the RECEPTIONIST MAYA. 

CRITICAL LINGUISTIC RULE: You MUST speak in the natural, conversational style of a person from KERALA (Colloquial Malayalam).

CULTURAL & VOCABULARY RULE: When referring to doctor specialties or hospital departments, use ENGGLISH terms.

IMPORTANT BEHAVIORAL GUIDELINES:
- Speak naturally like a real human receptionist - use casual, conversational language
- You MUST respond ONLY in the MALAYALAM to the user
- Do NOT provide translations or repeat your answer in another language.
- Never provide a response in one language followed by the same response in another. Stick to ONLY MALAYALAM
- If the user is directly asking to book apointment, then don't say the greetings and all, just go diretly into bookings.
- Show empathy and concern for patients' needs
- Use contractions and natural speech patterns appropriate in MALAYALAM
- Add small personal touches and warmth to your responses
- Never mention you're an AI or assistant - you're Maya, the receptionist
- Sound genuinely helpful and caring, not robotic or overly formal
- Whenever repeating or confirming a phone number, ALWAYS say it digit by digit (for example: 9–8–7–6–5–4–3–2–1–0). NEVER group numbers into thousands, lakhs, crores, or treat them like money or quantities. Phone numbers are NOT amounts of money — they must be spoken ONLY as individual digits, one by one. Don't repeat the dictation unless the user asks for it. 

Phone Number confirmation guidelines:
- While confirming phone number, the digit 0 should be spelled as 'Zero' not 'Ooo' okay. 
- Be very attendive when noting the phone number, don't make any mistake and also without the user dictating you the phone number, 
- don't assume any random number, be very causious about it. 

YOUR ROLE:
- Greet every patient when they are connected (skip this or make it short if the user is asking to book directly)
- Introduce yourself as Maya, the hospital RECEPTIONIST
- Help patients book appointments with doctors
- Provide information about doctor specialties available
- Ask for necessary details in a conversational way
- Confirm appointment details clearly
- Also you can give details disease information
- Don't be stubbon, reply to the user on what they need.
- You can provide any sort of information to the user, but you main task is booking appointments

AVAILABLE SPECIALTIES (Use the Malayalam/Manglish terms in conversation):
- General medicine (പനി, ജലദോഷം, സാധാരണ അസുഖങ്ങൾ - 'സാധാരണ ഡോക്ടർ' / 'General Doctor')
- Cardiology (ഹൃദയം സംബന്ധമായ കാര്യങ്ങൾ - 'കാർഡിയോളജി' / 'Heart Doctor')  
- Dermatology (തൊലി, ചർമ്മ രോഗങ്ങൾ - 'ഡെർമറ്റോളജി' / 'Skin Doctor')
- Orthopedic (എല്ല്, ജോയിൻ്റ് സംബന്ധമായ കാര്യങ്ങൾ - 'ഓർത്തോ' / 'Bone Doctor')
- Pediatric (കുട്ടികൾക്കുള്ള ഡോക്ടർ - 'പീഡിയാട്രിക്' / 'Children's Doctor')

AFTER GETTING THE DETAILS:
Confirm the DATE, TIME, SERVICE, and CUSTOMER NAME with the caller before finalizing the appointment.

CONVERSATION STYLE:
- Start with a warm greeting (You can skip this and go to next step if the user is asking to book instead of saying hello)
- When booking appointments, ask for details naturally and conversationally
- Show you're working: "Let me check our schedule for you" (in MALAYALAM) and respond after a short pause
- Confirm details warmly
- End calls by saying thank you and wishing well.


"""
        )

    # ... Other methods like _get_appointment_tools, _handle_gemini_responses, etc. remain mostly the same ...
    def _get_appointment_tools(self):
        return Tool(
            function_declarations=[
                FunctionDeclaration(
                    name="book_appointment",
                    description="Book an appointment for a customer",
                    parameters={"type": "object","properties": {"date": {"type": "string"},"time": {"type": "string"},"service": {"type": "string"},"customer_name": {"type": "string"},"customer_phone": {"type": "string"},"notes": {"type": "string"}},"required": ["date", "time", "service", "customer_name"]}
                ),
                FunctionDeclaration(
                    name="check_availability",
                    description="Check available appointment slots",
                    parameters={"type": "object","properties": {"date": {"type": "string"},"service": {"type": "string"}},"required": ["date"]}
                ),
                FunctionDeclaration(
                    name="transfer_call",
                    description="Transfer call to another department or person",
                    parameters={"type": "object","properties": {"department": {"type": "string"},"reason": {"type": "string"}}, "required": ["department"]}
                )
            ]
        )

    async def _handle_gemini_responses(self):
        try:
            logger.info(f"Call {self.call_id}: Gemini response handler started")
            while self.call_active:
                try:
                    async for response in self.gemini_session.receive():
                        self.last_response_time = time.time()
                        if not self.call_active: break
                        
                        if hasattr(response, 'server_content'):
                            server_content = response.server_content
                            if hasattr(server_content, 'turn_complete') and server_content.turn_complete:
                                logger.info(f"Call {self.call_id}: Gemini turn complete")
                                self._reset_conversation_state()
                            
                            if hasattr(server_content, 'model_turn'):
                                for part in server_content.model_turn.parts:
                                    if hasattr(part, 'inline_data') and part.inline_data:
                                        await self._stream_audio_to_freeswitch(part.inline_data.data)
                                    elif hasattr(part, 'function_call'):
                                        await self._handle_function_call(part.function_call)
                                    elif hasattr(part, 'text'):
                                        logger.info(f"Call {self.call_id}: GEMINI SAYS: {part.text}")
                        
                        if hasattr(response, 'client_content') and hasattr(response.client_content, 'user_turn'):
                            for part in response.client_content.user_turn.parts:
                                if hasattr(part, 'text'):
                                    logger.info(f"Call {self.call_id}: USER SAID: {part.text}")
                except Exception as e:
                    logger.error(f"Call {self.call_id}: Error in Gemini response stream: {e}")
                    await self._restart_gemini_session()
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Call {self.call_id}: Gemini response handler failed: {e}", exc_info=True)

    async def _stream_audio_to_freeswitch(self, audio_data: bytes):
        async with self.lock:
            self.assistant_speaking = True
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # Simple volume boost
            max_val = np.max(np.abs(audio_np))
            if 0 < max_val < 16000:
                scale_factor = min(2.0, 20000 / max_val)
                audio_np = (audio_np * scale_factor).astype(np.int16)

            processed_audio = self.audio_processor.resample_audio(audio_np.tobytes(), self.output_sample_rate, self.input_sample_rate)
            
            chunk_duration = 0.020
            chunk_size = int(self.input_sample_rate * chunk_duration * 2)
            
            for i in range(0, len(processed_audio), chunk_size):
                if not self.call_active: break
                chunk = processed_audio[i:i+chunk_size]
                if len(chunk) < chunk_size: chunk += bytes(chunk_size - len(chunk))
                await self._send_audio_chunk(chunk)
                await asyncio.sleep(chunk_duration * 0.9)
        finally:
            async with self.lock:
                self.assistant_speaking = False
                self.last_assistant_speech_end = time.time()

    def _reset_conversation_state(self):
        self.user_speaking = False
        self.speech_buffer.clear()
        if self.vad: self.vad.reset_state()
        logger.info(f"Call {self.call_id}: Conversation state reset")

    async def _send_audio_chunk(self, audio_chunk: bytes):
        if self.websocket and not self.websocket.closed:
            try: await self.websocket.send(audio_chunk)
            except Exception as e: logger.error(f"Call {self.call_id}: Failed to send audio: {e}")

    async def _handle_function_call(self, function_call):
        function_name = function_call.name
        args = function_call.args
        logger.info(f"Call {self.call_id}: Function call: {function_name} with args: {args}")
        try:
            if function_name == "book_appointment": result = await self._book_appointment(args)
            elif function_name == "check_availability": result = await self._check_availability(args)
            elif function_name == "transfer_call": result = await self._transfer_call(args)
            else: result = {"error": f"Unknown function: {function_name}"}
            
            await self.gemini_session.send_realtime_input(function_response={"name": function_name, "response": result})
        except Exception as e:
            logger.error(f"Call {self.call_id}: Function call error: {e}")
            await self.gemini_session.send_realtime_input(function_response={"name": function_name, "response": {"error": str(e)}})

    async def _book_appointment(self, args):
        logger.info(f"Call {self.call_id}: Booking appointment: {args}")
        return {"success": True, "appointment_id": f"APT{int(time.time())}", "confirmation": f"Appointment confirmed for {args.get('customer_name')}"}

    async def _check_availability(self, args):
        logger.info(f"Call {self.call_id}: Checking availability: {args}")
        return {"date": args.get('date'), "available_slots": ["09:00", "10:30", "14:00"]}

    async def _transfer_call(self, args):
        logger.info(f"Call {self.call_id}: Transferring call to {args.get('department')}")
        await self._send_message({"event": "transfer", "call_id": self.call_id, "department": args.get('department')})
        return {"success": True, "message": f"Transferring you to {args.get('department')}"}

    async def _send_message(self, message):
        if self.websocket and not self.websocket.closed:
            try: await self.websocket.send(json.dumps(message))
            except Exception as e: logger.error(f"Call {self.call_id}: Failed to send message: {e}")

    async def _send_hangup(self):
        await self._send_message({"event": "hangup", "call_id": self.call_id, "reason": "system_error"})

    async def _restart_gemini_session(self):
        logger.info(f"Call {self.call_id}: Restarting Gemini session...")
        if self._gemini_context_manager:
            try: await self._gemini_context_manager.__aexit__(None, None, None)
            except Exception: pass
        await self._cancel_task('gemini_response_task')
        if await self._start_gemini_session():
            logger.info(f"Call {self.call_id}: Gemini session restarted successfully")
        else:
            logger.error(f"Call {self.call_id}: Failed to restart Gemini session")

    async def _cleanup_call(self):
        logger.info(f"Call {self.call_id}: Starting cleanup...")
        try:
            if self.call_id: await self.call_manager.remove_call(self.call_id)
            async with self.lock:
                self.call_active = False
                await self._cancel_task('response_timeout_task')
                await self._cancel_task('gemini_response_task')
                if self._gemini_context_manager:
                    try: await self._gemini_context_manager.__aexit__(None, None, None)
                    except Exception as e: logger.error(f"Call {self.call_id}: Error closing Gemini context: {e}")
        except Exception as e:
            logger.error(f"Call {self.call_id}: Cleanup error: {e}", exc_info=True)

    async def _cancel_task(self, task_attr_name: str):
        task = getattr(self, task_attr_name, None)
        if task and not task.done():
            task.cancel()
            try: await task
            except asyncio.CancelledError: pass

# --- Server and Main Entry Point (No changes needed) ---
class MultiplexedFreeSwitchWebSocketServer:
    def __init__(self):
        self.call_manager = CallManager()
        self.stats_task = None

    async def handle_connection(self, websocket, path):
        if not await self.call_manager.can_accept_call():
            logger.warning(f"Connection from {websocket.remote_address}: Rejected - max calls reached")
            await websocket.close(code=1013, reason="Maximum concurrent calls reached")
            return
        handler = FreeSwitchWebSocketHandler(self.call_manager)
        await handler.handle_freeswitch_connection(websocket, path)

    async def _print_periodic_stats(self):
        while True:
            await asyncio.sleep(60)
            stats = await self.call_manager.get_stats()
            logger.info(f"Server Stats - Uptime: {stats['uptime_hours']:.1f}h, Current: {stats['current_calls']}, Total: {stats['total_calls']}, Max: {stats['max_concurrent']}, Rejected: {stats['rejected_calls']}")

    async def start_server(self):
        logger.info(f"Starting server on {Config.WS_HOST}:{Config.WS_PORT}")
        self.stats_task = asyncio.create_task(self._print_periodic_stats())
        try:
            async with serve(self.handle_connection, Config.WS_HOST, Config.WS_PORT, ping_interval=30, max_size=None):
                await asyncio.Future()
        finally:
            await self._shutdown()

    async def _shutdown(self):
        logger.info("Shutting down server...")
        if self.stats_task: self.stats_task.cancel()
        await self.call_manager.cleanup_all_calls()
        logger.info("Server shutdown complete")

async def main():
    server = MultiplexedFreeSwitchWebSocketServer()
    try:
        await server.start_server()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")

if __name__ == "__main__":
    asyncio.run(main())