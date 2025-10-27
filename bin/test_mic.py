import pjsua2 as pj
import asyncio
import websockets
import json
import numpy as np
import time
import threading
import queue
import logging
from dataclasses import dataclass
from google.cloud import speech

# --- Configuration ---
@dataclass
class TestConfig:
    # --- IMPORTANT: UPDATE THESE ---
    SIP_USER: str = "4qjpoz2h"
    SIP_PASSWORD: str = "Admin@123"
    SIP_DOMAIN: str = "pbx.voxbaysolutions.com"
    # --- GOOGLE CLOUD SPEECH-TO-TEXT ---
    # Set up authentication by running `gcloud auth application-default login`
    # or by setting the GOOGLE_APPLICATION_CREDENTIALS environment variable.
    # e.g., export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/keyfile.json"
    GCP_PROJECT_ID: str = "your-gcp-project-id" # Optional, but recommended

    # --- TECHNICAL DETAILS (usually no need to change) ---
    SIP_PORT: int = 5260
    SIP_TRANSPORT_PORT: int = 5060
    WS_HOST: str = "127.0.0.1" # Runs locally on your instance
    WS_PORT: int = 8081
    SIP_SAMPLE_RATE: int = 8000
    STT_SAMPLE_RATE: int = 16000 # Google Speech-to-Text best practice
    SAMPLES_PER_FRAME: int = 160 # 20ms at 8kHz
    BITS_PER_SAMPLE: int = 16
    CHANNELS: int = 1
    AUTO_ANSWER: bool = True
    LOG_LEVEL: int = logging.INFO

# --- Logging Setup ---
def setup_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger('AudioEchoTest')

logger = setup_logging(TestConfig.LOG_LEVEL)

# --- Google Speech-to-Text Handler ---
class TranscriptionClient:
    def __init__(self, sample_rate):
        self.client = speech.SpeechClient()
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=sample_rate,
                language_code="en-US",
                enable_automatic_punctuation=True,
            ),
            interim_results=True,
        )
        self.requests = (
            speech.StreamingRecognizeRequest(audio_content=chunk)
            for chunk in iter(self.audio_generator)
        )
        self.audio_queue = asyncio.Queue()

    async def audio_generator(self):
        while True:
            chunk = await self.audio_queue.get()
            if chunk is None:
                break
            yield chunk

    async def transcribe_stream(self):
        logger.info("Starting Google STT transcription...")
        try:
            # Re-create the generator for each new transcription stream
            requests = (
                speech.StreamingRecognizeRequest(audio_content=chunk)
                async for chunk in self.audio_generator()
            )

            responses = self.client.streaming_recognize(
                config=self.streaming_config,
                requests=requests
            )

            for response in responses:
                if not response.results:
                    continue

                result = response.results[0]
                if not result.alternatives:
                    continue

                transcript = result.alternatives[0].transcript

                if result.is_final:
                    print(f"✅ FINAL: {transcript}")
                else:
                    print(f"💬 INTERIM: {transcript}", end='\r')

        except Exception as e:
            logger.error(f"Could not transcribe audio: {e}")
        finally:
            logger.info("Transcription stream ended.")

    async def add_audio(self, audio_chunk):
        await self.audio_queue.put(audio_chunk)

    def end_stream(self):
        asyncio.create_task(self.audio_queue.put(None))

# --- WebSocket Echo Server ---
class EchoServer:
    def __init__(self, config: TestConfig):
        self.config = config
        self.transcriber = TranscriptionClient(config.STT_SAMPLE_RATE)
        self.websocket = None

    async def handler(self, websocket, path):
        self.websocket = websocket
        logger.info(f"WebSocket client connected from {websocket.remote_address}")
        # Start the transcription task once a client connects
        transcription_task = asyncio.create_task(self.transcriber.transcribe_stream())

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # 1. Echo the audio back immediately
                    await websocket.send(message)

                    # 2. Resample for STT and send to transcriber
                    audio_16k = self.resample(message, self.config.SIP_SAMPLE_RATE, self.config.STT_SAMPLE_RATE)
                    await self.transcriber.add_audio(audio_16k)

        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket client disconnected.")
        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
        finally:
            self.transcriber.end_stream()
            await transcription_task # Wait for it to finish

    @staticmethod
    def resample(audio_data: bytes, from_rate: int, to_rate: int) -> bytes:
        if from_rate == to_rate:
            return audio_data
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            num_samples = len(audio_array)
            duration = num_samples / from_rate
            new_num_samples = int(duration * to_rate)
            resampled_array = np.interp(
                np.linspace(0, num_samples, new_num_samples),
                np.arange(num_samples),
                audio_array
            ).astype(np.int16)
            return resampled_array.tobytes()
        except Exception as e:
            logger.error(f"Resampling error: {e}")
            return b''

    async def start(self):
        logger.info(f"Starting WebSocket echo server on {self.config.WS_HOST}:{self.config.WS_PORT}")
        server = await websockets.serve(self.handler, self.config.WS_HOST, self.config.WS_PORT)
        await server.wait_closed()


# --- PJSIP Components (Adapted from bridge.py) ---

class AudioBridge(pj.AudioMediaPort):
    def __init__(self, config: TestConfig):
        pj.AudioMediaPort.__init__(self)
        self.config = config
        self.capture_queue = queue.Queue(maxsize=100)
        self.playback_queue = queue.Queue(maxsize=100)
        self.active = False
        fmt = pj.MediaFormatAudio()
        fmt.type = pj.PJMEDIA_TYPE_AUDIO
        fmt.clockRate = config.SIP_SAMPLE_RATE
        fmt.channelCount = config.CHANNELS
        fmt.bitsPerSample = config.BITS_PER_SAMPLE
        fmt.frameTimeUsec = (config.SAMPLES_PER_FRAME * 1000000) // config.SIP_SAMPLE_RATE
        self.createPort("echo_test_bridge", fmt)

    def onFrameRequested(self, frame):
        try:
            if self.active and not self.playback_queue.empty():
                audio_chunk = self.playback_queue.get_nowait()
                frame.size = len(audio_chunk)
                frame.buf = pj.ByteVector(audio_chunk)
            else:
                frame.size = self.config.SAMPLES_PER_FRAME * 2
                frame.buf = pj.ByteVector(b'\x00' * frame.size)
        except queue.Empty:
            frame.size = self.config.SAMPLES_PER_FRAME * 2
            frame.buf = pj.ByteVector(b'\x00' * frame.size)
        except Exception as e:
            logger.error(f"onFrameRequested error: {e}")

    def onFrameReceived(self, frame):
        if self.active and frame.size > 0:
            try:
                audio_data = bytes(frame.buf)
                self.capture_queue.put_nowait(audio_data)
            except queue.Full:
                pass # Drop frame if queue is full
            except Exception as e:
                logger.error(f"onFrameReceived error: {e}")

    def start(self): self.active = True
    def stop(self): self.active = False

class AICall(pj.Call):
    def __init__(self, account, config: TestConfig, call_id=pj.PJSUA_INVALID_ID):
        pj.Call.__init__(self, account, call_id)
        self.config = config
        self.audio_bridge: AudioBridge = None
        self.ws_client_task = None

    def onCallState(self, prm):
        ci = self.getInfo()
        logger.info(f"Call state: {ci.stateText}")
        if ci.state == pj.PJSIP_INV_STATE_DISCONNECTED:
            if self.ws_client_task:
                self.ws_client_task.cancel()
            if self.audio_bridge:
                self.audio_bridge.stop()

    def onCallMediaState(self, prm):
        ci = self.getInfo()
        for mi in ci.media:
            if mi.type == pj.PJMEDIA_TYPE_AUDIO and mi.status == pj.PJSUA_CALL_MEDIA_ACTIVE:
                logger.info("Audio media is active")
                aud_media = self.getAudioMedia(mi.index)
                self.audio_bridge = AudioBridge(self.config)
                aud_media.startTransmit(self.audio_bridge)
                self.audio_bridge.startTransmit(aud_media)
                self.audio_bridge.start()
                self.ws_client_task = asyncio.run_coroutine_threadsafe(self.run_websocket_client(), loop=asyncio.get_event_loop())
                logger.info("Audio bridge connected and WebSocket client started.")

    async def run_websocket_client(self):
        uri = f"ws://{self.config.WS_HOST}:{self.config.WS_PORT}"
        try:
            async with websockets.connect(uri) as websocket:
                logger.info(f"WebSocket client connected to {uri}")
                # Create two tasks: one for sending, one for receiving
                send_task = asyncio.create_task(self.send_audio(websocket))
                receive_task = asyncio.create_task(self.receive_audio(websocket))
                await asyncio.gather(send_task, receive_task)
        except Exception as e:
            logger.error(f"WebSocket client error: {e}")

    async def send_audio(self, websocket):
        while self.audio_bridge.active:
            try:
                audio_data = await asyncio.to_thread(self.audio_bridge.capture_queue.get, timeout=1.0)
                await websocket.send(audio_data)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error sending audio: {e}")
                break

    async def receive_audio(self, websocket):
        async for message in websocket:
            try:
                if isinstance(message, bytes):
                    await asyncio.to_thread(self.audio_bridge.playback_queue.put, message, timeout=1.0)
            except queue.Full:
                logger.warning("Playback queue full, dropping audio frame.")
            except Exception as e:
                logger.error(f"Error receiving audio: {e}")
                break

class AISipAccount(pj.Account):
    def __init__(self, config: TestConfig):
        pj.Account.__init__(self)
        self.config = config

    def onRegState(self, prm):
        logger.info(f"SIP Registration: {self.getInfo().regStatusText}")

    def onIncomingCall(self, prm):
        logger.info("Incoming Call...")
        call = AICall(self, self.config, prm.callId)
        if self.config.AUTO_ANSWER:
            call_prm = pj.CallOpParam()
            call_prm.statusCode = 200
            call.answer(call_prm)
        return call

class SipThread(threading.Thread):
    def __init__(self, config: TestConfig):
        threading.Thread.__init__(self)
        self.daemon = True
        self.config = config
        self.endpoint = pj.Endpoint()
        self.account = None

    def run(self):
        try:
            self.endpoint.libCreate()
            ep_cfg = pj.EpConfig()
            ep_cfg.logConfig.level = 0 # Disable PJSIP's own logging
            self.endpoint.libInit(ep_cfg)

            transport_cfg = pj.TransportConfig()
            transport_cfg.port = self.config.SIP_TRANSPORT_PORT
            self.endpoint.transportCreate(pj.PJSIP_TRANSPORT_UDP, transport_cfg)
            self.endpoint.libStart()

            acc_cfg = pj.AccountConfig()
            acc_cfg.idUri = f"sip:{self.config.SIP_USER}@{self.config.SIP_DOMAIN}"
            acc_cfg.regConfig.registrarUri = f"sip:{self.config.SIP_DOMAIN}:{self.config.SIP_PORT}"
            cred = pj.AuthCredInfo("digest", "*", self.config.SIP_USER, 0, self.config.SIP_PASSWORD)
            acc_cfg.sipConfig.authCreds.append(cred)

            self.account = AISipAccount(self.config)
            self.account.create(acc_cfg)

            logger.info("SIP Thread started. Waiting for calls...")
            while True:
                self.endpoint.libHandleEvents(20) # 20ms
        except Exception as e:
            logger.error(f"SIP thread error: {e}")
            self.endpoint.libDestroy()

# --- Main Application ---
async def main():
    config = TestConfig()
    
    # 1. Start the SIP thread
    sip_thread = SipThread(config)
    sip_thread.start()
    
    # 2. Start the WebSocket server
    echo_server = EchoServer(config)
    
    logger.info("\n" + "="*50)
    logger.info("AUDIO ECHO & TRANSCRIPTION TEST")
    logger.info(f"📞 SIP User: {config.SIP_USER}@{config.SIP_DOMAIN}")
    logger.info("🚀 Ready to receive calls. Press Ctrl+C to exit.")
    logger.info("="*50 + "\n")

    await echo_server.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down.")
