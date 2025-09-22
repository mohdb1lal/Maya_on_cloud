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
import struct
from datetime import datetime, timedelta

# NEW: Firebase Admin Imports
import firebase_admin
from firebase_admin import credentials, firestore

# --- Configuration ---
@dataclass
class Config:
    WS_HOST = "0.0.0.0"
    WS_PORT = 8081
    PROJECT_ID = "docbooking-9ec13" # Your Google Cloud Project ID
    LOCATION = "us-central1"
    MODEL = "gemini-2.0-flash-live-preview-04-09"
    INPUT_SAMPLE_RATE = 16000
    OUTPUT_SAMPLE_RATE = 24000
    CHUNK_SIZE = 640
    VAD_THRESHOLD = 300
    MIN_SPEECH_DURATION = 0.6
    SILENCE_DURATION = 1.2
    RESPONSE_TIMEOUT = 10.0
    AUDIO_BUFFER_SIZE = 8192
    FADE_DURATION = 0.01
    NOISE_GATE_THRESHOLD = 80
    ECHO_BLOCK_DURATION = 1.0
    MIN_USER_PAUSE = 0.2
    LOG_LEVEL: int = logging.INFO

# --- Logging Setup ---
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

logger = setup_logging(Config.LOG_LEVEL)
# ... (logging setup remains the same) ...

# NEW: Firebase Initialization
try:
    # IMPORTANT: Set this environment variable before running the script
    # export GOOGLE_APPLICATION_CREDENTIALS="serviceAccountKey.json"
    cred = credentials.ApplicationDefault()
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    logger.info("✅ Firebase connection successful.")
except Exception as e:
    logger.error(f"❌ Firebase initialization failed: {e}")
    db = None
