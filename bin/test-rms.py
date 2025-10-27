#!/usr/bin/env python3
"""
Simple SIP Audio RMS Monitor
Just captures audio from SIP calls and shows RMS values
"""

import pjsua2 as pj
import time
import numpy as np
import logging

# Configuration
SIP_USER = "4qjpoz2h"
SIP_PASSWORD = "Admin@123"
SIP_DOMAIN = "pbx.voxbaysolutions.com"
SIP_PORT = 5260

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger('RMS_Monitor')


class SimpleAudioPort(pj.AudioMediaPort):
    """Simple audio port that just shows RMS values"""
    
    def __init__(self):
        pj.AudioMediaPort.__init__(self)
        self.frame_count = 0
        
        # Create audio format
        fmt = pj.MediaFormatAudio()
        fmt.type = pj.PJMEDIA_TYPE_AUDIO
        fmt.clockRate = 8000
        fmt.channelCount = 1
        fmt.bitsPerSample = 16
        fmt.frameTimeUsec = 20000  # 20ms
        
        self.createPort("rms_monitor", fmt)
        logger.info("✅ Audio port created")
    
    def calculate_rms(self, audio_data):
        """Calculate RMS value"""
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
    
    def onFrameReceived(self, frame):
        """Called when audio is received from phone"""
        self.frame_count += 1
        
        try:
            if frame.size > 0:
                # Try different methods to extract audio
                audio_data = None
                
                # Method 1: Direct slice
                try:
                    if hasattr(frame, 'buf') and frame.buf is not None:
                        audio_data = bytes(frame.buf[:frame.size])
                except:
                    pass
                
                # Method 2: Element by element (for Linux)
                if audio_data is None:
                    try:
                        audio_data = bytes([frame.buf[i] for i in range(frame.size)])
                    except:
                        pass
                
                # Calculate and display RMS
                if audio_data:
                    rms = self.calculate_rms(audio_data)
                    
                    # Show every frame for first 20, then every 50th frame
                    if self.frame_count <= 20 or self.frame_count % 50 == 0:
                        logger.info(f"🎤 Frame #{self.frame_count}: RMS = {rms:.1f}, Size = {len(audio_data)} bytes")
                else:
                    if self.frame_count <= 5:
                        logger.error(f"❌ Frame #{self.frame_count}: Could not extract audio data!")
                        logger.error(f"   Frame attributes: {[a for a in dir(frame) if not a.startswith('_')]}")
                        
        except Exception as e:
            if self.frame_count <= 5:
                logger.error(f"❌ Error in frame #{self.frame_count}: {e}")
    
    def onFrameRequested(self, frame):
        """Called when phone needs audio - just send silence"""
        frame.size = 320  # 160 samples * 2 bytes
        frame.type = pj.PJMEDIA_FRAME_TYPE_AUDIO
        frame.buf = pj.ByteVector()
        for _ in range(320):
            frame.buf.append(0)


class SimpleCall(pj.Call):
    """Simple call handler"""
    
    def __init__(self, account, call_id=pj.PJSUA_INVALID_ID):
        pj.Call.__init__(self, account, call_id)
        self.audio_port = None
        self.connected = False
    
    def onCallState(self, prm):
        """Handle call state"""
        ci = self.getInfo()
        logger.info(f"📞 Call state: {ci.stateText} ({ci.state})")
        
        if ci.state == pj.PJSIP_INV_STATE_CONFIRMED:
            self.connected = True
            logger.info("✅ Call CONNECTED - audio should start soon")
        elif ci.state == pj.PJSIP_INV_STATE_DISCONNECTED:
            self.connected = False
            logger.info("📵 Call DISCONNECTED")
            if ci.lastStatusCode != 200:
                logger.warning(f"⚠️  Disconnect reason: {ci.lastReason} (code: {ci.lastStatusCode})")
    
    def onCallMediaState(self, prm):
        """Handle media state"""
        ci = self.getInfo()
        
        logger.info(f"🎵 Media state changed - checking {len(ci.media)} media streams")
        
        for mi in ci.media:
            logger.info(f"   Media #{mi.index}: type={mi.type}, status={mi.status}")
            
            if mi.type == pj.PJMEDIA_TYPE_AUDIO and mi.status == pj.PJSUA_CALL_MEDIA_ACTIVE:
                logger.info("🎵 Audio ACTIVE - starting RMS monitoring")
                
                try:
                    # Get call audio
                    aud_media = self.getAudioMedia(mi.index)
                    
                    # Create audio port
                    self.audio_port = SimpleAudioPort()
                    
                    # Connect phone → port (to capture audio)
                    aud_media.startTransmit(self.audio_port)
                    
                    # Connect port → phone (to send silence)
                    self.audio_port.startTransmit(aud_media)
                    
                    logger.info("✅ RMS monitoring started - speak into the phone!")
                    logger.info("="*60)
                    
                except Exception as e:
                    logger.error(f"❌ Failed to setup audio: {e}")
                    import traceback
                    traceback.print_exc()


class SimpleAccount(pj.Account):
    """Simple account handler"""
    
    def __init__(self):
        pj.Account.__init__(self)
        self.active_calls = {}
    
    def onRegState(self, prm):
        """Handle registration state"""
        ai = self.getInfo()
        if ai.regIsActive:
            logger.info("✅ SIP Registered successfully")
        else:
            logger.info(f"📞 Registration status: {ai.regStatusText}")
    
    def onIncomingCall(self, prm):
        """Handle incoming call"""
        logger.info("📞 INCOMING CALL - Auto-answering...")
        
        # Create call and store it to prevent garbage collection
        call = SimpleCall(self, prm.callId)
        self.active_calls[prm.callId] = call
        
        # Auto-answer
        call_prm = pj.CallOpParam()
        call_prm.statusCode = 200
        call.answer(call_prm)
        
        logger.info("✅ Call answered - waiting for audio...")
        
        return call


def main():
    print("\n" + "="*60)
    print("     SIP AUDIO RMS MONITOR")
    print("="*60)
    print("This script will:")
    print("1. Register with your SIP server")
    print("2. Auto-answer incoming calls")
    print("3. Show RMS values of captured audio")
    print("="*60 + "\n")
    
    try:
        # Create endpoint
        ep = pj.Endpoint()
        ep.libCreate()
        
        # Configure endpoint
        ep_cfg = pj.EpConfig()
        ep_cfg.logConfig.level = 3
        ep_cfg.logConfig.consoleLevel = 3
        
        # Audio config
        ep_cfg.medConfig.clockRate = 8000
        ep_cfg.medConfig.channelCount = 1
        
        # Initialize
        ep.libInit(ep_cfg)
        logger.info(f"📚 PJSUA2 Version: {ep.libVersion().full}")
        
        # Create transport
        transport_cfg = pj.TransportConfig()
        transport_cfg.port = 5060
        ep.transportCreate(pj.PJSIP_TRANSPORT_UDP, transport_cfg)
        logger.info(f"✅ SIP transport on port {transport_cfg.port}")
        
        # Start library
        ep.libStart()
        
        # Create account
        acc_cfg = pj.AccountConfig()
        acc_cfg.idUri = f"sip:{SIP_USER}@{SIP_DOMAIN}"
        acc_cfg.regConfig.registrarUri = f"sip:{SIP_DOMAIN}:{SIP_PORT}"
        
        # Authentication
        cred = pj.AuthCredInfo("digest", "*", SIP_USER, 0, SIP_PASSWORD)
        acc_cfg.sipConfig.authCreds.append(cred)
        
        # Create account
        acc = SimpleAccount()
        acc.create(acc_cfg)
        
        logger.info(f"📞 Registered as: {SIP_USER}@{SIP_DOMAIN}")
        logger.info("⏳ Waiting for incoming calls...")
        logger.info("💡 When you receive a call, speak into the phone to see RMS values")
        logger.info("⌨️  Press Ctrl+C to stop\n")
        
        # Main loop
        while True:
            ep.libHandleEvents(10)
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        logger.info("\n⌨️  Stopping...")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if 'acc' in locals():
                acc.shutdown()
            if 'ep' in locals():
                ep.libDestroy()
        except:
            pass
        
        logger.info("✅ Shutdown complete")


if __name__ == "__main__":
    main()