#!/usr/bin/env python3
"""
RMS Monitor for PJSUA Phone Calls - Working Version
Uses conference bridge statistics - no custom media ports
"""

import pjsua2 as pj
import numpy as np
import time
import threading
import logging
from datetime import datetime
from typing import Optional
from collections import deque

# ============================================================================
# CONFIGURATION
# ============================================================================

class RMSMonitorConfig:
    """Configuration for RMS monitoring"""
    # SIP Configuration
    SIP_USER: str = "4qjpoz2h"
    SIP_PASSWORD: str = "Admin@123"
    SIP_DOMAIN: str = "pbx.voxbaysolutions.com"
    SIP_PORT: int = 5260
    SIP_TRANSPORT_PORT: int = 5060
    
    # Audio Configuration
    SAMPLE_RATE: int = 8000
    CHANNELS: int = 1
    
    # RMS Display Configuration
    UPDATE_INTERVAL: float = 0.2  # Print RMS every 200ms
    SHOW_VISUAL_METER: bool = True
    METER_WIDTH: int = 40
    
    # Call Configuration
    AUTO_ANSWER: bool = True
    
    # Logging
    LOG_LEVEL: int = logging.INFO


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(level: int = logging.INFO):
    """Configure logging"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('rms_monitor.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('RMSMonitor')

logger = setup_logging(RMSMonitorConfig.LOG_LEVEL)


# ============================================================================
# RMS DISPLAY
# ============================================================================

class RMSDisplay:
    """Display RMS values"""
    
    @staticmethod
    def create_visual_meter(level: int, width: int = 40) -> str:
        """Create a visual meter from signal level (0-255)"""
        normalized = min(level / 255.0, 1.0)
        filled = int(normalized * width)
        empty = width - filled
        
        if level < 30:
            color = "⬜"
            bar_char = "░"
            status = "🔇 SILENCE"
        elif level < 80:
            color = "🟩"
            bar_char = "▒"
            status = "🔉 LOW"
        elif level < 150:
            color = "🟨"
            bar_char = "▓"
            status = "🔊 MEDIUM"
        else:
            color = "🟥"
            bar_char = "█"
            status = "📢 HIGH"
        
        meter = color + "[" + (bar_char * filled) + ("·" * empty) + "]"
        return status, meter
    
    @staticmethod
    def level_to_db(level: int) -> float:
        """Convert signal level (0-255) to approximate dB"""
        if level <= 0:
            return -100.0
        # Approximate conversion: 0 = -100dB, 255 = 0dB
        return -100.0 + (level / 255.0) * 100.0


# ============================================================================
# RMS STATISTICS TRACKER
# ============================================================================

class RMSStats:
    """Track RMS statistics over time"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.rx_values = deque(maxlen=window_size)
        self.tx_values = deque(maxlen=window_size)
        self.start_time = time.time()
        self.sample_count = 0
    
    def add_levels(self, rx_level: int, tx_level: int):
        """Add RX and TX levels"""
        self.rx_values.append(rx_level)
        self.tx_values.append(tx_level)
        self.sample_count += 1
    
    def get_stats(self, direction: str = "rx") -> dict:
        """Get statistics for a direction"""
        values = self.rx_values if direction == "rx" else self.tx_values
        
        if not values:
            return {
                'count': 0,
                'min': 0,
                'max': 0,
                'avg': 0.0,
                'current': 0
            }
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'current': values[-1] if values else 0
        }
    
    def print_summary(self):
        """Print summary statistics"""
        duration = time.time() - self.start_time
        
        rx_stats = self.get_stats("rx")
        tx_stats = self.get_stats("tx")
        
        print("\n" + "=" * 80)
        print("📊 AUDIO LEVEL STATISTICS SUMMARY")
        print("=" * 80)
        print(f"⏱️  Call Duration: {duration:.1f} seconds")
        print(f"📈 Total Samples: {self.sample_count}")
        print()
        print("📥 RX AUDIO (from caller to you):")
        print(f"   Samples: {rx_stats['count']}")
        print(f"   Min Level: {rx_stats['min']}")
        print(f"   Max Level: {rx_stats['max']}")
        print(f"   Avg Level: {rx_stats['avg']:.2f}")
        print()
        print("📤 TX AUDIO (from you to caller):")
        print(f"   Samples: {tx_stats['count']}")
        print(f"   Min Level: {tx_stats['min']}")
        print(f"   Max Level: {tx_stats['max']}")
        print(f"   Avg Level: {tx_stats['avg']:.2f}")
        print("=" * 80)


# ============================================================================
# CUSTOM CALL CLASS
# ============================================================================

class RMSCall(pj.Call):
    """Call class with RMS monitoring using stream statistics"""
    
    def __init__(self, account, call_id: int, config: RMSMonitorConfig, stats: RMSStats):
        super().__init__(account, call_id)
        self.config = config
        self.stats = stats
        self.monitoring_thread = None
        self.stop_monitoring = False
        logger.info(f"📞 Call object created (ID: {call_id})")
    
    def onCallState(self, prm: pj.OnCallStateParam):
        """Handle call state changes"""
        try:
            call_info = self.getInfo()
            state_text = call_info.stateText
            
            logger.info(f"📞 Call State: {state_text}")
            
            if call_info.state == pj.PJSIP_INV_STATE_CONFIRMED:
                logger.info("✅ Call CONNECTED - Starting audio level monitoring")
                print("\n" + "=" * 80)
                print("🎙️  AUDIO LEVEL MONITORING ACTIVE")
                print("=" * 80)
                print("Legend: 🔇 Silence | 🔉 Low | 🔊 Medium | 📢 High")
                print("=" * 80 + "\n")
                
                # Start monitoring in a separate thread
                self.stop_monitoring = False
                self.monitoring_thread = threading.Thread(target=self._monitor_audio_levels)
                self.monitoring_thread.daemon = True
                self.monitoring_thread.start()
                
            elif call_info.state == pj.PJSIP_INV_STATE_DISCONNECTED:
                logger.info("📞 Call DISCONNECTED")
                self.stop_monitoring = True
                if self.monitoring_thread:
                    self.monitoring_thread.join(timeout=1.0)
                self.stats.print_summary()
                
        except Exception as e:
            logger.error(f"Error in onCallState: {e}")
    
    def _monitor_audio_levels(self):
        """Monitor audio levels using stream statistics"""
        try:
            logger.info("🎧 Audio level monitoring thread started")
            
            while not self.stop_monitoring:
                try:
                    # Get call info
                    call_info = self.getInfo()
                    
                    # Check if call is still active
                    if call_info.state != pj.PJSIP_INV_STATE_CONFIRMED:
                        break
                    
                    # Look for active audio media
                    found_audio = False
                    
                    for media_idx in range(len(call_info.media)):
                        media = call_info.media[media_idx]
                        
                        if (media.type == pj.PJMEDIA_TYPE_AUDIO and 
                            media.status == pj.PJSUA_CALL_MEDIA_ACTIVE):
                            
                            try:
                                # Get stream statistics - this is safe and doesn't require custom ports
                                stream_stat = self.getStreamStat(media_idx)
                                
                                # PJSIP provides signal levels as integers (0-255 scale typically)
                                rx_level = stream_stat.rtcp.rxStat.loss  # RX signal level
                                tx_level = stream_stat.rtcp.txStat.loss  # TX signal level
                                
                                # Try to get actual signal levels if available
                                # Different PJSIP versions expose this differently
                                try:
                                    # Some versions have these attributes
                                    if hasattr(stream_stat, 'rxLevel'):
                                        rx_level = abs(stream_stat.rxLevel)
                                    if hasattr(stream_stat, 'txLevel'):
                                        tx_level = abs(stream_stat.txLevel)
                                except:
                                    pass
                                
                                # Store in stats
                                self.stats.add_levels(rx_level, tx_level)
                                
                                # Print the levels
                                self._print_levels(rx_level, tx_level)
                                
                                found_audio = True
                                break
                                
                            except Exception as e:
                                logger.debug(f"Could not get stream stats: {e}")
                    
                    if not found_audio:
                        logger.debug("No active audio media found")
                    
                    # Sleep for the update interval
                    time.sleep(self.config.UPDATE_INTERVAL)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(0.5)
            
            logger.info("🎧 Audio level monitoring thread stopped")
            
        except Exception as e:
            logger.error(f"Error in _monitor_audio_levels: {e}")
            import traceback
            traceback.print_exc()
    
    def _print_levels(self, rx_level: int, tx_level: int):
        """Print audio levels for both directions"""
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        
        # Process RX (incoming from caller)
        rx_status, rx_meter = RMSDisplay.create_visual_meter(rx_level, self.config.METER_WIDTH)
        rx_db = RMSDisplay.level_to_db(rx_level)
        
        # Process TX (outgoing to caller)
        tx_status, tx_meter = RMSDisplay.create_visual_meter(tx_level, self.config.METER_WIDTH)
        tx_db = RMSDisplay.level_to_db(tx_level)
        
        if self.config.SHOW_VISUAL_METER:
            print(f"[{timestamp}] RX (←) | Level: {rx_level:3d} | dB: {rx_db:6.1f} | {rx_status:9s} | {rx_meter}")
            print(f"[{timestamp}] TX (→) | Level: {tx_level:3d} | dB: {tx_db:6.1f} | {tx_status:9s} | {tx_meter}")
            print()  # Blank line for readability
        else:
            print(f"[{timestamp}] RX (←) | Level: {rx_level:3d} | dB: {rx_db:6.1f} | {rx_status}")
            print(f"[{timestamp}] TX (→) | Level: {tx_level:3d} | dB: {tx_db:6.1f} | {tx_status}")


# ============================================================================
# CUSTOM ACCOUNT CLASS
# ============================================================================

class RMSAccount(pj.Account):
    """Account class that creates calls with RMS monitoring"""
    
    def __init__(self, config: RMSMonitorConfig, stats: RMSStats):
        super().__init__()
        self.config = config
        self.stats = stats
        self.current_call: Optional[RMSCall] = None
    
    def onRegState(self, prm: pj.OnRegStateParam):
        """Handle registration state changes"""
        account_info = self.getInfo()
        status = account_info.regStatusText
        
        if account_info.regIsActive:
            logger.info(f"✅ Registration successful: {status}")
        else:
            logger.warning(f"⚠️  Registration status: {status}")
    
    def onIncomingCall(self, prm: pj.OnIncomingCallParam):
        """Handle incoming calls"""
        try:
            call = RMSCall(self, prm.callId, self.config, self.stats)
            self.current_call = call
            
            call_info = call.getInfo()
            caller = call_info.remoteUri
            
            logger.info(f"📞 Incoming call from: {caller}")
            
            if self.config.AUTO_ANSWER:
                call_prm = pj.CallOpParam()
                call_prm.statusCode = 200
                call.answer(call_prm)
                logger.info("✅ Call auto-answered")
            
        except Exception as e:
            logger.error(f"❌ Error handling incoming call: {e}")
            import traceback
            traceback.print_exc()


# ============================================================================
# MAIN RMS MONITOR APPLICATION
# ============================================================================

class RMSMonitor:
    """Main RMS monitoring application"""
    
    def __init__(self, config: RMSMonitorConfig):
        self.config = config
        self.endpoint: Optional[pj.Endpoint] = None
        self.account: Optional[RMSAccount] = None
        self.stats = RMSStats()
        self.running = False
    
    def initialize(self) -> bool:
        """Initialize PJSUA2"""
        try:
            logger.info("🚀 Initializing RMS Monitor")
            
            # Create endpoint
            self.endpoint = pj.Endpoint()
            self.endpoint.libCreate()
            
            # Configure endpoint
            ep_cfg = pj.EpConfig()
            ep_cfg.logConfig.level = 3
            ep_cfg.logConfig.consoleLevel = 2  # Reduce console spam
            
            # Audio configuration - keep it simple
            ep_cfg.medConfig.clockRate = self.config.SAMPLE_RATE
            ep_cfg.medConfig.channelCount = self.config.CHANNELS
            
            # Initialize library
            self.endpoint.libInit(ep_cfg)
            logger.info(f"📚 PJSUA2 Version: {self.endpoint.libVersion().full}")
            
            # Create transport
            transport_cfg = pj.TransportConfig()
            transport_cfg.port = self.config.SIP_TRANSPORT_PORT
            
            try:
                self.endpoint.transportCreate(pj.PJSIP_TRANSPORT_UDP, transport_cfg)
                logger.info(f"✅ SIP transport on port {transport_cfg.port}")
            except:
                transport_cfg.port = self.config.SIP_TRANSPORT_PORT + 1
                self.endpoint.transportCreate(pj.PJSIP_TRANSPORT_UDP, transport_cfg)
                logger.info(f"✅ SIP transport on alternative port {transport_cfg.port}")
            
            # Start library
            self.endpoint.libStart()
            logger.info("✅ PJSUA2 library started")
            
            # Create account
            self._create_account()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_account(self):
        """Create SIP account"""
        try:
            acc_cfg = pj.AccountConfig()
            acc_cfg.idUri = f"sip:{self.config.SIP_USER}@{self.config.SIP_DOMAIN}"
            acc_cfg.regConfig.registrarUri = f"sip:{self.config.SIP_DOMAIN}:{self.config.SIP_PORT}"
            acc_cfg.regConfig.timeoutSec = 300
            
            # Authentication
            cred = pj.AuthCredInfo(
                "digest", "*", self.config.SIP_USER, 0, self.config.SIP_PASSWORD
            )
            acc_cfg.sipConfig.authCreds.append(cred)
            
            # Proxy
            acc_cfg.sipConfig.proxies.append(
                f"sip:{self.config.SIP_DOMAIN}:{self.config.SIP_PORT};transport=udp"
            )
            
            # Create account
            self.account = RMSAccount(self.config, self.stats)
            self.account.create(acc_cfg)
            
            logger.info(f"✅ Account created: {self.config.SIP_USER}@{self.config.SIP_DOMAIN}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create account: {e}")
            raise
    
    def run(self):
        """Run the monitor"""
        try:
            self.running = True
            
            print("\n" + "=" * 80)
            print("🎙️  AUDIO LEVEL MONITOR FOR PHONE CALLS")
            print("=" * 80)
            print(f"📞 SIP Account: {self.config.SIP_USER}@{self.config.SIP_DOMAIN}")
            print(f"🎵 Sample Rate: {self.config.SAMPLE_RATE} Hz")
            print(f"📊 Update Interval: {self.config.UPDATE_INTERVAL * 1000:.0f} ms")
            print("⏳ Waiting for incoming calls...")
            print("⌨️  Press Ctrl+C to stop")
            print("=" * 80 + "\n")
            
            while self.running:
                self.endpoint.libHandleEvents(10)
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            logger.info("\n⌨️  Keyboard interrupt received")
        except Exception as e:
            logger.error(f"❌ Runtime error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Shutdown gracefully"""
        logger.info("🛑 Shutting down RMS Monitor")
        self.running = False
        
        try:
            if self.account:
                del self.account
                logger.info("✅ Account destroyed")
            if self.endpoint:
                self.endpoint.libDestroy()
                del self.endpoint
                logger.info("✅ Endpoint destroyed")
            
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
║            AUDIO LEVEL MONITOR FOR PHONE CALLS               ║
║         Safe Implementation - No Custom Media Ports          ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Main entry point"""
    print_banner()
    
    # Create configuration
    config = RMSMonitorConfig()
    
    # Create and run monitor
    monitor = RMSMonitor(config)
    
    if monitor.initialize():
        monitor.run()
    else:
        logger.error("❌ Failed to initialize RMS monitor")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())