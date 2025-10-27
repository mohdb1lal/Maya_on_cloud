#!/usr/bin/env python3
"""
Audio Resampling Diagnostic Tool
Tests different resampling methods to identify the issue on GCP Debian
"""

import numpy as np
import sys
import platform

def test_scipy_resample():
    """Test scipy resampling"""
    print("\n=== Testing scipy.signal.resample_poly ===")
    try:
        from scipy import signal
        
        # Create test audio (8kHz, 320 bytes = 160 samples)
        samples = 160
        test_audio = np.random.randint(-1000, 1000, samples, dtype=np.int16)
        
        # Calculate input RMS
        input_rms = np.sqrt(np.mean(np.square(test_audio.astype(np.float32))))
        print(f"Input: {samples} samples, RMS: {input_rms:.2f}")
        
        # Method 1: Direct resampling (might lose data)
        print("\nMethod 1: Direct int16 resampling")
        resampled1 = signal.resample_poly(test_audio, 2, 1)  # 8kHz -> 16kHz
        resampled1 = np.clip(resampled1, -32768, 32767).astype(np.int16)
        rms1 = np.sqrt(np.mean(np.square(resampled1.astype(np.float32))))
        print(f"Output: {len(resampled1)} samples, RMS: {rms1:.2f}")
        print(f"First 10 samples: {resampled1[:10]}")
        
        # Method 2: Float conversion (recommended)
        print("\nMethod 2: Float conversion before resampling")
        audio_float = test_audio.astype(np.float32)
        resampled2 = signal.resample_poly(audio_float, 2, 1)
        resampled2 = np.clip(resampled2, -32768, 32767).astype(np.int16)
        rms2 = np.sqrt(np.mean(np.square(resampled2.astype(np.float32))))
        print(f"Output: {len(resampled2)} samples, RMS: {rms2:.2f}")
        print(f"First 10 samples: {resampled2[:10]}")
        
        # Method 3: With padding type
        print("\nMethod 3: With explicit padding")
        resampled3 = signal.resample_poly(audio_float, 2, 1, padtype='constant')
        resampled3 = np.clip(resampled3, -32768, 32767).astype(np.int16)
        rms3 = np.sqrt(np.mean(np.square(resampled3.astype(np.float32))))
        print(f"Output: {len(resampled3)} samples, RMS: {rms3:.2f}")
        print(f"First 10 samples: {resampled3[:10]}")
        
        return True
        
    except ImportError:
        print("scipy not available")
        return False
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_numpy_interp():
    """Test numpy interpolation"""
    print("\n=== Testing numpy interpolation ===")
    try:
        # Create test audio
        samples = 160
        test_audio = np.random.randint(-1000, 1000, samples, dtype=np.int16)
        
        # Calculate input RMS
        input_rms = np.sqrt(np.mean(np.square(test_audio.astype(np.float32))))
        print(f"Input: {samples} samples, RMS: {input_rms:.2f}")
        
        # Upsample using interpolation
        old_indices = np.arange(samples)
        new_indices = np.linspace(0, samples - 1, samples * 2)
        resampled = np.interp(new_indices, old_indices, test_audio)
        resampled = resampled.astype(np.int16)
        
        # Calculate output RMS
        output_rms = np.sqrt(np.mean(np.square(resampled.astype(np.float32))))
        print(f"Output: {len(resampled)} samples, RMS: {output_rms:.2f}")
        print(f"First 10 samples: {resampled[:10]}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_byte_conversion():
    """Test byte to numpy conversion"""
    print("\n=== Testing byte conversion ===")
    try:
        # Create test bytes (like from PJSIP)
        test_bytes = bytes([0x00, 0x10, 0xFF, 0x7F, 0x00, 0x80, 0xFF, 0xFF] * 40)  # 320 bytes
        print(f"Input bytes: {len(test_bytes)} bytes")
        
        # Convert to numpy
        audio_array = np.frombuffer(test_bytes, dtype=np.int16)
        print(f"Numpy array: {len(audio_array)} samples")
        print(f"First 10 samples: {audio_array[:10]}")
        
        # Calculate RMS
        rms = np.sqrt(np.mean(np.square(audio_array.astype(np.float32))))
        print(f"RMS: {rms:.2f}")
        
        # Test with all zeros
        zero_bytes = bytes(320)
        zero_array = np.frombuffer(zero_bytes, dtype=np.int16)
        zero_rms = np.sqrt(np.mean(np.square(zero_array.astype(np.float32))))
        print(f"\nAll zeros test - RMS: {zero_rms:.2f}")
        
        # Test with known pattern
        pattern = []
        for i in range(160):
            val = int(1000 * np.sin(2 * np.pi * i / 20))
            pattern.extend([val & 0xFF, (val >> 8) & 0xFF])
        pattern_bytes = bytes(pattern)
        pattern_array = np.frombuffer(pattern_bytes, dtype=np.int16)
        pattern_rms = np.sqrt(np.mean(np.square(pattern_array.astype(np.float32))))
        print(f"Sine pattern test - RMS: {pattern_rms:.2f}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def check_environment():
    """Check the system environment"""
    print("=== System Information ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    # Check numpy
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
        print(f"NumPy int16 range: {np.iinfo(np.int16).min} to {np.iinfo(np.int16).max}")
    except ImportError:
        print("NumPy not available")
    
    # Check scipy
    try:
        import scipy
        print(f"SciPy version: {scipy.__version__}")
    except ImportError:
        print("SciPy not available")
    
    # Check endianness
    import struct
    print(f"System endianness: {sys.byteorder}")
    
def simulate_pjsip_audio():
    """Simulate the exact scenario from PJSIP"""
    print("\n=== Simulating PJSIP Audio Processing ===")
    
    # Simulate audio with various RMS values like in your logs
    test_cases = [
        ("Silent", np.zeros(160, dtype=np.int16)),
        ("Low volume", np.random.randint(-20, 20, 160, dtype=np.int16)),
        ("Normal volume", np.random.randint(-1000, 1000, 160, dtype=np.int16)),
        ("High volume", np.random.randint(-10000, 10000, 160, dtype=np.int16)),
    ]
    
    for name, audio in test_cases:
        print(f"\n{name} audio:")
        
        # Convert to bytes (as PJSIP provides)
        audio_bytes = audio.tobytes()
        
        # Calculate input RMS
        input_rms = np.sqrt(np.mean(np.square(audio.astype(np.float32))))
        print(f"  Input RMS: {input_rms:.2f}")
        
        # Test resampling
        try:
            from scipy import signal
            
            # Method that might fail on Linux
            audio_from_bytes = np.frombuffer(audio_bytes, dtype=np.int16)
            resampled_direct = signal.resample_poly(audio_from_bytes, 2, 1)
            direct_rms = np.sqrt(np.mean(np.square(resampled_direct.astype(np.float32))))
            
            # Fixed method
            audio_float = audio_from_bytes.astype(np.float32)
            resampled_float = signal.resample_poly(audio_float, 2, 1)
            resampled_fixed = np.clip(resampled_float, -32768, 32767).astype(np.int16)
            fixed_rms = np.sqrt(np.mean(np.square(resampled_fixed.astype(np.float32))))
            
            print(f"  Direct resample RMS: {direct_rms:.2f}")
            print(f"  Fixed resample RMS: {fixed_rms:.2f}")
            
            if direct_rms == 0 and input_rms > 0:
                print("  ⚠️ WARNING: Audio lost in direct resampling!")
            
        except Exception as e:
            print(f"  Error: {e}")

def main():
    """Run all diagnostics"""
    print("=" * 60)
    print("AUDIO RESAMPLING DIAGNOSTIC TOOL")
    print("=" * 60)
    
    check_environment()
    
    print("\n" + "=" * 60)
    print("RUNNING TESTS")
    print("=" * 60)
    
    # Run tests
    tests_passed = []
    tests_passed.append(("Byte conversion", test_byte_conversion()))
    tests_passed.append(("NumPy interpolation", test_numpy_interp()))
    tests_passed.append(("SciPy resample", test_scipy_resample()))
    
    # Simulate PJSIP scenario
    simulate_pjsip_audio()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in tests_passed:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    print("""
1. Make sure scipy is installed:
   pip install scipy

2. Always convert audio to float32 before resampling:
   audio_float = audio_int16.astype(np.float32)
   
3. Use resample_poly with explicit parameters:
   resampled = signal.resample_poly(audio_float, to_rate, from_rate, padtype='constant')
   
4. Convert back to int16 with proper clipping:
   output = np.clip(resampled, -32768, 32767).astype(np.int16)
   
5. If scipy is not available, use the numpy interpolation fallback.

The fixed version in bridgenew_fixed.py implements all these recommendations.
    """)

if __name__ == "__main__":
    main()