# 🐛 Audio Resampling Bug Fix Report

**Platform:** Debian (GCP Server)
**Component:** SIP-to-AI Audio Bridge
**File Affected:** `bridgenew.py`
**Module:** `AudioResampler`

---

## 🧩 1. What Was the Issue?

The core issue was that audio captured from the SIP call (🎙️ **8kHz**) was being resampled to **16kHz** for AI processing — but the output turned into **pure silence** 🫢.

This problem occurred **only on the Debian (Linux) server**.
The **exact same code worked perfectly on macOS** during development.

### 🔍 Logs Showing the Issue:

```
Input (8kHz):  Bridging audio packet #13900: RMS = 2445.8 (Audio is present)
Output (16kHz): Queued audio: 320→640 bytes (8kHz→16kHz), RMS: 0.0 (Audio is lost/silence)
```

✅ **Conclusion:**
The issue occurred inside the `AudioResampler.resample()` function.

---

## 💥 2. Which Part of the Code Caused It?

The problem originated in the **`AudioResampler` class** inside **`bridgenew.py`**.

Specifically, the culprit was the following line:

```python
resampled_data = signal.resample_poly(audio_array_int16, to_rate, from_rate)
```

### ⚙️ Root Cause:

* On **Debian’s version of `scipy`**, passing a **`np.int16`** array directly to `scipy.signal.resample_poly()` caused it to **fail silently**, outputting an array of all zeros.
* On **macOS**, the same function handled `np.int16` correctly — hence the bug remained hidden during development.

---

## 💡 3. What Is the Solution?

The solution was to **explicitly convert the audio data to floating-point format (`np.float32`)** before resampling.
Floating-point is the **standard data type for signal processing**, and all `scipy` versions handle it correctly.

### ✅ Steps:

1. Convert `int16` → `float32` before resampling.
2. Perform resampling using `scipy.signal.resample_poly()`.
3. Clip values to the valid int16 range (`-32768` to `32767`).
4. Convert back to `int16` before returning as bytes.

---

## 🔁 4. Alternative Solutions Considered

| 🧠 Approach                              | ⚙️ Description                                                          | ⚖️ Trade-off                                        |
| ---------------------------------------- | ----------------------------------------------------------------------- | --------------------------------------------------- |
| **1. Use a Different `scipy` Algorithm** | Try `scipy.signal.resample` (Fourier-based) instead of `resample_poly`. | Might work, but less efficient for real-time audio. |
| **2. Use a Pure NumPy Fallback**         | Replace with `np.interp` (linear interpolation).                        | Works, but lower audio quality.                     |
| **3. Fix the Server Environment**        | Recompile `scipy` or its dependencies (OpenBLAS, ATLAS).                | Time-consuming, unreliable.                         |
| **4. Use Another Library**               | Add dependencies like `librosa` or `soxr`.                              | Heavy dependencies; unnecessary for this fix.       |

---

## 🛠️ 5. The Chosen Fix

✅ **We implemented Solution #3: Explicit Float Conversion**

### Reasons:

* 🧩 **Minimal code change** required.
* 🧪 **Proven to work** via diagnostic script (`audio_diagnostic.py`) on Debian.
* 🎧 **Maintains high-quality resampling** using `scipy.signal.resample_poly`.
* 🚫 **No new dependencies** added.

---

## 🧰 6. Code Changes

### 🧨 Old Code (Buggy)

```python
class AudioResampler:
    @staticmethod
    def resample(audio_data: bytes, from_rate: int, to_rate: int) -> bytes:
        try:
            # Convert to int16 numpy array
            audio_array_int16 = np.frombuffer(audio_data, dtype=np.int16)
            
            if from_rate == to_rate:
                return audio_data

            from scipy import signal
            
            # ❌ Fails silently on Debian
            resampled_data = signal.resample_poly(audio_array_int16, to_rate, from_rate)
            
            resampled_int16 = np.clip(resampled_data, -32768, 32767).astype(np.int16)
            return resampled_int16.tobytes()
            
        except Exception as e:
            logger.error(f"Resampling error: {e}")
            return audio_data
```

---

### ✅ New Code (Fixed)

```python
class AudioResampler:
    @staticmethod
    def resample(audio_data: bytes, from_rate: int, to_rate: int) -> bytes:
        try:
            if len(audio_data) % 2 != 0:
                audio_data = audio_data[:-1]
            
            if len(audio_data) == 0:
                return b''
            
            # 1️⃣ Convert to int16
            audio_array_int16 = np.frombuffer(audio_data, dtype=np.int16)
            
            if from_rate == to_rate:
                return audio_data
            
            # 🚀 FIX: Convert to float32 for safe resampling
            audio_array_float = audio_array_int16.astype(np.float32)
            
            from scipy import signal
            
            # 2️⃣ Resample safely as float
            resampled_float = signal.resample_poly(audio_array_float, to_rate, from_rate)
            
            # 3️⃣ Clip and convert back to int16
            resampled_int16 = np.clip(resampled_float, -32768, 32767).astype(np.int16)
            
            return resampled_int16.tobytes()
            
        except Exception as e:
            logger.error(f"Resampling error: {e}")
            return audio_data
```

---

## 🧾 Summary

| 🔹 Item        | 🔍 Description                                                |
| -------------- | ------------------------------------------------------------- |
| **Problem**    | Audio became silent after resampling (8kHz → 16kHz) on Debian |
| **Root Cause** | `scipy.signal.resample_poly` mishandled `np.int16` data type  |
| **Fix**        | Convert `np.int16` → `np.float32` before resampling           |
| **Tested On**  | ✅ macOS (Dev) & ✅ Debian (GCP Server)                         |
| **Result**     | Clean, audible 16kHz audio after fix 🎶                       |

---

## 🧠 Lessons Learned

* 🧩 Always ensure data is in **float format** before resampling.
* 🧪 Test audio pipelines on **target OS environments**, not just development machines.
* 📦 Keep dependencies minimal and compatible across systems.

---

**🧰 Author:** *Engineering Team – SIP-to-AI Audio Bridge*
**📅 Date:** *October 2025*
**💬 Status:** ✅ **Resolved and Verified**