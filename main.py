import aubio
import numpy as np
import sounddevice as sd
import time

# ---------------- SETTINGS ----------------
SR = 48000
BUFFER_SIZE = 1024   # slightly more stable than 512
HOP_SIZE = 256

# ---------------- AUBIO PITCH DETECTOR ----------------
pitch_o = aubio.pitch("yin", BUFFER_SIZE, HOP_SIZE, SR)
pitch_o.set_unit("Hz")
pitch_o.set_silence(-40)






#ihilhs

# ---------------- STATE (stability layer) ----------------
last_pitch = 0.0
smoothed_pitch = 0.0

alpha = 0.85              # smoothing strength (0.8–0.95 good range)
conf_threshold = 0.5      # lower = more continuous output

# ---------------- AUDIO CALLBACK ----------------
def callback(indata, frames, time_info, status):
    global last_pitch, smoothed_pitch

    audio = indata[:, 0].astype(np.float32)

    # --- timing start ---
    t0 = time.perf_counter()

    pitch = pitch_o(audio)[0]
    confidence = pitch_o.get_confidence()

    # compute decibel level from RMS
    rms = np.sqrt(np.mean(audio**2))
    db = 20.0 * np.log10(rms + 1e-12)

    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000

    # ---------------- STABILITY LOGIC ----------------

    # only update if confident
    if confidence > conf_threshold and pitch > 0:
        last_pitch = pitch

    # smoothing (always applied so output doesn't jump)
    smoothed_pitch = alpha * smoothed_pitch + (1 - alpha) * last_pitch

    if confidence > 0.8:
        print(
            f"Pitch: {smoothed_pitch:7.2f} Hz | "
            f"Conf: {confidence:.2f} | "
            f"Level: {db:5.1f} dBFS | "
            f"Algo: {latency_ms:.2f} ms"
        )

    else:
        print(
            f"not playing anything | Level: {db:5.1f} dBFS"
        )

# ---------------- STREAM ----------------
print("Starting stabilized YIN pitch detection...")

with sd.InputStream(
    device=25,
    samplerate=SR,
    blocksize=HOP_SIZE,
    channels=1,
    callback=callback
):
    input("Running... press Enter to stop\n")