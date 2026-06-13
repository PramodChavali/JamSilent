import aubio
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
from datetime import datetime
import os
import time
import threading

# ---------------- SETTINGS ----------------
SR = 22050
BUFFER_SIZE = 256
HOP_SIZE = 64

INPUT_DEVICE  = 1
OUTPUT_DEVICE = 4

sd.default.latency = 'low'
sd.default.extra_settings = None

# ---------------- AUBIO PITCH DETECTOR ----------------
pitch_o = aubio.pitch("yinfast", BUFFER_SIZE, HOP_SIZE, SR)
pitch_o.set_unit("Hz")
pitch_o.set_silence(-40)

# ---------------- SHARED STATE ----------------
last_pitch = 0.0
conf_threshold = 0.5
play_conf_threshold = 0.8
play_level_threshold = -20.0

last_good_time = 0.0
HOLD_TIME = 0.05

_freq  = 0.0
_phase = 0.0

# ---------------- RECORDING STATE ----------------
is_recording = False
recorded_chunks = []

# ---------------- SET PITCH ----------------
def set_pitch(hz):
    global _freq
    _freq = hz

def trumpet_wave(phases):
    return (
        1.0 * np.sin(phases) +
        0.6 * np.sin(2 * phases) +
        0.35 * np.sin(3 * phases) +
        0.2 * np.sin(4 * phases) +
        0.12 * np.sin(5 * phases) +
        0.08 * np.sin(6 * phases)
    )

# ---------------- RECORDING FUNCTIONS ----------------
def save_recording():
    if not recorded_chunks:
        print("Nothing to save.")
        return

    audio = np.concatenate(recorded_chunks)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("recordings", exist_ok=True)
    filename = f"recordings/silentjam_{timestamp}.wav"

    wavfile.write(filename, SR, audio)
    print(f"Saved: {filename}")

# ---------------- OUTPUT CALLBACK ----------------
def output_callback(outdata, frames, time_info, status):
    global _phase

    freq = _freq

    if freq > 0:
        phase_inc = 2.0 * np.pi * freq / SR
        phases = _phase + phase_inc * np.arange(frames, dtype=np.float32)

        wave = trumpet_wave(phases)
        wave = np.tanh(wave * 2.2)
        wave *= 0.25

        _phase = float((phases[-1] + phase_inc) % (2.0 * np.pi))

        outdata[:, 0] = wave.astype(np.float32)
        outdata[:, 1] = outdata[:, 0]

        if is_recording:
            recorded_chunks.append(wave.astype(np.float32).copy())

    else:
        _phase = 0.0
        outdata.fill(0)

        if is_recording:
            recorded_chunks.append(np.zeros(frames, dtype=np.float32))

# ---------------- INPUT CALLBACK ----------------
def input_callback(indata, frames, time_info, status):
    global last_pitch, last_good_time

    audio = indata[:, 0].astype(np.float32)

    t0 = time.perf_counter()

    pitch = pitch_o(audio)[0]
    confidence = pitch_o.get_confidence()

    t1 = time.perf_counter()

    rms = np.mean(audio * audio)
    db  = 10.0 * np.log10(rms + 1e-12)

    now = t0

    if confidence > conf_threshold and pitch > 0:
        last_pitch = pitch
        last_good_time = now

    valid_recently = (now - last_good_time) < HOLD_TIME

    if confidence > play_conf_threshold and db > play_level_threshold and valid_recently:
        set_pitch(last_pitch)
    else:
        set_pitch(0)

# ---------------- LAUNCH ----------------
print("Starting Silent Jam...")

with sd.InputStream(
        device=INPUT_DEVICE,
        samplerate=SR,
        blocksize=HOP_SIZE,
        channels=1,
        callback=input_callback,
        latency='low'
), sd.OutputStream(
        device=OUTPUT_DEVICE,
        samplerate=SR,
        blocksize=HOP_SIZE,
        channels=2,
        callback=output_callback,
        latency='low'
):
    print("Controls: R = start/stop recording | Q = quit")

    while True:
        cmd = input("> ").strip().lower()

        if cmd == "r":
            if not is_recording:
                recorded_chunks.clear()
                is_recording = True
                print("Recording started...")
            else:
                is_recording = False
                print("Recording stopped. Saving...")
                save_recording()

        elif cmd == "q":
            if is_recording:
                is_recording = False
                save_recording()
            print("Goodbye! Thanks for using the :)")
            break

        else:
            print("Unknown command. R = record | Q = quit")