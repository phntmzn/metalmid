# gpu_chord_synthesis.py

# === METAL DEPENDENCIES ===
import objc
from Cocoa import NSObject
from Metal import MTLCreateSystemDefaultDevice, MTLResourceStorageModeShared
from Foundation import NSData

# === PYTHON STD + MIDIUTIL ===
import random
import os
import time
import ctypes
import threading
import struct
from pathlib import Path
from tqdm import tqdm
from multiprocessing import cpu_count
from midiutil import MIDIFile

# === CONFIGURATION ===
TEMPO = 157
TOTAL_FILES = 10
NUM_CHORDS = 4
USE_GPU = True
OUTPUT_DIR = Path.home() / "Desktop" / "MIDI_Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === CONSTANTS ===
BEATS_PER_BAR = 4
BARS = 4
DURATIONS = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125]

# === MOCKED b.py STRUCTURES ===
notes = {
    'C': 60, 'C#': 61, 'D': 62, 'D#': 63, 'E': 64, 'F': 65,
    'F#': 66, 'G': 67, 'G#': 68, 'A': 69, 'A#': 70, 'B': 71
}

chords = {
    'Major': [0, 4, 7], 'Minor': [0, 3, 7], 'Diminished': [0, 3, 6],
    'Augmented': [0, 4, 8], 'Sus2': [0, 2, 7], 'Sus4': [0, 5, 7],
    'Maj7': [0, 4, 7, 11], 'Min7': [0, 3, 7, 10]
}

# === METAL PROCESSOR ===
class SimpleMetalProcessor:
    def __init__(self):
        self.device = MTLCreateSystemDefaultDevice()
        self.queue = self.device.newCommandQueue()
        self.lock = threading.Lock()

        shader_source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void generate_random(device float *output [[buffer(0)]],
                                   constant uint &seed [[buffer(1)]],
                                   uint id [[thread_position_in_grid]]) {
            uint state = seed + id * 1664525u + 1013904223u;
            state = state * 1664525u + 1013904223u;
            output[id] = float(state) / 4294967295.0f;
        }
        """
        self.library = self.device.newLibraryWithSource_options_error_(shader_source, None, None)[0]
        func = self.library.newFunctionWithName_("generate_random")
        self.pipeline = self.device.newComputePipelineStateWithFunction_error_(func, None)[0]

    def generate_random_values(self, seed, count):
        with self.lock:
            buf_size = count * 4
            output_buf = self.device.newBufferWithLength_options_(buf_size, MTLResourceStorageModeShared)
            seed_val = ctypes.c_uint32(seed % (2**32))
            seed_buf = self.device.newBufferWithBytes_length_options_(ctypes.byref(seed_val), ctypes.sizeof(seed_val), MTLResourceStorageModeShared)
            cmd_buf = self.queue.commandBuffer()
            encoder = cmd_buf.computeCommandEncoder()
            encoder.setComputePipelineState_(self.pipeline)
            encoder.setBuffer_offset_atIndex_(output_buf, 0, 0)
            encoder.setBuffer_offset_atIndex_(seed_buf, 0, 1)

            threads = min(32, count)
            total_threads = ((count + threads - 1) // threads) * threads
            grid = (total_threads, 1, 1)
            tg_size = (threads, 1, 1)

            encoder.dispatchThreads_threadsPerThreadgroup_(grid, tg_size)
            encoder.endEncoding()
            cmd_buf.commit()
            cmd_buf.waitUntilCompleted()

            raw = output_buf.contents()
            array_type = ctypes.c_float * count
            return list(ctypes.cast(raw, ctypes.POINTER(array_type)).contents)

# === GPU VALUE GENERATOR ===
metal_processor = SimpleMetalProcessor()

def generate_values_gpu(seed, count, vtype):
    vals = metal_processor.generate_random_values(seed, count)
    if vtype == 'velocity': return [int(40 + v * 87) for v in vals]
    if vtype == 'note_offset': return [int(-12 + v * 24) for v in vals]
    if vtype == 'duration_index': return [int(v * len(DURATIONS)) for v in vals]
    return vals

# === MIDI FILE GENERATOR ===
def generate_midi(index):
    seed = int(time.time()) + index
    vels = generate_values_gpu(seed, NUM_CHORDS * 4, 'velocity')
    offs = generate_values_gpu(seed+1, NUM_CHORDS * 4, 'note_offset')
    durs = generate_values_gpu(seed+2, NUM_CHORDS, 'duration_index')

    midi = MIDIFile(1)
    midi.addTempo(0, 0, TEMPO)

    chord_names = list(chords.keys())
    time_pos = 0.0

    for i in range(NUM_CHORDS):
        cname = chord_names[i % len(chord_names)]
        cnotes = chords[cname]
        dur = DURATIONS[abs(durs[i]) % len(DURATIONS)]

        for j, base in enumerate(cnotes):
            idx = i * 4 + j
            note = max(0, min(127, base + offs[idx] + 60))
            vel = max(1, min(127, vels[idx]))
            midi.addNote(0, 0, note, time_pos, dur, vel)

        time_pos += dur

    fname = f"gpu_{index:05d}_c{NUM_CHORDS}_t{TEMPO}.mid"
    with open(OUTPUT_DIR / fname, "wb") as f:
        midi.writeFile(f)
    return f"✅ {fname}"

# === MAIN EXECUTION ===
def main():
    print("\n🎵 GPU-Assisted MIDI Generation: Locrian Chords")
    print("Output:", OUTPUT_DIR)
    for i in tqdm(range(TOTAL_FILES), desc="Generating"):
        print(generate_midi(i))

if __name__ == "__main__":
    main()
