# metal_arpeggio_engine.py

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

# === MIDI THEORY ===
notes = { 'C': 60, 'C#': 61, 'D': 62, 'D#': 63, 'E': 64, 'F': 65,
          'F#': 66, 'G': 67, 'G#': 68, 'A': 69, 'A#': 70, 'B': 71 }

chords = {
    'Major': [0, 4, 7], 'Minor': [0, 3, 7], 'Diminished': [0, 3, 6],
    'Augmented': [0, 4, 8], 'Sus2': [0, 2, 7], 'Sus4': [0, 5, 7],
    'Maj7': [0, 4, 7, 11], 'Min7': [0, 3, 7, 10]
}

scales = {
    "locrian": [0, 1, 3, 5, 6, 8, 10]
}

# === CONFIGURATION ===
TEMPO = 157
TOTAL_FILES = 10
BARS = 4
BEATS_PER_BAR = 4
OUTPUT_DIR = Path.home() / "Desktop" / "MIDI_Arps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    def generate(self, seed, count):
        with self.lock:
            buf_len = count * 4
            obuf = self.device.newBufferWithLength_options_(buf_len, MTLResourceStorageModeShared)
            sval = ctypes.c_uint32(seed % (2**32))
            sbuf = self.device.newBufferWithBytes_length_options_(ctypes.byref(sval), ctypes.sizeof(sval), MTLResourceStorageModeShared)
            cb = self.queue.commandBuffer()
            enc = cb.computeCommandEncoder()
            enc.setComputePipelineState_(self.pipeline)
            enc.setBuffer_offset_atIndex_(obuf, 0, 0)
            enc.setBuffer_offset_atIndex_(sbuf, 0, 1)
            grid = (count, 1, 1)
            tg = (min(32, count), 1, 1)
            enc.dispatchThreads_threadsPerThreadgroup_(grid, tg)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            data = obuf.contents()
            farray = ctypes.cast(data, ctypes.POINTER(ctypes.c_float * count)).contents
            return list(farray)

metal = SimpleMetalProcessor()

# === LOCRIAN ARPEGGIO GENERATOR ===
def generate_locrian_arpeggio(seed):
    chromatic = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_map = {note: i for i, note in enumerate(chromatic)}
    reverse_map = {i: note for note, i in note_map.items()}
    
    root_idx = int(metal.generate(seed, 1)[0] * 12) % 12
    prog_idx = int(metal.generate(seed + 1, 1)[0] * 4) % 4
    
    root_note = chromatic[root_idx]
    root_semitone = note_map[root_note]

    progressions = [[0,6,5,4], [0,3,1,4], [0,6,0], [0,1,3,5]]
    chord_types = ["Diminished", "Minor", "Major", "Minor", "Major", "Major", "Minor"]

    scale = [(root_semitone + i) % 12 for i in scales["locrian"]]
    scale_notes = [reverse_map[i] for i in scale]

    midi = MIDIFile(1)
    midi.addTempo(0, 0, TEMPO)
    
    prog = progressions[prog_idx % len(progressions)]
    step_dur = BEATS_PER_BAR / 4

    for bar in range(BARS):
        for step in range(4):
            time = bar * BEATS_PER_BAR + step * step_dur
            degree = prog[(bar*4 + step) % len(prog)]
            note_name = scale_notes[degree % len(scale_notes)]
            chord_type = chord_types[degree % len(chord_types)]
            root_midi = notes[note_name]
            intervals = chords.get(chord_type, [0, 3, 7])
            note = root_midi + intervals[step % len(intervals)] + 12
            midi.addNote(0, 0, note, time, step_dur, 100)
    return midi

# === MAIN ===
def main():
    print("\n🎹 Metal Arpeggio Engine — Locrian Mode")
    for i in tqdm(range(TOTAL_FILES)):
        midi = generate_locrian_arpeggio(seed=int(time.time()) + i)
        fname = f"arpeggio_{i:04d}_locrian.mid"
        with open(OUTPUT_DIR / fname, "wb") as f:
            midi.writeFile(f)
        print("✅", fname)

if __name__ == "__main__":
    main()
