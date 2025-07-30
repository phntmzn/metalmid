# shader_powered_midi.py

import objc
import ctypes
import threading
import time
from pathlib import Path
from Foundation import NSData
from Cocoa import NSObject
from Metal import (
    MTLCreateSystemDefaultDevice,
    MTLResourceStorageModeShared
)
from midiutil import MIDIFile
from tqdm import tqdm

# === MIDI THEORY DEFINITIONS ===
NOTES = {n: i for i, n in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"], start=60)}
CHORDS = {
    'Major': [0, 4, 7], 'Minor': [0, 3, 7],
    'Diminished': [0, 3, 6], 'Augmented': [0, 4, 8]
}

# === METAL SHADER ===
SHADER_SRC = """
#include <metal_stdlib>
using namespace metal;

kernel void harmonic_random(device float *output [[buffer(0)]],
                            constant uint &seed [[buffer(1)]],
                            uint id [[thread_position_in_grid]]) {
    uint state = seed + id * 1103515245u + 12345u;
    state ^= (state >> 11);
    state ^= (state << 7);
    state ^= (state >> 5);
    output[id] = float(state % 127) / 127.0;
}
"""

# === METAL PROCESSOR ===
class MetalShaderProcessor:
    def __init__(self):
        self.device = MTLCreateSystemDefaultDevice()
        self.queue = self.device.newCommandQueue()
        self.lock = threading.Lock()

        library, err = self.device.newLibraryWithSource_options_error_(SHADER_SRC, None, None)
        func = library.newFunctionWithName_("harmonic_random")
        self.pipeline = self.device.newComputePipelineStateWithFunction_error_(func, None)[0]

    def run(self, seed: int, count: int):
        with self.lock:
            buf_len = count * 4
            out_buf = self.device.newBufferWithLength_options_(buf_len, MTLResourceStorageModeShared)
            seed_val = ctypes.c_uint32(seed % (2 ** 32))
            seed_buf = self.device.newBufferWithBytes_length_options_(ctypes.byref(seed_val), ctypes.sizeof(seed_val), MTLResourceStorageModeShared)
            cmd = self.queue.commandBuffer()
            enc = cmd.computeCommandEncoder()
            enc.setComputePipelineState_(self.pipeline)
            enc.setBuffer_offset_atIndex_(out_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(seed_buf, 0, 1)
            grid = (count, 1, 1)
            group = (min(32, count), 1, 1)
            enc.dispatchThreads_threadsPerThreadgroup_(grid, group)
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
            raw = out_buf.contents()
            farray = ctypes.cast(raw, ctypes.POINTER(ctypes.c_float * count)).contents
            return list(farray)

# === MIDI GENERATOR ===
def generate_midi(seed, processor, tempo=140, bars=4):
    midi = MIDIFile(1)
    midi.addTempo(0, 0, tempo)
    root_index = int(time.time() * 1000) % 12
    root_note = list(NOTES.values())[root_index % 12]
    
    values = processor.run(seed, bars * 4)
    
    for i, val in enumerate(values):
        chord_type = list(CHORDS.keys())[int(val * len(CHORDS)) % len(CHORDS)]
        intervals = CHORDS[chord_type]
        chord_root = root_note + (i % 3) * 2  # staggered motion
        start = i * 1.0
        dur = 1.0
        for j, offset in enumerate(intervals):
            midi.addNote(0, 0, chord_root + offset, start, dur, 90)
    return midi

# === MAIN ===
def main():
    OUTPUT_DIR = Path.home() / "Desktop" / "ShaderMIDIs"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    processor = MetalShaderProcessor()

    for i in tqdm(range(10), desc="Generating MIDI"):
        seed = int(time.time()) + i * 1337
        midi = generate_midi(seed, processor)
        fname = OUTPUT_DIR / f"shader_{i:04d}.mid"
        with open(fname, "wb") as f:
            midi.writeFile(f)
        print(f"✅ Saved {fname.name}")

if __name__ == "__main__":
    main()
