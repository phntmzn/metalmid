# metal_shader_sequencer.py

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

# === SEQUENCER DEFINITIONS ===
TEMPO = 132
STEP_COUNT = 16
TRACK_COUNT = 3
NOTE_RANGE = (60, 84)
OUTPUT_DIR = Path.home() / "Desktop" / "ShaderSequencer"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === METAL SHADER ===
SEQUENCER_SHADER = """
#include <metal_stdlib>
using namespace metal;

kernel void sequencer(device float *output [[buffer(0)]],
                      constant uint &seed [[buffer(1)]],
                      uint tid [[thread_position_in_grid]]) {
    uint s = seed + tid * 1664525u + 1013904223u;
    s ^= (s >> 13);
    s ^= (s << 17);
    s ^= (s >> 5);
    output[tid] = float(s % 128) / 128.0;
}
"""

class MetalSequencer:
    def __init__(self):
        self.device = MTLCreateSystemDefaultDevice()
        self.queue = self.device.newCommandQueue()
        self.lock = threading.Lock()

        lib, err = self.device.newLibraryWithSource_options_error_(SEQUENCER_SHADER, None, None)
        fn = lib.newFunctionWithName_("sequencer")
        self.pipeline = self.device.newComputePipelineStateWithFunction_error_(fn, None)[0]

    def generate_sequence(self, seed: int, size: int):
        with self.lock:
            out_buf = self.device.newBufferWithLength_options_(size * 4, MTLResourceStorageModeShared)
            seed_val = ctypes.c_uint32(seed % (2 ** 32))
            seed_buf = self.device.newBufferWithBytes_length_options_(ctypes.byref(seed_val), ctypes.sizeof(seed_val), MTLResourceStorageModeShared)
            cmd = self.queue.commandBuffer()
            enc = cmd.computeCommandEncoder()
            enc.setComputePipelineState_(self.pipeline)
            enc.setBuffer_offset_atIndex_(out_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(seed_buf, 0, 1)
            grid = (size, 1, 1)
            group = (min(32, size), 1, 1)
            enc.dispatchThreads_threadsPerThreadgroup_(grid, group)
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
            raw = out_buf.contents()
            buf = ctypes.cast(raw, ctypes.POINTER(ctypes.c_float * size)).contents
            return list(buf)

# === MIDI SEQUENCER TRACK ===
def generate_midi_sequence(seed: int, sequencer: MetalSequencer):
    midi = MIDIFile(1)
    midi.addTempo(0, 0, TEMPO)

    size = STEP_COUNT * TRACK_COUNT
    vals = sequencer.generate_sequence(seed, size)

    for track in range(TRACK_COUNT):
        for step in range(STEP_COUNT):
            idx = track * STEP_COUNT + step
            val = vals[idx]
            if val > 0.6:
                note = NOTE_RANGE[0] + int(val * (NOTE_RANGE[1] - NOTE_RANGE[0]))
                midi.addNote(0, track, note, step * 0.5, 0.5, 90)
    return midi

# === MAIN ===
def main():
    print("🎚️ Metal Shader Sequencer: 3-Track MIDI Generator")
    sequencer = MetalSequencer()

    for i in tqdm(range(8), desc="Sequencing"):
        seed = int(time.time()) + i * 777
        midi = generate_midi_sequence(seed, sequencer)
        fname = OUTPUT_DIR / f"sequencer_{i:04d}.mid"
        with open(fname, "wb") as f:
            midi.writeFile(f)
        print("✅", fname.name)

if __name__ == "__main__":
    main()
