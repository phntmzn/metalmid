# a.py

import os
import subprocess
import random
from pathlib import Path
from tempfile import NamedTemporaryFile
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from tqdm import tqdm
import numpy as np
from midiutil import MIDIFile

import objc
from Cocoa import NSObject
from Metal import *
from Metal import MTLCreateSystemDefaultDevice
from Foundation import NSData

from b import notes, chords, time_value_durations

def MTLSizeMake(width, height, depth):
    size = objc.createStructType('MTLSize', b'{MTLSize=QQQ}', ['width', 'height', 'depth'])
    return size(width, height, depth)

class MetalRenderer:
    def initMetal(self):
        self.device = MTLCreateSystemDefaultDevice()
        self.commandQueue = self.device.newCommandQueue()

        shader_source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void audioPostProcess(device float* inAudio  [[ buffer(0) ]],
                                     device float* outAudio [[ buffer(1) ]],
                                     constant uint& effectType [[ buffer(2) ]],
                                     uint id [[ thread_position_in_grid ]]) {
            float input = inAudio[id];
            float output = 0.0;

            switch (effectType) {
                case 0:
                    output = input;
                    break;
                case 1:
                    output = input * 0.5;
                    break;
                case 2:
                    output = clamp(input * 5.0, -1.0, 1.0);
                    break;
                case 3:
                    output = (id % 100 < 90) ? input : 0.0;
                    break;
                case 4:
                    output = -input;
                    break;
                case 5:
                    output = inAudio[id % 512];
                    break;
                default:
                    output = input;
                    break;
            }
            outAudio[id] = output;
        }
        """

        with NamedTemporaryFile(delete=False, suffix=".metal") as metal_file:
            metal_file.write(shader_source.encode('utf-8'))
            metal_file_path = Path(metal_file.name)

        air_path = metal_file_path.with_suffix(".air")
        metallib_path = metal_file_path.with_suffix(".metallib")

        subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", str(metal_file_path), "-o", str(air_path)],
            check=True
        )
        subprocess.run(
            ["xcrun", "metallib", str(air_path), "-o", str(metallib_path)],
            check=True
        )

        data = NSData.dataWithContentsOfFile_(str(metallib_path))
        self.library = self.device.newLibraryWithData_error_(data, None)
        self.kernel = self.library.newFunctionWithName_("audioPostProcess")
        self.pipeline = self.device.newComputePipelineStateWithFunction_error_(self.kernel, None)

    def processAudio(self, in_buffer, out_buffer, effect_type, sample_count):
        commandBuffer = self.commandQueue.commandBuffer()
        encoder = commandBuffer.computeCommandEncoder()

        encoder.setComputePipelineState_(self.pipeline)
        encoder.setBuffer_offset_atIndex_(in_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(out_buffer, 0, 1)

        effect_type_buf = self.device.newBufferWithLength_options_(4, 0)
        effect_type_ptr = objc.ObjCInstance(effect_type_buf.contents()).cast('I')
        effect_type_ptr[0] = effect_type
        encoder.setBuffer_offset_atIndex_(effect_type_buf, 0, 2)

        threads_per_threadgroup = MTLSizeMake(256, 1, 1)
        threadgroups = MTLSizeMake((sample_count + 255) // 256, 1, 1)

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(threadgroups, threads_per_threadgroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

    def generateVelocities(self, sample_count):
        self.initMetal()
        in_data = np.random.rand(sample_count).astype(np.float32)
        in_buf = self.device.newBufferWithBytes_length_options_(
            in_data.ctypes.data, in_data.nbytes, 0)
        out_buf = self.device.newBufferWithLength_options_(
            in_data.nbytes, 0)
        self.processAudio(in_buf, out_buf, 1, sample_count)
        out_ptr = objc.ObjCInstance(out_buf.contents()).cast('f')
        return np.frombuffer(out_ptr, dtype=np.float32, count=sample_count)

# === CONFIGURATION ===
TOTAL_FILES = 200
TEMPO = 157
OUTPUT_DIR = Path.home() / "Desktop" / "eva_ascii"
POOL_SIZE = max(4, cpu_count())
DURATION_MINUTES = 1 + 10 / 60
BEATS_PER_MINUTE = TEMPO
TOTAL_BEATS = BEATS_PER_MINUTE * DURATION_MINUTES
BEATS_PER_BAR = 4
BARS = TOTAL_BEATS // BEATS_PER_BAR

def chord_pattern(renderer=None) -> str:
    chromatic_scale = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_to_semitone = {note: i for i, note in enumerate(chromatic_scale)}
    semitone_to_note = {i: note for i, note in enumerate(chromatic_scale)}
    aeolian_intervals = [0, 2, 3, 5, 7, 8, 10]
    aeolian_chord_types = ["Minor", "Diminished", "Major", "Minor", "Minor", "Major", "Major"]

    root_note = random.choice(chromatic_scale)
    root_semitone = note_to_semitone[root_note]
    aeolian_scale = [semitone_to_note[(root_semitone + i) % 12] for i in aeolian_intervals]

    minor_progressions = [
        [0, 3, 4, 6],
        [0, 5, 3, 4],
        [0, 4, 6, 5],
        [0, 3, 0]
    ]
    progression_degrees = random.choice(minor_progressions)

    duration = time_value_durations["eighth_note"]
    total_steps = int(TOTAL_BEATS / duration)

    if renderer:
        velocity_array = np.clip(renderer.generateVelocities(total_steps) * 128, 50, 120).astype(int)
    else:
        velocity_array = np.clip(np.random.normal(90, 6, total_steps), 70, 110).astype(int)

    midi = MIDIFile(1)
    midi.addTempo(0, 0, TEMPO)

    drum_channel = 9
    snare = 38
    kick = 36
    hihat = 42
    crash = 49
    toms = [snare]

    for step in range(total_steps):
        vel = velocity_array[step]
        time = step * duration
        bar_index = step // 4

        if step % 16 in [12]:
            for i in range(3):
                trip_time = time + i * (duration / 3)
                midi.addNote(0, drum_channel, random.choice(toms + [snare]), trip_time, duration / 3, int(vel * 0.8))
        if step % 16 in [13]:
            for i in range(3):
                trip_time = time + i * (duration / 3)
                midi.addNote(0, drum_channel, kick, trip_time, duration / 3, int(vel * 0.75))

        prog_idx = bar_index % len(progression_degrees)
        scale_degree = progression_degrees[prog_idx]
        root_scale_note = aeolian_scale[scale_degree]
        chord_type = aeolian_chord_types[scale_degree]
        intervals = chords[chord_type]
        root_note_number = notes[root_scale_note]
        interval = intervals[step % len(intervals)]
        note = root_note_number + interval + 3
        midi.addNote(0, 0, note, time, duration, 100)

        midi.addNote(0, drum_channel, hihat, time, duration, int(vel * 0.7))
        midi.addNote(0, drum_channel, 51, time, duration, int(vel * 0.5))

        if step % 4 == 0:
            midi.addNote(0, drum_channel, kick, time, duration, vel)
            if step % 16 == 0:
                midi.addNote(0, drum_channel, 49, time, duration, int(vel * 0.9))

        if step % 16 == 2 or step % 16 == 10:
            midi.addNote(0, drum_channel, snare, time, duration, vel)

        if step % 4 == 1:
            for i in range(6):
                midi.addNote(0, drum_channel, snare, time + i * (duration / 6), duration / 6, int(vel * 0.5))

        if step % 8 == 6:
            midi.addNote(0, drum_channel, random.choice(toms), time, duration, int(vel * 0.9))

    tmp = NamedTemporaryFile(delete=False, suffix=".mid")
    with open(tmp.name, "wb") as f:
        midi.writeFile(f)

    return tmp.name

def render_one(index):
    midi_output = OUTPUT_DIR / f"{index:05}.mid"
    try:
        renderer = MetalRenderer()
        midi_path = chord_pattern(renderer)
        os.rename(midi_path, midi_output)
        print(f"✅ Exported {midi_output.name} (GPU)")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"🎼 Generating {TOTAL_FILES} MIDI files using Metal and multiprocessing...")
    with ProcessPoolExecutor(max_workers=POOL_SIZE) as executor:
        futures = [executor.submit(render_one, i) for i in range(TOTAL_FILES)]
        for f in tqdm(as_completed(futures), total=TOTAL_FILES):
            try:
                f.result()
            except Exception as e:
                print(f"❌ Task failed: {e}")

if __name__ == "__main__":
    main()
