# Direct import for Metal and dependencies
import objc
from Cocoa import NSObject
from Metal import MTLCreateSystemDefaultDevice, MTLResourceStorageModeShared
from Foundation import NSData

METAL_AVAILABLE = True
print("🔧 Metal framework loaded successfully")
# a.py — GPU-assisted MIDI generator (.mid output) - Fixed Version

import random
import os
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from tqdm import tqdm
import threading
import ctypes
import struct

from midiutil import MIDIFile

# Mock the b.py imports since they're not provided
# You should replace these with your actual imports from b.py
notes = {
    'C': 60, 'C#': 61, 'D': 62, 'D#': 63, 'E': 64, 'F': 65,
    'F#': 66, 'G': 67, 'G#': 68, 'A': 69, 'A#': 70, 'B': 71
}

chords = {
    'Major': [0, 4, 7],
    'Minor': [0, 3, 7],
    'Diminished': [0, 3, 6],
    'Augmented': [0, 4, 8],
    'Sus2': [0, 2, 7],
    'Sus4': [0, 5, 7],
    'Maj7': [0, 4, 7, 11],
    'Min7': [0, 3, 7, 10]
}

time_value_durations = {
    "whole_note": 4.0,
    "half_note": 2.0,
    "quarter_note": 1.0,
    "eighth_note": 0.5,
    "sixteenth_note": 0.25,
    "thirty_second_note": 0.125
}

# === Additional definitions for scales and modes ===
scales = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10]
}

modes = {
    "ionian": scales["major"],
    "aeolian": scales["minor"],
    "dorian": scales["dorian"],
    "phrygian": scales["phrygian"],
    "lydian": scales["lydian"],
    "mixolydian": scales["mixolydian"],
    "locrian": scales["locrian"]
}

# Convert durations dict to list for index-based access
DURATIONS = list(time_value_durations.values())

# === CONFIGURATION ===
TOTAL_FILES = 10
TEMPO = 157
OUTPUT_DIR = Path.home() / "Desktop" / "MIDI_Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
POOL_SIZE = max(4, cpu_count())

# === CONSTANTS ===
BEATS_PER_MINUTE = TEMPO
DURATION_MINUTES = 2
TOTAL_BEATS = BEATS_PER_MINUTE * DURATION_MINUTES
BEATS_PER_BAR = 4
BARS = 4



class SimpleMetalProcessor:
    """Simplified Metal processor with safer buffer handling"""
    
    def __init__(self):
        if not METAL_AVAILABLE or MTLCreateSystemDefaultDevice is None:
            raise RuntimeError("Metal not available")
            
        self.device = MTLCreateSystemDefaultDevice()
        if not self.device:
            raise RuntimeError("Metal device not available")
        
        self.queue = self.device.newCommandQueue()
        if not self.queue:
            raise RuntimeError("Could not create Metal command queue")
            
        self.lock = threading.Lock()
        
        # Simple shader that generates pseudo-random numbers
        shader_source = """
        #include <metal_stdlib>
        using namespace metal;

        // === Scale and mode interval definitions ===
        constant int major_scale[7] = {0, 2, 4, 5, 7, 9, 11};
        constant int minor_scale[7] = {0, 2, 3, 5, 7, 8, 10};
        constant int dorian_mode[7] = {0, 2, 3, 5, 7, 9, 10};
        constant int phrygian_mode[7] = {0, 1, 3, 5, 7, 8, 10};
        constant int lydian_mode[7] = {0, 2, 4, 6, 7, 9, 11};
        constant int mixolydian_mode[7] = {0, 2, 4, 5, 7, 9, 10};
        constant int locrian_mode[7] = {0, 1, 3, 5, 6, 8, 10};

        kernel void generate_random(device float *output [[buffer(0)]],
                                   constant uint &seed [[buffer(1)]],
                                   uint id [[thread_position_in_grid]]) {
            // Simple LCG random number generator
            uint state = seed + id * 1664525u + 1013904223u;
            state = state * 1664525u + 1013904223u;
            
            // Convert to float between 0 and 1
            output[id] = float(state) / 4294967295.0f;
        }
        """
        
        try:
            # Compile shader using string-based compilation (safer)
            self.library = self.device.newLibraryWithSource_options_error_(
                shader_source, None, None
            )[0]
            
            if not self.library:
                raise RuntimeError("Failed to compile Metal shader")
                
            func = self.library.newFunctionWithName_("generate_random")
            if not func:
                raise RuntimeError("Could not find kernel function")
                
            self.pipeline = self.device.newComputePipelineStateWithFunction_error_(
                func, None
            )[0]
            
            if not self.pipeline:
                raise RuntimeError("Could not create compute pipeline")
                
            print("✅ Simple Metal processor initialized")
            
        except Exception as e:
            raise RuntimeError(f"Metal setup failed: {e}")
    
    def generate_random_values(self, seed, count):
        """Generate random float values between 0 and 1"""
        if count <= 0:
            return []
        if count > 4096:
            raise ValueError(f"Count too high for safe Metal dispatch: {count}")
        with self.lock:
            try:
                # Create output buffer
                buffer_size = count * 4  # 4 bytes per float
                output_buffer = self.device.newBufferWithLength_options_(
                    buffer_size, MTLResourceStorageModeShared
                )
                if not output_buffer:
                    raise RuntimeError("Could not create output buffer")
                # Create seed buffer (single uint32), pass pointer directly for correct Metal binding
                seed_value = ctypes.c_uint32(seed % (2**32))
                seed_buffer = self.device.newBufferWithBytes_length_options_(
                    ctypes.byref(seed_value), ctypes.sizeof(seed_value), MTLResourceStorageModeShared
                )
                if not seed_buffer:
                    raise RuntimeError("Could not create seed buffer")
                # Create command buffer and encoder
                cmd_buffer = self.queue.commandBuffer()
                if not cmd_buffer:
                    raise RuntimeError("Could not create command buffer")
                encoder = cmd_buffer.computeCommandEncoder()
                if not encoder:
                    raise RuntimeError("Could not create compute encoder")
                # Set up compute pass
                encoder.setComputePipelineState_(self.pipeline)
                encoder.setBuffer_offset_atIndex_(output_buffer, 0, 0)
                encoder.setBuffer_offset_atIndex_(seed_buffer, 0, 1)
                # Dispatch threads
                threads_per_group = min(32, count)  # Conservative thread group size
                # Calculate total_threads as the next multiple of threads_per_group >= count
                total_threads = ((count + threads_per_group - 1) // threads_per_group) * threads_per_group
                grid_size = self.MTLSizeMake(total_threads, 1, 1)
                threadgroup_size = self.MTLSizeMake(threads_per_group, 1, 1)
                encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)
                encoder.endEncoding()
                # Execute and wait
                cmd_buffer.commit()
                cmd_buffer.waitUntilCompleted()
                # Read results using safe buffer copy
                raw_ptr = output_buffer.contents()
                if raw_ptr is None:
                    raise RuntimeError("Output buffer has no contents")

                array_type = ctypes.c_float * count
                buf = ctypes.cast(raw_ptr, ctypes.POINTER(array_type)).contents
                return list(buf)
            except Exception as e:
                raise RuntimeError(f"Metal computation failed: {e}")
    
    def MTLSizeMake(self, width, height, depth):
        """Helper to create MTLSize"""
        try:
            size_type = objc.createStructType('MTLSize', b'{MTLSize=QQQ}', ['width', 'height', 'depth'])
            return size_type(width, height, depth)
        except:
            # Fallback if struct creation fails
            return (width, height, depth)

# Global processor instance
metal_processor = None
processor_lock = threading.Lock()

def get_metal_processor():
    """Get or create the Metal processor"""
    global metal_processor
    with processor_lock:
        if metal_processor is None:
            if METAL_AVAILABLE:
                try:
                    metal_processor = SimpleMetalProcessor()
                    return metal_processor
                except Exception as e:
                    print(f"⚠️  Metal processor creation failed: {e}")
                    raise RuntimeError("Metal processor not available")
            else:
                raise RuntimeError("Metal processor not available")
        return metal_processor

def generate_chord_pattern():
    """Generate a MIDI chord pattern using Locrian mode"""
    # Chromatic scale notes in semitones from C
    chromatic_scale = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_to_semitone = {note: i for i, note in enumerate(chromatic_scale)}
    semitone_to_note = {i: note for i, note in enumerate(chromatic_scale)}

    # === Use GPU-based random selection for root note and progression ===
    seed = int(time.time()) + 42  # Optional external seed

    # Metal-style Locrian progressions
    metal_locrian_progressions = [
        [0, 6, 5, 4],  # i° - bVII - bVI - V
        [0, 3, 1, 4],  # i° - iv - II - V
        [0, 6, 0],     # i° - bVII - i°
        [0, 1, 3, 5]   # i° - II - iv - bVI
    ]
    root_idx = generate_values_gpu(seed, 1, 'raw')[0]
    prog_idx = generate_values_gpu(seed + 1, 1, 'raw')[0]

    root_note = chromatic_scale[int(root_idx * len(chromatic_scale)) % len(chromatic_scale)]
    progression_degrees = metal_locrian_progressions[int(prog_idx * len(metal_locrian_progressions)) % len(metal_locrian_progressions)]
    # Old random selection lines (now replaced by GPU):
    # root_note = random.choice(chromatic_scale)
    # progression_degrees = random.choice(metal_locrian_progressions)

    # Locrian mode intervals in semitones from tonic
    locrian_intervals = [0, 1, 3, 5, 6, 8, 10]

    # Chord types for Locrian mode (triads)
    locrian_chord_types = [
        "Diminished",  # i°
        "Minor",       # II
        "Major",       # bIII
        "Minor",       # iv
        "Major",       # V
        "Major",       # bVI
        "Minor"        # bVII
    ]

    root_semitone = note_to_semitone[root_note]

    # Build Locrian scale notes based on root
    locrian_scale = []
    for interval in locrian_intervals:
        semitone = (root_semitone + interval) % 12
        locrian_scale.append(semitone_to_note[semitone])

    duration = time_value_durations["eighth_note"]
    steps_per_bar = 4  # quarter notes per bar

    midi = MIDIFile(1)
    midi.addTempo(0, 0, TEMPO)

    # === Generate 4-bar arpeggio ===
    steps_per_bar = 4
    duration = BEATS_PER_BAR / steps_per_bar  # each note gets one step

    for bar_index in range(BARS):  # 4 bars
        for step in range(steps_per_bar):
            time = float(bar_index * BEATS_PER_BAR + step * duration)
            prog_idx = (bar_index * steps_per_bar + step) % len(progression_degrees)
            scale_degree = progression_degrees[prog_idx]
            root_scale_note = locrian_scale[scale_degree]
            chord_type = locrian_chord_types[scale_degree]
            intervals = chords.get(chord_type, chords["Minor"])
            root_note_number = notes[root_scale_note]

            # Add arpeggiated notes: one note per step
            interval = intervals[step % len(intervals)]
            note = root_note_number + interval + 60
            midi.addNote(0, 0, note, time, duration, 100)
    return midi


def generate_values_gpu(seed, count, value_type):
    """GPU-based random value generation"""
    processor = get_metal_processor()
    if not processor:
        raise RuntimeError("Metal processor not available")
    try:
        # Get raw random values from GPU
        raw_values = processor.generate_random_values(seed, count)
        if not raw_values:
            raise RuntimeError("Metal processor not available")
        # Map to appropriate ranges
        mapped_values = []
        for val in raw_values:
            if value_type == 'velocity':
                mapped_values.append(int(40 + val * 87))
            elif value_type == 'note_offset':
                mapped_values.append(int(-12 + val * 24))
            elif value_type == 'duration_index':
                mapped_values.append(int(val * len(time_value_durations)))
            else:
                mapped_values.append(val)
        return mapped_values
    except Exception as e:
        print(f"⚠️  GPU generation failed: {e}")
        raise RuntimeError("Metal processor not available")

def generate_midi_file(args):
    """Generate a single MIDI file"""
    index, num_chords, use_gpu = args
    
    try:
        # Generate base seed
        base_seed = random.randint(0, 100000)
        # Generate parameter arrays
        try:
            if use_gpu:
                velocities = generate_values_gpu(base_seed, num_chords * 4, 'velocity')
                note_offsets = generate_values_gpu(base_seed + 1, num_chords * 4, 'note_offset')
                duration_indices = generate_values_gpu(base_seed + 2, num_chords, 'duration_index')
                generation_method = "GPU"
            else:
                raise RuntimeError("GPU not available; CPU fallback not allowed")
        except Exception as e:
            return f"❌ MIDI {index} parameter generation failed: {str(e)}"
        
        # Validate we have the right number of values
        if (len(velocities) < num_chords * 4 or 
            len(note_offsets) < num_chords * 4 or 
            len(duration_indices) < num_chords):
            return f"❌ MIDI {index} insufficient parameters: v={len(velocities)}, n={len(note_offsets)}, d={len(duration_indices)}"
        
        # Debug print for first file
        if index == 0:
            print(f"Debug - Velocities: {velocities[:4]}")
            print(f"Debug - Note offsets: {note_offsets[:4]}")
            print(f"Debug - Duration indices: {duration_indices[:2]}")
        
        # Create MIDI file
        try:
            midi = MIDIFile(1)
            track = 0
            time = 0.0
            
            midi.addTrackName(track, time, f"{generation_method} Track {index}")
            
            # Use fixed tempo
            tempo = TEMPO
            midi.addTempo(track, time, TEMPO)
            channel = 0
        except Exception as e:
            return f"❌ MIDI {index} MIDI setup failed: {str(e)}"
        
        # Generate chords
        try:
            chord_names = list(chords.keys())
            
            max_time = BARS * BEATS_PER_BAR
            for chord_idx in range(num_chords):
                if time >= max_time:
                    break
                # Select chord - use index instead of velocity to avoid issues
                chord_name = chord_names[chord_idx % len(chord_names)]
                chord_notes = chords[chord_name]

                # Get duration
                duration_idx = abs(int(duration_indices[chord_idx])) % len(DURATIONS)
                duration = float(DURATIONS[duration_idx])

                # Add notes
                for note_idx, base_note in enumerate(chord_notes):
                    # Calculate final parameters
                    array_idx = chord_idx * 4 + note_idx

                    if array_idx < len(note_offsets):
                        note_offset = int(note_offsets[array_idx])
                    else:
                        note_offset = 0

                    final_note = max(0, min(127, int(base_note) + note_offset + 60))  # Add base note (C4)

                    if array_idx < len(velocities):
                        velocity = max(1, min(127, abs(int(velocities[array_idx]))))
                    else:
                        velocity = 64

                    # Debug for first chord of first file
                    if index == 0 and chord_idx == 0 and note_idx == 0:
                        print(f"Debug - First note: {final_note}, velocity: {velocity}, time: {time}, duration: {duration}")

                    midi.addNote(track, channel, final_note, time, duration, velocity)

                time += duration
        except Exception as e:
            return f"❌ MIDI {index} chord generation failed: {str(e)}"
        
        # Save file
        try:
            method_prefix = generation_method.lower()
            filename = f"{method_prefix}_{index:05d}_c{num_chords}_t{tempo}.mid"
            out_path = OUTPUT_DIR / filename
            
            with open(out_path, "wb") as f:
                midi.writeFile(f)
            
            return f"✅ {filename}"
        except Exception as e:
            return f"❌ MIDI {index} file save failed: {str(e)}"
        
    except Exception as e:
        import traceback
        return f"❌ MIDI {index} unexpected error: {str(e)}\n{traceback.format_exc()}"

def main():
    """Main function"""
    print("🎵 GPU-Assisted MIDI Generator v2 (Fixed)")
    print("=" * 40)
    
    # Configuration
    total_files = 10
    num_chords = 4
    use_gpu = True  # Disable GPU by default due to Metal buffer issues
    max_workers = cpu_count()
    
    print(f"Generating {total_files} MIDI files...")
    print(f"Chords per file: {num_chords}")
    print(f"GPU enabled: {use_gpu and METAL_AVAILABLE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Worker threads: {max_workers}")
    print()
    
    # Test Metal availability
    if use_gpu and METAL_AVAILABLE:
        processor = get_metal_processor()
        if processor:
            print("🚀 GPU acceleration ready")
        else:
            raise RuntimeError("GPU acceleration unavailable")
    else:
        raise RuntimeError("GPU not available")
    print()
    # Generate files
    args_list = [(i, num_chords, use_gpu) for i in range(total_files)]
    start_time = time.time()
    results = []
    for args in tqdm(args_list, desc="Generating"):
        results.append(generate_midi_file(args))
    end_time = time.time()
    
    # Results
    print("\n" + "=" * 40)
    print("Generation Complete!")
    print(f"Time elapsed: {end_time - start_time:.2f} seconds")
    print(f"Average: {(end_time - start_time) / total_files:.3f} seconds per file")
    print()
    
    successes = [r for r in results if r.startswith("✅")]
    failures = [r for r in results if r.startswith("❌")]
    
    gpu_files = [r for r in successes if "gpu_" in r]
    cpu_files = [r for r in successes if "cpu_" in r]
    
    print(f"✅ Generated: {len(successes)}")
    if gpu_files:
        print(f"  🚀 GPU: {len(gpu_files)}")
    if cpu_files:
        print(f"  💻 CPU: {len(cpu_files)}")
    
    if failures:
        print(f"❌ Failed: {len(failures)}")
        for failure in failures[:3]:  # Show first 3 failures
            print(f"  {failure}")

if __name__ == "__main__":
    main()
