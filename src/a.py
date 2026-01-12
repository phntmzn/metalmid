# a.py — GPU-assisted MIDI generator (.mid output) - FIXED VERSION

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

# Direct import for Metal and dependencies
try:
    import objc
    from Cocoa import NSObject
    from Metal import MTLCreateSystemDefaultDevice, MTLResourceStorageModeShared
    from Foundation import NSData
    METAL_AVAILABLE = True
    print("🔧 Metal framework loaded successfully")
except ImportError as e:
    METAL_AVAILABLE = False
    print(f"⚠️  Metal not available: {e}")

# Mock the b.py imports since they're not provided
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

# Convert durations dict to list for index-based access
DURATIONS = list(time_value_durations.values())

# === CONFIGURATION ===
TOTAL_FILES = 1000
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
                    
                # Create seed buffer - allocate empty buffer and write to it
                seed_value = seed % (2**32)
                seed_buffer = self.device.newBufferWithLength_options_(
                    4,  # 4 bytes for uint32
                    MTLResourceStorageModeShared
                )
                if not seed_buffer:
                    raise RuntimeError("Could not create seed buffer")
                
                # Write seed value directly to buffer memory
                seed_ptr = seed_buffer.contents()
                if seed_ptr is None:
                    raise RuntimeError("Seed buffer has no contents")
                ctypes.memmove(seed_ptr, ctypes.byref(ctypes.c_uint32(seed_value)), 4)
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
                threads_per_group = min(32, count)
                grid_size = (count, 1, 1)
                threadgroup_size = (threads_per_group, 1, 1)
                encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)
                encoder.endEncoding()
                
                # Execute and wait
                cmd_buffer.commit()
                cmd_buffer.waitUntilCompleted()
                
                # Read results
                raw_ptr = output_buffer.contents()
                if raw_ptr is None:
                    raise RuntimeError("Output buffer has no contents")

                byte_count = count * 4
                data = ctypes.string_at(raw_ptr, byte_count)
                return list(struct.unpack(f"{count}f", data))
                
            except Exception as e:
                raise RuntimeError(f"Metal computation failed: {e}")


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
                    return None
            else:
                return None
        return metal_processor


def generate_values_gpu(seed, count, value_type):
    """GPU-based random value generation with CPU fallback"""
    # Ensure seed is positive
    seed = abs(seed) % (2**31)
    
    processor = get_metal_processor()
    
    # Try GPU first
    if processor:
        try:
            raw_values = processor.generate_random_values(seed, count)
            if raw_values and len(raw_values) == count:
                # Map to appropriate ranges
                mapped_values = []
                for val in raw_values:
                    if value_type == 'velocity':
                        mapped_values.append(int(40 + val * 87))
                    elif value_type == 'note_offset':
                        mapped_values.append(int(-12 + val * 24))
                    elif value_type == 'duration_index':
                        mapped_values.append(int(val * len(DURATIONS)))
                    elif value_type == 'chord_index':
                        mapped_values.append(int(val * len(chords)))
                    elif value_type == 'raw':
                        mapped_values.append(val)  # Keep raw 0-1 values
                    else:
                        mapped_values.append(val)
                return mapped_values, True  # Success, used GPU
        except Exception as e:
            print(f"⚠️  GPU generation failed, using CPU: {e}")
    
    # CPU fallback
    random.seed(seed)
    mapped_values = []
    for _ in range(count):
        val = random.random()
        if value_type == 'velocity':
            mapped_values.append(int(40 + val * 87))
        elif value_type == 'note_offset':
            mapped_values.append(int(-12 + val * 24))
        elif value_type == 'duration_index':
            mapped_values.append(int(val * len(DURATIONS)))
        elif value_type == 'chord_index':
            mapped_values.append(int(val * len(chords)))
        else:
            mapped_values.append(val)
    return mapped_values, False  # Used CPU


def generate_midi_file(args):
    """Generate a single MIDI file"""
    index, num_chords, use_gpu, base_time = args
    
    try:
        # Generate base seed using file index and base time to avoid collisions
        # Use prime number multiplier to ensure good distribution
        base_seed = abs((base_time + index * 104729) % (2**31))
        
        # Common chord progressions in various keys
        progressions = {
            'I-V-vi-IV': [0, 4, 5, 3],      # Pop progression (C-G-Am-F)
            'I-IV-V': [0, 3, 4],             # Classic rock (C-F-G)
            'ii-V-I': [1, 4, 0],             # Jazz turnaround (Dm-G-C)
            'I-vi-IV-V': [0, 5, 3, 4],       # 50s progression (C-Am-F-G)
            'I-IV-vi-V': [0, 3, 5, 4],       # Sensitive (C-F-Am-G)
            'vi-IV-I-V': [5, 3, 0, 4],       # Emotional (Am-F-C-G)
            'I-V-IV': [0, 4, 3],             # Simple rock (C-G-F)
            'I-bVII-IV': [0, 6, 3],          # Mixolydian rock (C-Bb-F)
        }
        
        # Major scale chord qualities (I-vii°)
        major_scale_chords = ['Major', 'Minor', 'Minor', 'Major', 'Major', 'Minor', 'Diminished']
        
        # Note names for key signature
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Generate random selections
        key_vals, gpu1 = generate_values_gpu(base_seed, 1, 'raw')
        prog_vals, gpu2 = generate_values_gpu(base_seed + 1, 1, 'raw')
        
        # Select key and progression
        root_note = int(key_vals[0] * 12) % 12  # 0-11 (C to B)
        root_name = note_names[root_note]
        
        prog_items = list(progressions.items())
        prog_name, selected_progression = prog_items[int(prog_vals[0] * len(prog_items)) % len(prog_items)]

        # Build a human-readable chord list for ONE cycle of the selected progression (for filename)
        degree_offsets = [0, 2, 4, 5, 7, 9, 11]  # Major scale intervals
        quality_suffix = {
            "Major": "",
            "Minor": "m",
            "Diminished": "dim",
            "Augmented": "aug",
            "Sus2": "sus2",
            "Sus4": "sus4",
            "Maj7": "maj7",
            "Min7": "m7",
        }

        chord_symbols = []
        for deg in selected_progression:
            base_quality = major_scale_chords[deg % 7]
            # Match the user's example style: use iii as a 7th chord when minor (e.g., Em7 in C major)
            qual = "Min7" if (deg % 7 == 2 and base_quality == "Minor") else base_quality

            chord_root = (root_note + degree_offsets[deg % 7]) % 12
            chord_root_name = note_names[chord_root]
            suffix = quality_suffix.get(qual, "")
            chord_symbols.append(f"{chord_root_name}{suffix}")

        filename_chords = "-".join(chord_symbols)
        
        # Generate enough repetitions of the progression to fill num_chords
        full_progression = []
        while len(full_progression) < num_chords:
            full_progression.extend(selected_progression)
        full_progression = full_progression[:num_chords]
        
        # Calculate exact counts needed
        total_notes = num_chords * 4  # Assume max 4 notes per chord
        
        # Generate parameter arrays with correct counts
        try:
            if use_gpu:
                velocities, gpu3 = generate_values_gpu(base_seed + 2, total_notes, 'velocity')
                duration_indices, gpu4 = generate_values_gpu(base_seed + 3, num_chords, 'duration_index')
                
                used_gpu = gpu1 and gpu2 and gpu3 and gpu4
                generation_method = "GPU" if used_gpu else "CPU"
            else:
                velocities, _ = generate_values_gpu(base_seed + 2, total_notes, 'velocity')
                duration_indices, _ = generate_values_gpu(base_seed + 3, num_chords, 'duration_index')
                generation_method = "CPU"
                
        except Exception as e:
            return f"❌ MIDI {index} parameter generation failed: {str(e)}"
        
        # Validate counts
        if (len(velocities) < total_notes or 
            len(duration_indices) < num_chords):
            return f"❌ MIDI {index} insufficient parameters"
        
        # Create MIDI file
        try:
            midi = MIDIFile(1)
            track = 0
            time_pos = 0.0
            
            midi.addTrackName(track, time_pos, f"{generation_method} {root_name} Major")
            midi.addTempo(track, time_pos, TEMPO)
            
            # Add key signature (0 = C major, positive = sharps, negative = flats)
            # For simplicity, use 0 for all keys (would need circle of fifths for proper implementation)
            midi.addKeySignature(track, time_pos, 0, 0, 0)  # C major
            
            channel = 0
            
        except Exception as e:
            return f"❌ MIDI {index} MIDI setup failed: {str(e)}"
        
        # Generate chords
        try:
            max_time = BARS * BEATS_PER_BAR
            note_counter = 0
            
            for chord_idx in range(num_chords):
                if time_pos >= max_time:
                    break
                
                # Get scale degree from progression
                scale_degree = full_progression[chord_idx]
                
                # Get chord quality based on scale degree
                chord_quality = major_scale_chords[scale_degree % 7]
                chord_intervals = chords[chord_quality]
                
                # Calculate root note (scale degree + key)
                chord_root = (root_note + degree_offsets[scale_degree % 7]) % 12
                
                # Get duration
                duration_idx = duration_indices[chord_idx] % len(DURATIONS)
                duration = float(DURATIONS[duration_idx])
                
                # Add notes
                for note_idx, interval in enumerate(chord_intervals):
                    if note_counter >= len(velocities):
                        break
                        
                    # Build note: middle C (60) + chord root + interval
                    final_note = 60 + chord_root + interval
                    final_note = max(0, min(127, final_note))
                    
                    velocity = max(1, min(127, velocities[note_counter]))
                    
                    midi.addNote(track, channel, final_note, time_pos, duration, velocity)
                    note_counter += 1
                
                time_pos += duration
                
        except Exception as e:
            return f"❌ MIDI {index} chord generation failed: {str(e)}"
        
        # Save file
        try:
            # Filename format requested: "Cmaj - C-Em7-Am-F 100bpm.mid"
            filename = f"{root_name}maj - {filename_chords} {TEMPO}bpm.mid"

            # Collision-safe: if the same key/progression repeats, append the index
            out_path = OUTPUT_DIR / filename
            if out_path.exists():
                filename = f"{root_name}maj - {filename_chords} {TEMPO}bpm_{index:05d}.mid"
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
    print("🎵 GPU-Assisted MIDI Generator v3 (FIXED)")
    print("=" * 40)
    
    # Configuration - use TOTAL_FILES constant
    total_files = TOTAL_FILES
    num_chords = 8
    use_gpu = True
    
    print(f"Generating {total_files} MIDI files...")
    print(f"Chords per file: {num_chords}")
    print(f"Tempo: {TEMPO} BPM")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Worker threads: {POOL_SIZE}")
    print()
    
    # Test Metal availability
    if use_gpu and METAL_AVAILABLE:
        processor = get_metal_processor()
        if processor:
            print("🚀 GPU acceleration ready")
        else:
            print("⚠️  GPU unavailable, will use CPU fallback")
    else:
        print("💻 Using CPU generation")
    print()
    
    # Generate base time for seed generation
    base_time = int(time.time() * 1000)
    
    # Generate files with parallel processing
    args_list = [(i, num_chords, use_gpu, base_time) for i in range(total_files)]
    start_time = time.time()
    
    results = []
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=POOL_SIZE) as executor:
        # Submit all tasks and track with progress bar
        futures = [executor.submit(generate_midi_file, args) for args in args_list]
        
        # Collect results with progress bar
        for future in tqdm(futures, desc="Generating", total=total_files):
            results.append(future.result())
    
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
    
    print(f"✅ Generated: {len(successes)}/{total_files}")
    if gpu_files:
        print(f"  🚀 GPU: {len(gpu_files)}")
    if cpu_files:
        print(f"  💻 CPU: {len(cpu_files)}")
    
    if failures:
        print(f"❌ Failed: {len(failures)}")
        for failure in failures[:5]:  # Show first 5 failures
            print(f"  {failure}")
    
    print(f"\nFiles saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
