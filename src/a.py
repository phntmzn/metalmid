# a.py — Simplified GPU-assisted MIDI generator (.mid output)

import random
import os
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from midiutil import MIDIFile
from b import notes, chords, time_value_durations

# Convert durations dict to list for index-based access
DURATIONS = list(time_value_durations.values())

OUTPUT_DIR = Path.home() / "Desktop" / "MIDI_Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Try to import Metal, but gracefully fall back if not available
METAL_AVAILABLE = False
MTLCreateSystemDefaultDevice = None
MTLResourceStorageModeShared = 0

try:
    import ctypes
    import struct
    import subprocess
    from tempfile import NamedTemporaryFile
    import threading
    
    import objc
    from Cocoa import NSObject
    from Metal import MTLCreateSystemDefaultDevice, MTLResourceStorageModeShared
    from Foundation import NSData
    
    METAL_AVAILABLE = True
    print("🔧 Metal framework loaded successfully")
except ImportError as e:
    print(f"⚠️  Metal framework not available: {e}")
except Exception as e:
    print(f"⚠️  Metal initialization failed: {e}")

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
            
        with self.lock:
            try:
                # Create output buffer
                buffer_size = count * 4  # 4 bytes per float
                output_buffer = self.device.newBufferWithLength_options_(
                    buffer_size, MTLResourceStorageModeShared
                )
                
                if not output_buffer:
                    raise RuntimeError("Could not create output buffer")
                
                # Create seed buffer (single uint32)
                seed_value = ctypes.c_uint32(seed % (2**32))
                seed_bytes = ctypes.string_at(ctypes.byref(seed_value), 4)
                seed_buffer = self.device.newBufferWithBytes_length_options_(
                    seed_bytes, 4, MTLResourceStorageModeShared
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
                num_groups = (count + threads_per_group - 1) // threads_per_group
                
                threadgroup_size = self.MTLSizeMake(threads_per_group, 1, 1)
                grid_size = self.MTLSizeMake(count, 1, 1)
                
                encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)
                encoder.endEncoding()
                
                # Execute and wait
                cmd_buffer.commit()
                cmd_buffer.waitUntilCompleted()
                
                # Read results
                ptr = output_buffer.contents()
                if not ptr:
                    raise RuntimeError("Could not access buffer contents")
                    
                # Convert to Python floats
                float_array = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float * count)).contents
                return [float(val) for val in float_array]
                
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
                    metal_processor = False
            else:
                metal_processor = False
        
        return metal_processor if metal_processor is not False else None

def generate_values_cpu(seed, count, value_type):
    """CPU fallback for random value generation"""
    if count <= 0:
        return []
        
    random.seed(seed)
    values = []
    
    for _ in range(count):
        rand_val = random.random()  # 0.0 to 1.0
        
        if value_type == 'velocity':
            # Map to MIDI velocity range (40-127)
            values.append(int(40 + rand_val * 87))
        elif value_type == 'note_offset':
            # Map to note offset range (-12 to +12)
            values.append(int(-12 + rand_val * 24))
        elif value_type == 'duration_index':
            # Map to duration index (0 to len-1)
            values.append(int(rand_val * len(time_value_durations)) % len(time_value_durations))
        else:
            values.append(rand_val)
    
    return values

def generate_values_gpu(seed, count, value_type):
    """GPU-based random value generation"""
    processor = get_metal_processor()
    if not processor:
        return generate_values_cpu(seed, count, value_type)
    
    try:
        # Get raw random values from GPU
        raw_values = processor.generate_random_values(seed, count)
        if not raw_values:
            return generate_values_cpu(seed, count, value_type)
        
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
        print(f"⚠️  GPU generation failed, using CPU: {e}")
        return generate_values_cpu(seed, count, value_type)

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
                duration_indices = generate_values_cpu(base_seed + 2, num_chords, 'duration_index')  # Use CPU for this
                generation_method = "GPU" if get_metal_processor() else "CPU"
            else:
                velocities = generate_values_cpu(base_seed, num_chords * 4, 'velocity')
                note_offsets = generate_values_cpu(base_seed + 1, num_chords * 4, 'note_offset')
                duration_indices = generate_values_cpu(base_seed + 2, num_chords, 'duration_index')
                generation_method = "CPU"
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
            
            # Vary tempo
            tempo = int(80 + (index % 80))
            midi.addTempo(track, time, tempo)
            channel = 0
        except Exception as e:
            return f"❌ MIDI {index} MIDI setup failed: {str(e)}"
        
        # Generate chords
        try:
            chord_names = list(chords.keys())
            
            for chord_idx in range(num_chords):
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

                    final_note = max(0, min(127, int(base_note) + note_offset))

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
    print("🎵 GPU-Assisted MIDI Generator v2")
    print("=" * 40)
    
    # Configuration
    total_files = 10
    num_chords = 8
    use_gpu = False  # Disable GPU by default due to Metal buffer issues
    max_workers = 4
    
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
            print("⚠️  GPU acceleration unavailable, using CPU")
    else:
        print("💻 Using CPU generation")
    print()
    
    # Generate files
    args_list = [(i, num_chords, use_gpu) for i in range(total_files)]
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(generate_midi_file, args_list),
            total=total_files,
            desc="Generating"
        ))
    
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
