import torch
from typing import Optional, Tuple, List
from models import build_model, generate_speech, list_available_voices
from tqdm.auto import tqdm
import soundfile as sf
from pathlib import Path
import numpy as np
import time
import os

# Constants
SAMPLE_RATE = 24000
DEFAULT_MODEL_PATH = os.path.abspath('kokoro-v1_0.pth')
DEFAULT_OUTPUT_FILE = os.path.abspath('output.wav')
DEFAULT_LANGUAGE = 'a'  # 'a' for American English, 'b' for British English
DEFAULT_TEXT = "Hello, welcome to this text-to-speech test."

# Configure tqdm for better Windows console support
tqdm.monitor_interval = 0

def print_menu():
    """Print the main menu options."""
    print("\n=== Kokoro TTS Menu ===")
    print("1. List available voices")
    print("2. Generate speech")
    print("3. Exit")
    return input("Select an option (1-3): ").strip()

def select_voice(voices: List[str]) -> str:
    """Interactive voice selection."""
    print("\nAvailable voices:")
    for i, voice in enumerate(voices, 1):
        print(f"{i}. {voice}")
    
    while True:
        try:
            choice = input("\nSelect a voice number (or press Enter for default 'af_bella'): ").strip()
            if not choice:
                return "af_bella"
            choice = int(choice)
            if 1 <= choice <= len(voices):
                return voices[choice - 1]
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def get_text_input() -> str:
    """Get text input from user."""
    print("\nEnter the text you want to convert to speech")
    print("(or press Enter for default text)")
    text = input("> ").strip()
    return text if text else DEFAULT_TEXT

def get_speed() -> float:
    """Get speech speed from user."""
    while True:
        try:
            speed = input("\nEnter speech speed (0.5-2.0, default 1.0): ").strip()
            if not speed:
                return 1.0
            speed = float(speed)
            if 0.5 <= speed <= 2.0:
                return speed
            print("Speed must be between 0.5 and 2.0")
        except ValueError:
            print("Please enter a valid number.")

def save_audio_with_retry(audio_data: np.ndarray, sample_rate: int, output_path: Path, max_retries: int = 3, retry_delay: float = 1.0) -> bool:
    """
    Attempt to save audio data to file with retry logic.
    Returns True if successful, False otherwise.
    """
    # Make sure output_path is an absolute path
    output_path = Path(os.path.abspath(output_path))
    
    for attempt in range(max_retries):
        try:
            sf.write(str(output_path), audio_data, sample_rate)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"\nFailed to save audio (attempt {attempt + 1}/{max_retries}): {e}")
                print("The output file might be in use by another program (e.g., media player).")
                print(f"Please close any programs that might be using '{output_path}'")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"\nError: Could not save audio after {max_retries} attempts: {e}")
                print(f"Please ensure '{output_path}' is not open in any other program and try again.")
                return False
    return False

def main() -> None:
    try:
        # Set up device safely
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except (RuntimeError, AttributeError, ImportError) as e:
            print(f"CUDA initialization error: {e}. Using CPU instead.")
            device = 'cpu'  # Fallback if CUDA check fails
        print(f"Using device: {device}")
        
        # Build model
        print("\nInitializing model...")
        with tqdm(total=1, desc="Building model") as pbar:
            model = build_model(DEFAULT_MODEL_PATH, device)
            pbar.update(1)
        
        # Cache for voices to avoid redundant calls
        voices_cache = None
        
        while True:
            choice = print_menu()
            
            if choice == "1":
                # List voices
                voices_cache = list_available_voices()
                print("\nAvailable voices:")
                for voice in voices_cache:
                    print(f"- {voice}")
                    
            elif choice == "2":
                # Generate speech
                # Use cached voices if available
                if voices_cache is None:
                    voices_cache = list_available_voices()
                
                if not voices_cache:
                    print("No voices found! Please check the voices directory.")
                    continue
                
                # Get user inputs
                voice = select_voice(voices_cache)
                text = get_text_input()
                
                # Validate text (don't allow extremely long inputs)
                if len(text) > 10000:  # Reasonable limit for text length
                    print("Text is too long. Please enter a shorter text.")
                    continue
                    
                speed = get_speed()
                
                print(f"\nGenerating speech for: '{text}'")
                print(f"Using voice: {voice}")
                print(f"Speed: {speed}x")
                
                # Generate speech
                all_audio = []
                voice_path = os.path.abspath(os.path.join("voices", f"{voice}.pt"))
                
                # Verify voice file exists
                if not os.path.exists(voice_path):
                    print(f"Error: Voice file not found: {voice_path}")
                    continue
                
                # Set a timeout for generation with per-segment timeout
                max_gen_time = 300  # 5 minutes max total
                max_segment_time = 60  # 60 seconds max per segment
                start_time = time.time()
                segment_start_time = start_time
                
                try:
                    # Setup watchdog timer for overall process
                    import threading
                    generation_complete = False
                    
                    def watchdog_timer():
                        if not generation_complete:
                            print("\nWatchdog: Generation taking too long, process will be cancelled")
                            # Can't directly interrupt generator, but this will inform user
                    
                    # Start watchdog timer
                    watchdog = threading.Timer(max_gen_time, watchdog_timer)
                    watchdog.daemon = True  # Don't prevent program exit
                    watchdog.start()
                    
                    # Initialize generator
                    try:
                        generator = model(text, voice=voice_path, speed=speed, split_pattern=r'\n+')
                    except (ValueError, TypeError, RuntimeError) as e:
                        print(f"Error initializing speech generator: {e}")
                        watchdog.cancel()
                        continue
                    except Exception as e:
                        print(f"Unexpected error initializing generator: {type(e).__name__}: {e}")
                        watchdog.cancel()
                        continue
                    
                    # Process segments
                    with tqdm(desc="Generating speech") as pbar:
                        for gs, ps, audio in generator:
                            # Check overall timeout
                            current_time = time.time()
                            if current_time - start_time > max_gen_time:
                                print("\nWarning: Total generation time exceeded limit, stopping")
                                break
                            
                            # Check per-segment timeout
                            segment_elapsed = current_time - segment_start_time
                            if segment_elapsed > max_segment_time:
                                print(f"\nWarning: Segment took too long ({segment_elapsed:.1f}s), stopping")
                                break
                            
                            # Reset segment timer
                            segment_start_time = current_time
                            
                            # Process audio if available
                            if audio is not None:
                                # Only convert if it's a numpy array, not if already tensor
                                audio_tensor = audio if isinstance(audio, torch.Tensor) else torch.from_numpy(audio).float()
                                
                                all_audio.append(audio_tensor)
                                print(f"\nGenerated segment: {gs}")
                                if ps:  # Only print phonemes if available
                                    print(f"Phonemes: {ps}")
                                pbar.update(1)
                    
                    # Mark generation as complete (for watchdog)
                    generation_complete = True
                    watchdog.cancel()
                    
                except ValueError as e:
                    print(f"Value error during speech generation: {e}")
                except RuntimeError as e:
                    print(f"Runtime error during speech generation: {e}")
                    # If CUDA out of memory, provide more helpful message
                    if "CUDA out of memory" in str(e):
                        print("CUDA out of memory error - try using a shorter text or switching to CPU")
                except KeyError as e:
                    print(f"Key error during speech generation: {e}")
                    print("This might be caused by a missing voice configuration")
                except FileNotFoundError as e:
                    print(f"File not found: {e}")
                except Exception as e:
                    print(f"Unexpected error during speech generation: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Save audio
                if all_audio:
                    try:
                        # Handle single segment case without concatenation
                        if len(all_audio) == 1:
                            final_audio = all_audio[0]
                        else:
                            try:
                                final_audio = torch.cat(all_audio, dim=0)
                            except RuntimeError as e:
                                print(f"Error concatenating audio segments: {e}")
                                continue
                        
                        output_path = Path(DEFAULT_OUTPUT_FILE)
                        if save_audio_with_retry(final_audio.numpy(), SAMPLE_RATE, output_path):
                            print(f"\nAudio saved to {output_path.absolute()}")
                            # Play a system beep to indicate completion
                            try:
                                print('\a')  # ASCII bell - should make a sound on most systems
                            except:
                                pass
                        else:
                            print("Failed to save audio file")
                    except Exception as e:
                        print(f"Error processing audio: {type(e).__name__}: {e}")
                else:
                    print("Error: Failed to generate audio")
                    
            elif choice == "3":
                print("\nGoodbye!")
                break
                
            else:
                print("\nInvalid choice. Please try again.")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Comprehensive cleanup with error handling
        try:
            print("\nPerforming cleanup...")
            
            # Ensure model is properly released
            if 'model' in locals() and model is not None:
                print("Cleaning up model resources...")
                # First clear any references to voice models
                if hasattr(model, 'voices'):
                    try:
                        voices_count = len(model.voices)
                        model.voices.clear()
                        print(f"Cleared {voices_count} voice references")
                    except Exception as voice_error:
                        print(f"Error clearing voice references: {voice_error}")
                
                # Clear any other model attributes that might hold references
                try:
                    for attr in list(model.__dict__.keys()):
                        if hasattr(model, attr) and not attr.startswith('__'):
                            try:
                                delattr(model, attr)
                            except:
                                pass
                except Exception as attr_error:
                    print(f"Error clearing model attributes: {attr_error}")
                    
                # Then delete the model
                try:
                    del model
                    model = None
                    print("Model reference deleted")
                except Exception as del_error:
                    print(f"Error deleting model: {del_error}")
            
            # Clean up voice cache
            if 'voices_cache' in locals() and voices_cache is not None:
                try:
                    voices_cache.clear()
                    voices_cache = None
                    print("Voice cache cleared")
                except Exception as cache_error:
                    print(f"Error clearing voice cache: {cache_error}")
            
            # Clean up any CUDA resources
            if torch.cuda.is_available():
                try:
                    print("Cleaning up CUDA resources...")
                    torch.cuda.empty_cache()
                    print("CUDA cache emptied")
                except Exception as cuda_error:
                    print(f"Error clearing CUDA cache: {cuda_error}")
            
            # Make sure patched functions are restored
            try:
                from models import _cleanup_monkey_patches
                _cleanup_monkey_patches()
                print("Monkey patches restored")
            except Exception as patch_error:
                print(f"Error restoring monkey patches: {patch_error}")
            
            # Final garbage collection
            try:
                import gc
                gc.collect()
                print("Garbage collection completed")
            except Exception as gc_error:
                print(f"Error during garbage collection: {gc_error}")
                
            print("Cleanup completed")
            
        except Exception as e:
            print(f"Error during cleanup: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 