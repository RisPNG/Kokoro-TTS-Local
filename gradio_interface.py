"""
Kokoro-TTS Local Generator
-------------------------
A Gradio interface for the Kokoro-TTS system with automatic detection for Japanese voices.
"""

import gradio as gr
import os
import sys
import platform
from datetime import datetime
import shutil
from pathlib import Path
import soundfile as sf
from pydub import AudioSegment
import torch
import numpy as np

# Import your custom modules
from models import list_available_voices, build_model, generate_speech

# For Japanese pipeline
from kokoro import KPipeline

# Global configuration
CONFIG_FILE = "tts_config.json"
DEFAULT_OUTPUT_DIR = "outputs"
SAMPLE_RATE = 24000

# Initialize model globally
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None

# Create a global pipeline for Japanese voices
jp_pipeline = None

# Example set of known Japanese voices
JAPANESE_VOICES = ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"]
# You can expand or modify based on your use case.


def get_available_voices():
    """Get list of available voice models."""
    global model
    try:
        # Initialize model once if needed to trigger voice downloads
        if model is None:
            print("Initializing main (non-Japanese) model...")
            model = build_model(None, device)

        voices = list_available_voices()
        if not voices:
            print("No voices found after initialization. Attempting to download...")
            download_voice_files()  # Assuming you have a function that downloads voice files
            voices = list_available_voices()

        print("Available voices:", voices)
        return voices
    except Exception as e:
        print(f"Error getting voices: {e}")
        return []


def convert_audio(input_path: str, output_path: str, fmt: str):
    """Convert audio to specified format."""
    try:
        if fmt == "wav":
            return input_path
        audio = AudioSegment.from_wav(input_path)
        if fmt == "mp3":
            audio.export(output_path, format="mp3", bitrate="192k")
        elif fmt == "aac":
            audio.export(output_path, format="aac", bitrate="192k")
        return output_path
    except Exception as e:
        print(f"Error converting audio: {e}")
        return input_path


def generate_tts_with_logs(voice_name, text, fmt):
    """Generate TTS audio with progress logging and auto-detection for Japanese voices."""
    global model
    global jp_pipeline

    try:
        # Prepare directories
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

        # Build the Japanese pipeline if it's a Japanese voice and pipeline hasn't been built yet
        if voice_name in JAPANESE_VOICES:
            if jp_pipeline is None:
                print("Initializing Japanese pipeline (KPipeline, lang_code='j')...")
                jp_pipeline = KPipeline(lang_code="j")
        else:
            # Fallback to English / standard pipeline if not yet built
            if model is None:
                print("Initializing main pipeline model...")
                model = build_model(None, device)

        # Generate a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"tts_{timestamp}"
        wav_path = os.path.join(DEFAULT_OUTPUT_DIR, f"{base_name}.wav")

        # --- MAIN LOGIC: Choose which pipeline to use based on voice_name ---
        if voice_name in JAPANESE_VOICES:
            # Use Japanese pipeline
            print(
                f"\n[JP] Generating Japanese speech for '{text}' with voice={voice_name}"
            )
            # The pipeline returns a generator. Typically you’d do:
            generator = jp_pipeline(
                text,
                voice=f"voices/{voice_name}.pt",  # Adjust if you store .pt files differently
                speed=1.0,
                split_pattern=r"\n+",
            )

            all_audio = []
            # If your JP pipeline yields (gs, ps, audio), iterate:
            for gs, ps, audio in generator:
                if audio is not None:
                    if isinstance(audio, np.ndarray):
                        audio = torch.from_numpy(audio).float()
                    all_audio.append(audio)
                    print(f"Generated segment (JP): {gs}")
                    print(f"Phonemes (JP): {ps}")

        else:
            # Use the standard pipeline
            print(
                f"\n[EN/Other] Generating speech for '{text}' with voice={voice_name}"
            )
            generator = model(
                text,
                voice=f"voices/{voice_name}.pt",  # Adjust to match your file structure
                speed=1.0,
                split_pattern=r"\n+",
            )

            all_audio = []
            for gs, ps, audio in generator:
                if audio is not None:
                    if isinstance(audio, np.ndarray):
                        audio = torch.from_numpy(audio).float()
                    all_audio.append(audio)
                    print(f"Generated segment: {gs}")
                    print(f"Phonemes: {ps}")

        if not all_audio:
            raise Exception("No audio segments were generated.")

        # Combine all audio segments
        final_audio = torch.cat(all_audio, dim=0)
        sf.write(wav_path, final_audio.numpy(), SAMPLE_RATE)

        # Convert to requested format if needed
        if fmt != "wav":
            output_path = os.path.join(DEFAULT_OUTPUT_DIR, f"{base_name}.{fmt}")
            return convert_audio(wav_path, output_path, fmt)

        return wav_path

    except Exception as e:
        print(f"Error generating speech: {e}")
        import traceback

        traceback.print_exc()
        return None


def create_interface(server_name="0.0.0.0", server_port=7860):
    """Create and launch the Gradio interface."""
    voices = get_available_voices()
    if not voices:
        print("No voices found! Please check the voices directory.")
        return

    with gr.Blocks(title="Kokoro TTS Generator") as interface:
        gr.Markdown("# Kokoro TTS Generator")

        with gr.Row():
            with gr.Column():
                voice = gr.Dropdown(
                    choices=voices, value=voices[0] if voices else None, label="Voice"
                )
                text = gr.Textbox(
                    lines=3,
                    placeholder="Enter text to convert to speech...",
                    label="Text",
                )
                fmt = gr.Radio(
                    choices=["wav", "mp3", "aac"], value="wav", label="Output Format"
                )
                generate = gr.Button("Generate Speech")

            with gr.Column():
                output = gr.Audio(label="Generated Audio")

        generate.click(
            fn=generate_tts_with_logs, inputs=[voice, text, fmt], outputs=output
        )

    interface.launch(server_name=server_name, server_port=server_port, share=True)


if __name__ == "__main__":
    create_interface()
