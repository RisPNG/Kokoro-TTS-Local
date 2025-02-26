import os
import sys
import platform
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import gradio as gr

#############################################
# 1) IMPORT THE KPipeline + MODEL
#############################################

# This assumes you have KPipeline (and KModel) in a local module called kokoro
# containing exactly the code from your KPipeLine snippet in your question.
# Adjust your import to match your actual file structure.

# e.g. from kokoro.pipeline import KPipeline, KModel
# or simply:
from kokoro import KPipeline
from kokoro.model import KModel


#############################################
# 2) MAP VOICES → LANG CODE, CREATE PIPELINE DICT
#############################################

# According to VOICES.md, we can detect the language by the prefix:
LANG_MAP = {
    # American English
    "af_": "a",
    "am_": "a",

    # British English
    "bf_": "b",
    "bm_": "b",

    # Japanese
    "jf_": "j",
    "jm_": "j",

    # Mandarin Chinese
    "zf_": "z",
    "zm_": "z",

    # Spanish
    "ef_": "e",
    "em_": "e",

    # French
    "ff_": "f",

    # Hindi
    "hf_": "h",
    "hm_": "h",

    # Italian
    "if_": "i",
    "im_": "i",

    # Brazilian Portuguese
    "pf_": "p",
    "pm_": "p",
}

# Keep a global dictionary of {lang_code -> pipeline}
pipelines = {}

# If you want a single KModel for *all* languages (to share GPU memory)
global_model = None

DEFAULT_OUTPUT_DIR = "outputs"
SAMPLE_RATE = 24000


#############################################
# 3) PLACEHOLDERS FOR ANY MISSING LOGIC
#############################################

def list_available_voices():
    """
    Placeholder. Return a list of all voice names (like 'af_heart', 'bf_alice', etc.)
    that you have in your local environment or HF hub.
    """
    # You can do a local directory listing, or a curated list:
    return [
        "af_heart", "af_alloy", "af_aoede", "af_bella", "am_adam", 
        "bf_alice", "bf_emma", "bm_daniel", "bm_george",
        "jf_alpha", "jm_kumo", "zf_xiaobei", "zm_yunjian",
        "ef_dora", "em_alex", "ff_siwis", "hf_alpha", "hm_omega",
        "if_sara", "im_nicola", "pf_dora", "pm_santa",
    ]


def build_model(config, device):
    """
    Placeholder. Suppose you want to load a single large KModel:
    """
    print("[build_model] Loading KModel onto device:", device)
    model = KModel().to(device).eval()
    return model


#############################################
# 4) HELPER FUNCTIONS
#############################################

def get_pipeline_for_voice(voice_name: str) -> KPipeline:
    """
    From the voice name's prefix (e.g. 'af_', 'jf_'), figure out
    the correct language code and return or create the KPipeline.
    """
    # Try the first 3 chars, fallback to American English if unknown
    prefix = voice_name[:3].lower()
    lang_code = LANG_MAP.get(prefix, "a")

    if lang_code not in pipelines:
        print(f"[INFO] Creating pipeline for lang_code='{lang_code}'")
        # If we want one shared KModel for everything:
        if global_model is not None:
            pipelines[lang_code] = KPipeline(lang_code=lang_code, model=global_model)
        else:
            # This pipeline will create its own KModel
            pipelines[lang_code] = KPipeline(lang_code=lang_code, model=True)
    return pipelines[lang_code]


def convert_audio(input_path: str, output_path: str, fmt: str) -> str:
    """
    Convert .wav file to mp3/aac if needed. If 'fmt' is 'wav', do nothing.
    """
    if fmt == "wav":
        return input_path

    audio = AudioSegment.from_wav(input_path)
    if fmt == "mp3":
        audio.export(output_path, format="mp3", bitrate="192k")
    elif fmt == "aac":
        audio.export(output_path, format="aac", bitrate="192k")
    else:
        print(f"[convert_audio] Unrecognized format: {fmt}, leaving WAV as-is.")
        return input_path

    return output_path


#############################################
# 5) MAIN TTS GENERATION LOGIC
#############################################

def generate_tts_with_logs(voice_name: str, text: str, fmt: str) -> str:
    """
    Generate speech for the given text & voice, auto-selecting the right pipeline.
    Return the path to the resulting audio file (either .wav, .mp3, or .aac).
    """
    try:
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

        # 1) Choose the pipeline
        pipeline = get_pipeline_for_voice(voice_name)

        # 2) Timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"tts_{voice_name}_{timestamp}"
        wav_path = os.path.join(DEFAULT_OUTPUT_DIR, f"{base_name}.wav")

        print(f"\n[INFO] Generating TTS with voice='{voice_name}', text='{text}'")

        # 3) Call pipeline → generator of KPipeline.Result
        results_gen = pipeline(
            text,
            voice=voice_name,   # e.g. "af_heart"
            speed=1.0,
            split_pattern=r"\n+",
        )

        all_audio = []
        for result in results_gen:
            # result is an instance of KPipeline.Result
            audio_tensor = result.audio
            if audio_tensor is not None:
                # Move to CPU, ensure float
                if not isinstance(audio_tensor, torch.Tensor):
                    audio_tensor = torch.tensor(audio_tensor, dtype=torch.float32)
                audio_tensor = audio_tensor.float().cpu()
                all_audio.append(audio_tensor)

                # Debug logs
                print(f"[Segment] Graphemes: {result.graphemes}")
                print(f"[Segment] Phonemes:  {result.phonemes}")

        if not all_audio:
            raise RuntimeError("No audio segments generated.")

        # 4) Concatenate all audio
        final_audio = torch.cat(all_audio, dim=0)

        # 5) Write to .wav
        sf.write(wav_path, final_audio.numpy(), SAMPLE_RATE)
        print(f"[INFO] Wrote WAV: {wav_path}")

        # 6) Convert if needed
        if fmt != "wav":
            out_file = os.path.join(DEFAULT_OUTPUT_DIR, f"{base_name}.{fmt}")
            out_file = convert_audio(wav_path, out_file, fmt)
            print(f"[INFO] Converted to: {out_file}")
            return out_file

        return wav_path

    except Exception as e:
        print(f"[ERROR] generate_tts_with_logs: {e}")
        import traceback
        traceback.print_exc()
        return ""


#############################################
# 6) GRADIO INTERFACE
#############################################

def create_interface(server_name="0.0.0.0", server_port=7860):
    """
    Build a Gradio UI for Kokoro TTS.
    """
    # Optionally pre-initialize a single global model if you like
    global global_model
    if global_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        global_model = build_model(None, device)

    # List your voices (from local or your metadata)
    voices = list_available_voices()

    with gr.Blocks(title="Kokoro TTS Generator - Multi-Lingual") as interface:
        gr.Markdown("# Kokoro TTS Generator (Multi-Language)")

        with gr.Row():
            with gr.Column():
                voice_box = gr.Dropdown(
                    choices=voices, 
                    value=voices[0] if voices else None, 
                    label="Voice"
                )
                text_box = gr.Textbox(
                    lines=3,
                    placeholder="Enter text to convert to speech...",
                    label="Text",
                )
                fmt_radio = gr.Radio(
                    choices=["wav", "mp3", "aac"],
                    value="wav",
                    label="Output Format"
                )
                generate_button = gr.Button("Generate Speech")

            with gr.Column():
                output_audio = gr.Audio(label="Generated Audio")

        # Wiring: generate_button calls generate_tts_with_logs
        # We'll feed the resulting file path back to the Audio component
        generate_button.click(
            fn=generate_tts_with_logs,
            inputs=[voice_box, text_box, fmt_radio],
            outputs=output_audio
        )

    interface.launch(server_name=server_name, server_port=server_port, share=True)


#############################################
# 7) MAIN ENTRY POINT
#############################################

if __name__ == "__main__":
    create_interface(server_name="0.0.0.0", server_port=7860)
