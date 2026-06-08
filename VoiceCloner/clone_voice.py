import argparse
from pathlib import Path
import subprocess
import sys
from itertools import product

import numpy as np
import soundfile as sf
import torch

# Allow imports when run from VoiceCloner/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorts_bot_lib.text import strip_paralinguistic_tags


def _log(msg: str) -> None:
    print(msg, flush=True)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS: custom preset voice, voice design, or clone."
    )
    parser.add_argument(
        "--mode",
        choices=("custom", "design", "clone"),
        default="custom",
        help="custom=preset speaker+instruct; design=voice from description; clone=ref audio",
    )
    parser.add_argument("--ref-audio", default="", help="Reference clip (clone mode only).")
    parser.add_argument(
        "--ref-text",
        default="",
        help="Transcript of reference audio (clone mode).",
    )
    parser.add_argument("--text", required=True, help="Target text to synthesize.")
    parser.add_argument(
        "--language",
        default="English",
        help='Language (e.g. "English", "Auto").',
    )
    parser.add_argument(
        "--speaker",
        default="Ryan",
        help="CustomVoice preset: Ryan, Aiden, Dylan, Eric, …",
    )
    parser.add_argument(
        "--instruct",
        default="",
        help="Style instruction (custom/design modes).",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        help="Model id or local snapshot directory.",
    )
    parser.add_argument("--output", default="cloned_output.wav")
    parser.add_argument(
        "--device",
        default=default_device(),
        help="cuda:0, mps (Apple Silicon), or cpu",
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--speed-values", default="")
    parser.add_argument("--pitch-values", default="")
    parser.add_argument(
        "--list-speakers",
        action="store_true",
        help="Print supported CustomVoice speakers and exit.",
    )
    return parser.parse_args()


def parse_csv_floats(value: str) -> list[float]:
    if not value.strip():
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def suffix_from_values(speed: float, pitch: float) -> str:
    speed_tag = f"{speed:.2f}".replace("-", "m")
    pitch_tag = f"{pitch:.2f}".replace("-", "m")
    return f"_speed_{speed_tag}_pitch_{pitch_tag}"


def apply_speed_pitch(input_wav: Path, output_wav: Path, speed: float, pitch: float) -> None:
    if speed <= 0:
        raise ValueError("Speed must be > 0.")

    tempo_filters = []
    remaining = speed
    while remaining > 2.0:
        tempo_filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        tempo_filters.append("atempo=0.5")
        remaining /= 0.5
    tempo_filters.append(f"atempo={remaining:.4f}")

    if pitch != 0.0:
        tmp_pitched = input_wav.with_name(input_wav.stem + "_pitched_tmp.wav")
        pitch_cents = int(round(pitch * 100))
        subprocess.run(
            ["sox", str(input_wav), str(tmp_pitched), "pitch", str(pitch_cents)],
            check=True,
        )
        speed_input = tmp_pitched
    else:
        speed_input = input_wav
        tmp_pitched = None

    af = ",".join(tempo_filters)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(speed_input), "-af", af, str(output_wav)],
        check=True,
        capture_output=True,
    )

    if tmp_pitched and tmp_pitched.exists():
        tmp_pitched.unlink()


def synthesize(model, args: argparse.Namespace, *, text: str, instruct: str):
    mode = args.mode.strip().lower()
    if mode == "clone":
        ref_audio_path = Path(args.ref_audio)
        if not ref_audio_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")
        if not args.ref_text.strip():
            raise ValueError("clone mode requires --ref-text")
        return model.generate_voice_clone(
            text=text,
            language=args.language,
            ref_audio=str(ref_audio_path),
            ref_text=args.ref_text,
        )
    if mode == "design":
        if not instruct.strip():
            raise ValueError("design mode requires --instruct")
        return model.generate_voice_design(
            text=text,
            language=args.language,
            instruct=instruct,
        )
    kwargs: dict = {
        "text": text,
        "language": args.language,
        "speaker": args.speaker,
    }
    if instruct.strip():
        kwargs["instruct"] = instruct
    return model.generate_custom_voice(**kwargs)


def synthesize_script(model, args: argparse.Namespace):
    """Single-pass synthesis — avoids segment gaps that cause unnatural pauses."""
    spoken = strip_paralinguistic_tags(args.text)
    spoken = " ".join(spoken.split())
    if not spoken.strip():
        raise ValueError("No speakable text after removing delivery tags.")
    return synthesize(model, args, text=spoken, instruct=args.instruct)


def main() -> None:
    args = parse_args()
    use_cuda = args.device.startswith("cuda")
    use_mps = args.device == "mps"
    # bfloat16 only on CUDA; MPS and CPU use float32
    dtype = torch.bfloat16 if use_cuda else torch.float32

    _log(f"Loading voice model on {args.device} (first run can take 1-2 min)...")
    from qwen_tts import Qwen3TTSModel

    if use_mps or use_cuda:
        # device_map doesn't work with MPS; load on CPU then move
        model = Qwen3TTSModel.from_pretrained(args.model, dtype=dtype)
        try:
            model.model = model.model.to(args.device)
            model.device = torch.device(args.device)
            _log(f"Model moved to {args.device}.")
        except Exception as e:
            _log(f"Could not move to {args.device}: {e}. Staying on CPU.")
    else:
        model = Qwen3TTSModel.from_pretrained(
            args.model,
            device_map=args.device,
            dtype=dtype,
        )
    _log("Model loaded.")

    if args.list_speakers:
        speakers = model.get_supported_speakers()
        _log("Supported speakers: " + ", ".join(speakers))
        return

    _log("Generating speech...")
    wavs, sr = synthesize_script(model, args)
    output_path = Path(args.output)
    _log(f"Saving audio to {output_path}...")
    sf.write(output_path, wavs[0], sr)

    speed_values = parse_csv_floats(args.speed_values)
    pitch_values = parse_csv_floats(args.pitch_values)

    if not speed_values and not pitch_values:
        if args.speed == 1.0 and args.pitch == 0.0:
            _log(f"Saved TTS audio to: {output_path}")
            return
        transformed_path = output_path.with_name(output_path.stem + "_sp.wav")
        apply_speed_pitch(output_path, transformed_path, args.speed, args.pitch)
        _log(f"Saved TTS audio to: {output_path}")
        _log(f"Saved speed/pitch version to: {transformed_path}")
        return

    if not speed_values:
        speed_values = [1.0]
    if not pitch_values:
        pitch_values = [0.0]

    _log(f"Saved base TTS audio to: {output_path}")
    for speed, pitch in product(speed_values, pitch_values):
        variant_path = output_path.with_name(
            output_path.stem + suffix_from_values(speed, pitch) + output_path.suffix
        )
        apply_speed_pitch(output_path, variant_path, speed, pitch)
        _log(f"Saved variant: {variant_path}")


if __name__ == "__main__":
    main()
