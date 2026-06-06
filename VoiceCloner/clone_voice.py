import argparse
from pathlib import Path
import subprocess
from itertools import product
import io
from contextlib import redirect_stdout

import soundfile as sf
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clone a voice with Qwen3-TTS Base model.")
    parser.add_argument("--ref-audio", required=True, help="Path to reference voice audio snippet.")
    parser.add_argument(
        "--ref-text",
        required=True,
        help="Exact transcript of the reference audio. Important for quality.",
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Target text to synthesize in cloned voice.",
    )
    parser.add_argument(
        "--language",
        default="English",
        help='Language of target text (e.g., "English", "Chinese", or "Auto").',
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Model id or local model path.",
    )
    parser.add_argument(
        "--output",
        default="cloned_output.wav",
        help="Output wav file path.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help='Torch device, e.g. "cuda:0" or "cpu".',
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Single output speed factor (e.g. 0.9 slower, 1.1 faster).",
    )
    parser.add_argument(
        "--pitch",
        type=float,
        default=0.0,
        help="Single output pitch shift in semitones (e.g. -2, +3).",
    )
    parser.add_argument(
        "--speed-values",
        default="",
        help='Comma-separated speed factors for batch tests, e.g. "0.9,1.0,1.1".',
    )
    parser.add_argument(
        "--pitch-values",
        default="",
        help='Comma-separated pitch semitones for batch tests, e.g. "-2,0,2".',
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
    
    # Build atempo chain — ffmpeg atempo is limited to 0.5–2.0 per filter, chain for >2.0
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
        # Use sox for pitch shift only (no tempo change)
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


def main() -> None:
    args = parse_args()
    ref_audio_path = Path(args.ref_audio)
    if not ref_audio_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

    use_gpu = args.device.startswith("cuda")
    dtype = torch.bfloat16 if use_gpu else torch.float32

    # qwen_tts prints a flash-attn warning on import; suppress noisy import output on non-CUDA setups.
    with redirect_stdout(io.StringIO()):
        from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=dtype,
    )

    wavs, sr = model.generate_voice_clone(
        text=args.text,
        language=args.language,
        ref_audio=str(ref_audio_path),
        ref_text=args.ref_text,
    )
    output_path = Path(args.output)
    sf.write(output_path, wavs[0], sr)

    speed_values = parse_csv_floats(args.speed_values)
    pitch_values = parse_csv_floats(args.pitch_values)

    if not speed_values and not pitch_values:
        if args.speed == 1.0 and args.pitch == 0.0:
            print(f"Saved cloned audio to: {output_path}")
            return
        transformed_path = output_path.with_name(output_path.stem + "_sp.wav")
        apply_speed_pitch(output_path, transformed_path, args.speed, args.pitch)
        print(f"Saved cloned audio to: {output_path}")
        print(f"Saved speed/pitch version to: {transformed_path}")
        return

    if not speed_values:
        speed_values = [1.0]
    if not pitch_values:
        pitch_values = [0.0]

    print(f"Saved base cloned audio to: {output_path}")
    for speed, pitch in product(speed_values, pitch_values):
        variant_path = output_path.with_name(
            output_path.stem + suffix_from_values(speed, pitch) + output_path.suffix
        )
        apply_speed_pitch(output_path, variant_path, speed, pitch)
        print(f"Saved variant: {variant_path}")


if __name__ == "__main__":
    main()
