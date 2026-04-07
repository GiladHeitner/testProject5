import argparse
import os
import shlex
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def run(command: str) -> None:
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {command}\n{result.stderr.strip()}"
        )


def build_atempo_chain(factor: float) -> str:
    if factor <= 0:
        raise ValueError("Tempo factor must be positive.")
    parts = []
    remaining = factor
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    parts.append(f"atempo={remaining:.5f}")
    return ",".join(parts)


def stylize_narration_audio(
    in_audio_path: Path, out_audio_path: Path, speech_speed: float, pitch_factor: float
) -> None:
    if speech_speed <= 0:
        raise ValueError("speech_speed must be > 0.")
    if pitch_factor <= 0:
        raise ValueError("pitch_factor must be > 0.")

    speed_chain = (
        f"aresample=48000,"
        f"rubberband=tempo={speech_speed:.5f}:pitch={pitch_factor:.5f}:formant=preserved,"
        f"aresample=48000"
    )
    master_chain = ",".join(
    [
        "highpass=f=80",
        "lowpass=f=15500",

        # Adam-ish: subtle, not aggressive
        "equalizer=f=140:t=q:w=1:g=1.5",
        "equalizer=f=300:t=q:w=1:g=-1.8",
        "equalizer=f=3200:t=q:w=1:g=1.6",
        "equalizer=f=5200:t=q:w=1:g=1.0",
        "equalizer=f=8200:t=q:w=1:g=-1.2",

        # tiny rasp (less than before)
        "asoftclip=type=tanh:threshold=0.32",

        # compression: gentler + more natural
        "acompressor=threshold=-22dB:ratio=2.6:attack=8:release=160:makeup=4",
        "alimiter=limit=0.95",

        # normalize but less “pumpy”
        "loudnorm=I=-16:TP=-1.5:LRA=9",
        ]
    )
    filter_chain = f"{speed_chain},{master_chain}"
    cmd = (
        f"ffmpeg -y -i {shlex.quote(str(in_audio_path))} "
        f"-filter:a {shlex.quote(filter_chain)} "
        f"-c:a libmp3lame -q:a 2 {shlex.quote(str(out_audio_path))}"
    )
    try:
        run(cmd)
    except RuntimeError:
        # Fallback when rubberband filter is unavailable.
        tempo_after_pitch = speech_speed / pitch_factor
        fallback_speed_chain = (
            f"aresample=44100,"
            f"asetrate=44100*{pitch_factor:.5f},"
            f"aresample=44100,"
            f"{build_atempo_chain(tempo_after_pitch)}"
        )
        fallback_filter = f"{fallback_speed_chain},{master_chain}"
        fallback_cmd = (
            f"ffmpeg -y -i {shlex.quote(str(in_audio_path))} "
            f"-filter:a {shlex.quote(fallback_filter)} "
            f"-c:a libmp3lame -q:a 2 {shlex.quote(str(out_audio_path))}"
        )
        run(fallback_cmd)


def normalize_audio_duration(
    in_audio_path: Path, out_audio_path: Path, target_seconds: float
) -> None:
    if target_seconds <= 0:
        out_audio_path.write_bytes(in_audio_path.read_bytes())
        return
    probe_cmd = (
        f"ffprobe -v error -show_entries format=duration "
        f"-of default=noprint_wrappers=1:nokey=1 {shlex.quote(str(in_audio_path))}"
    )
    result = subprocess.run(probe_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        out_audio_path.write_bytes(in_audio_path.read_bytes())
        return
    try:
        current = float(result.stdout.strip())
    except ValueError:
        out_audio_path.write_bytes(in_audio_path.read_bytes())
        return
    if current <= 0:
        out_audio_path.write_bytes(in_audio_path.read_bytes())
        return

    factor = current / target_seconds
    filter_chain = build_atempo_chain(factor)
    cmd = (
        f"ffmpeg -y -i {shlex.quote(str(in_audio_path))} "
        f"-filter:a {shlex.quote(filter_chain)} "
        f"-c:a libmp3lame -q:a 2 {shlex.quote(str(out_audio_path))}"
    )
    run(cmd)


def main() -> None:
    root_env = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=root_env)
    parser = argparse.ArgumentParser(description="Generate OpenAI TTS voice test samples.")
    parser.add_argument(
        "--text",
        default=(
            "Back in middle school, she thought nobody would notice the secret notes. "
            "Then one teacher looked RIGHT at her notebook and everything almost fell apart."
        ),
    )
    parser.add_argument(
        "--voices",
        default="nova",
        help="Comma-separated OpenAI voices to test.",
    )
    parser.add_argument("--speech-speed", type=float, default=1.35)
    parser.add_argument("--pitch-factor", type=float, default=1.00)
    parser.add_argument("--target-seconds", type=float, default=0.0)
    parser.add_argument("--out-dir", default="output/voice_tests")
    parser.add_argument(
        "--instructions",
       default=(
    "Ultra fast YouTube Shorts narration. "
    "Speak extremely quickly with nonstop flow. "
    "Do not pause for commas periods or punctuation. "
    "No pauses between sentences. "
    "Words should flow together rapidly like a fast storyteller. "
    "Maintain constant high speed from start to finish. "
    "Voice is confident super raspy adult male narrator. "
    "Zero dramatic pauses zero gaps minimal breaths."
),
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Set it in project root .env.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=api_key)

    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    if not voices:
        raise RuntimeError("No voices provided.")

    speed_label = f"{args.speech_speed:.3f}".replace(".", "p")
    pitch_label = f"{args.pitch_factor:.3f}".replace(".", "p")
    print(
        f"Using speech_speed={args.speech_speed:.3f}, "
        f"pitch_factor={args.pitch_factor:.3f}."
    )

    for voice in voices:
        raw_path = out_dir / f"{voice}_raw_{speed_label}_{pitch_label}.mp3"
        styled_path = out_dir / f"{voice}_styled_{speed_label}_{pitch_label}.mp3"
        final_path = (
            out_dir
            / f"{voice}_final_{int(args.target_seconds)}s_{speed_label}_{pitch_label}.mp3"
        )

        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=args.text,
            response_format="mp3",
            instructions=args.instructions,
        )
        speech.stream_to_file(str(raw_path))
        stylize_narration_audio(
            in_audio_path=raw_path,
            out_audio_path=styled_path,
            speech_speed=args.speech_speed,
            pitch_factor=args.pitch_factor,
        )
        normalize_audio_duration(
            in_audio_path=styled_path,
            out_audio_path=final_path,
            target_seconds=args.target_seconds,
        )
        print(f"Created: {final_path}")


if __name__ == "__main__":
    main()
