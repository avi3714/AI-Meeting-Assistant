from pathlib import Path
import subprocess
import sys


def preprocess_audio(input_path: Path) -> Path:
    """Convert audio to mono, 16 kHz, 16-bit PCM WAV."""

    if not input_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    output_dir = Path("processed_audio")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{input_path.stem}_processed.wav"

    command = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(output_path),
    ]

    print(f"Input audio: {input_path}")
    print("Converting audio...")

    subprocess.run(command, check=True)

    print("\nConversion successful!")
    print(f"Output audio: {output_path.resolve()}")

    return output_path


def main():
    if len(sys.argv) != 2:
        print(
            'Usage: python scripts/preprocess_audio.py '
            '"audio/meeting.wav"'
        )
        sys.exit(1)

    input_path = Path(sys.argv[1])

    try:
        preprocess_audio(input_path)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
