
from pathlib import Path
import json
import sys
import torch
import gc
import soundfile as sf
import numpy as np

from pyannote.audio import Pipeline


MODEL_NAME = "pyannote/speaker-diarization-community-1"

CHUNK_DURATION = 30
CHUNK_OVERLAP = 5
SAMPLE_RATE = 16000

# CPU is the stable baseline for your 4 GB RTX 2050.
DEVICE = "cpu"


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    temp_path.replace(path)


def load_audio(audio_path):
    print("\nLoading audio...")

    audio, sample_rate = sf.read(
        str(audio_path),
        dtype="float32"
    )

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Expected {SAMPLE_RATE} Hz audio, got {sample_rate} Hz. "
            "Run preprocessing first."
        )

    duration = len(audio) / sample_rate

    print(f"Audio duration: {duration:.2f} seconds")

    return audio, duration


def diarize_audio(audio_path: Path):

    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    recording_name = audio_path.stem.replace(
        "_processed", ""
    )

    checkpoint_dir = Path("diarization") / (
        f"{recording_name}_chunks"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    final_path = (
        Path("diarization") /
        f"{recording_name}.json"
    )

    audio, duration = load_audio(audio_path)

    chunk_samples = CHUNK_DURATION * SAMPLE_RATE
    overlap_samples = CHUNK_OVERLAP * SAMPLE_RATE
    step_samples = chunk_samples - overlap_samples

    total_chunks = int(
        np.ceil(
            max(0, len(audio) - overlap_samples) /
            step_samples
        )
    )

    print(f"\nTotal chunks: {total_chunks}")
    print(f"Chunk duration: {CHUNK_DURATION}s")
    print(f"Chunk overlap: {CHUNK_OVERLAP}s")

    print("\nLoading pyannote model...")

    pipeline = Pipeline.from_pretrained(
        MODEL_NAME,
        token=True
    )

    pipeline.to(torch.device(DEVICE))

    print(f"Diarization running on: {DEVICE}")

    all_turns = []

    # Load completed chunks if resuming.
    for chunk_index in range(total_chunks):

        checkpoint_path = (
            checkpoint_dir /
            f"chunk_{chunk_index:04d}.json"
        )

        if checkpoint_path.exists():
            with checkpoint_path.open(
                "r",
                encoding="utf-8"
            ) as file:
                chunk_result = json.load(file)

            all_turns.extend(chunk_result["turns"])

            print(
                f"[{chunk_index + 1}/{total_chunks}] "
                "Checkpoint loaded."
            )

            continue

        start_sample = chunk_index * step_samples

        end_sample = min(
            start_sample + chunk_samples,
            len(audio)
        )

        chunk_audio = audio[start_sample:end_sample]

        chunk_start_time = (
            start_sample / SAMPLE_RATE
        )

        print(
            f"\n[{chunk_index + 1}/{total_chunks}] "
            f"Processing {chunk_start_time:.1f}s - "
            f"{end_sample / SAMPLE_RATE:.1f}s"
        )

        waveform = torch.from_numpy(
            chunk_audio.copy()
        ).unsqueeze(0)

        try:
            result = pipeline({
                "waveform": waveform,
                "sample_rate": SAMPLE_RATE
            })

            chunk_turns = []

            for turn, speaker in (
                result.speaker_diarization
            ):
                chunk_turns.append({
                    "start": round(
                        turn.start + chunk_start_time, 3
                    ),
                    "end": round(
                        turn.end + chunk_start_time, 3
                    ),
                    "speaker": str(speaker),
                    "chunk": chunk_index
                })

            # Save immediately after each successful chunk.
            save_json(
                {"turns": chunk_turns},
                checkpoint_path
            )

            all_turns.extend(chunk_turns)

            print(
                f"Detected {len(chunk_turns)} turns."
            )

        except KeyboardInterrupt:
            print("\nInterrupted. Completed chunks are saved.")
            raise

        except Exception as error:
            print(
                f"\nChunk {chunk_index} failed: {error}"
            )
            raise

        finally:
            del waveform
            if "result" in locals():
                del result

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not all_turns:
        raise RuntimeError(
            "No speaker turns were detected."
        )

    # Sort turns chronologically.
    all_turns.sort(
        key=lambda item: item["start"]
    )

    # Remove chunk metadata from final turns.
    # Speaker IDs are still local to each chunk at this stage.
    final_turns = []

    for turn in all_turns:
        final_turns.append({
            "start": turn["start"],
            "end": turn["end"],
            "speaker": turn["speaker"]
        })

    # IMPORTANT:
    # Do not normalize speaker IDs globally yet.
    # Cross-chunk speaker matching will be implemented next.

    speakers = sorted({
        turn["speaker"]
        for turn in final_turns
    })

    output = {
        "audio_file": str(audio_path),
        "number_of_speakers": len(speakers),
        "number_of_turns": len(final_turns),
        "speakers": speakers,
        "turns": final_turns
    }

    save_json(output, final_path)

    print("\nChunked diarization completed!")
    print(f"Chunks processed: {total_chunks}")
    print(f"Speaker labels found: {len(speakers)}")
    print(f"Speaker turns: {len(final_turns)}")
    print(f"Output: {final_path.resolve()}")


def main():

    if len(sys.argv) != 2:
        print(
            'Usage: python scripts/diarize.py '
            '"processed_audio/meeting_processed.wav"'
        )
        sys.exit(1)

    try:
        diarize_audio(Path(sys.argv[1]))

    except KeyboardInterrupt:
        print("\nDiarization stopped safely.")
        sys.exit(1)

    except Exception as error:
        print(f"\nDiarization failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
