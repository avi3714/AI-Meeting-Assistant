
from pathlib import Path
import sys
import json

from faster_whisper import WhisperModel


CHUNK_DURATION = 300       # 5 minutes
CHUNK_OVERLAP = 5          # 5 seconds


def main():

    # STEP 1: Validate input.

    if len(sys.argv) != 2:
        print(
            'Usage: python scripts/transcribe.py '
            '"processed_audio/meeting_processed.wav"'
        )
        sys.exit(1)

    audio_path = Path(sys.argv[1])

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"Input audio: {audio_path}")

    # STEP 2: Load Faster-Whisper.

    print("\nLoading Faster-Whisper model...")

    model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8",
    )

    print("Model loaded successfully!")

    # STEP 3: Get audio duration.

    from faster_whisper.audio import decode_audio

    audio = decode_audio(str(audio_path), sampling_rate=16000)

    sample_rate = 16000
    total_duration = len(audio) / sample_rate

    print(f"\nAudio duration: {total_duration:.2f} seconds")

    # STEP 4: Divide audio into overlapping chunks.

    transcript_segments = []

    chunk_start = 0.0
    chunk_number = 1

    while chunk_start < total_duration:

        chunk_end = min(
            chunk_start + CHUNK_DURATION,
            total_duration
        )

        print(
            f"\nProcessing chunk {chunk_number}: "
            f"{chunk_start:.2f}s - {chunk_end:.2f}s"
        )

        start_sample = int(chunk_start * sample_rate)
        end_sample = int(chunk_end * sample_rate)

        audio_chunk = audio[start_sample:end_sample]

        # STEP 5: Transcribe the current chunk.

        segments, info = model.transcribe(
            audio_chunk,
            language="en",
            beam_size=5,
            vad_filter=True,
            word_timestamps=True,
        )

        for segment in segments:

            words = []

            if segment.words:

                for word in segment.words:

                    words.append({
                        "start": float(word.start) + chunk_start,
                        "end": float(word.end) + chunk_start,
                        "word": word.word,
                    })

            transcript_segments.append({
                "start": float(segment.start) + chunk_start,
                "end": float(segment.end) + chunk_start,
                "text": segment.text.strip(),
                "words": words,
            })

            print(
                f"[{segment.start + chunk_start:.2f}s - "
                f"{segment.end + chunk_start:.2f}s] "
                f"{segment.text.strip()}"
            )

        # STEP 6: Move forward, retaining overlap.

        if chunk_end >= total_duration:
            break

        chunk_start = chunk_end - CHUNK_OVERLAP
        chunk_number += 1

    # STEP 7: Sort segments chronologically.

    transcript_segments.sort(key=lambda item: item["start"])

    # STEP 8: Remove duplicate words caused by overlapping chunks.

    all_words = []

    for segment in transcript_segments:
        all_words.extend(segment["words"])

    all_words.sort(key=lambda item: item["start"])

    unique_words = []

    for word in all_words:

        if unique_words:

            previous = unique_words[-1]

            same_word = (
                word["word"].strip().lower()
                == previous["word"].strip().lower()
            )

            overlap = min(
                word["end"], previous["end"]
            ) - max(
                word["start"], previous["start"]
            )

            if same_word and overlap > 0:
                continue

        unique_words.append(word)

    # STEP 9: Rebuild segment structure from unique words.

    transcript_segments = []

    for word in unique_words:

        if (
            not transcript_segments
            or word["start"] - transcript_segments[-1]["end"] > 1.5
        ):
            transcript_segments.append({
                "start": word["start"],
                "end": word["end"],
                "text": word["word"].strip(),
                "words": [word],
            })

        else:
            current = transcript_segments[-1]
            current["end"] = word["end"]
            current["text"] += word["word"]
            current["words"].append(word)

    # STEP 10: Save JSON output.

    output_dir = Path("transcriptions")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = audio_path.stem.replace("_processed", "")
    output_path = output_dir / f"{output_name}.json"

    result = {
        "audio_file": audio_path.name,
        "language": "en",
        "duration": total_duration,
        "segments": transcript_segments,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nChunked transcription completed!")
    print(f"Chunks processed: {chunk_number}")
    print(f"Unique words: {len(unique_words)}")
    print(f"Segments: {len(transcript_segments)}")
    print(f"JSON saved to: {output_path.resolve()}")


if __name__ == "__main__":

    try:
        main()

    except Exception as error:
        print(f"\nTranscription failed: {error}")
        sys.exit(1)
