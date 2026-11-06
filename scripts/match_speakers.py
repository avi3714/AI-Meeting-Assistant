from pathlib import Path
import json
import sys
import shutil

import numpy as np
import soundfile as sf
import torch
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

SAMPLE_RATE = 16000

CHUNK_DURATION = 30
CHUNK_OVERLAP = 5
STEP_DURATION = CHUNK_DURATION - CHUNK_OVERLAP

MAX_SPEAKERS = 4

GROUP_GAP_SECONDS = 1.25

MIN_PROFILE_SECONDS = 4.0
MIN_MATCH_SECONDS = 1.5

MATCH_THRESHOLD = 0.72
MATCH_MARGIN = 0.035

MAX_EMBEDDINGS_PER_PROFILE = 8

MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_DIR = "pretrained_models/ecapa"


# ------------------------------------------------------------
# Similarity and embedding functions
# ------------------------------------------------------------

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    return float(
        np.dot(a, b) /
        (
            (np.linalg.norm(a) + 1e-10) *
            (np.linalg.norm(b) + 1e-10)
        )
    )


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)
    return vector / (np.linalg.norm(vector) + 1e-10)


def load_audio(path):
    audio, sample_rate = sf.read(
        str(path),
        dtype="float32"
    )

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Expected {SAMPLE_RATE} Hz audio, got {sample_rate} Hz"
        )

    return audio


def load_checkpoints(folder):
    paths = sorted(folder.glob("chunk_*.json"))

    if not paths:
        raise FileNotFoundError(
            f"No chunk checkpoints found in {folder}"
        )

    checkpoints = []

    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            checkpoints.append(json.load(f))

    return checkpoints


def extract_embedding(model, audio):
    waveform = torch.tensor(
        audio,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        embedding = model.encode_batch(waveform)

    return normalize(
        embedding.squeeze().cpu().numpy()
    )


# ------------------------------------------------------------
# Build temporal speech groups
# ------------------------------------------------------------

def build_groups(checkpoints, audio_duration):
    """
    Build temporally local speech groups from each chunk's
    local speaker turns.

    Chunk ownership windows prevent the same overlap audio
    from being counted twice.

    Groups are kept separate when the same local label
    reappears after a silence gap.
    """

    groups = []

    for chunk_index, checkpoint in enumerate(checkpoints):

        owner_start = chunk_index * STEP_DURATION

        owner_end = min(
            (chunk_index + 1) * STEP_DURATION,
            audio_duration
        )

        if chunk_index == len(checkpoints) - 1:
            owner_end = audio_duration

        by_speaker = {}

        for turn in checkpoint.get("turns", []):

            start = max(
                float(turn["start"]),
                owner_start,
                0.0
            )

            end = min(
                float(turn["end"]),
                owner_end,
                audio_duration
            )

            if end <= start:
                continue

            by_speaker.setdefault(
                str(turn["speaker"]),
                []
            ).append((start, end))

        for local_speaker, intervals in by_speaker.items():

            intervals.sort()

            current = []
            current_end = None

            for start, end in intervals:

                if (
                    current
                    and start - current_end > GROUP_GAP_SECONDS
                ):

                    groups.append(
                        make_group(
                            chunk_index,
                            local_speaker,
                            current
                        )
                    )

                    current = []

                current.append((start, end))

                current_end = max(
                    end,
                    current_end or end
                )

            if current:

                groups.append(
                    make_group(
                        chunk_index,
                        local_speaker,
                        current
                    )
                )

    groups.sort(
        key=lambda g: (
            g["start"],
            g["end"],
            g["chunk"]
        )
    )

    return groups


def make_group(chunk, local_speaker, intervals):

    return {
        "chunk": chunk,
        "local_speaker": local_speaker,

        "start": min(
            x[0] for x in intervals
        ),

        "end": max(
            x[1] for x in intervals
        ),

        "intervals": intervals,

        "speech_duration": sum(
            end - start
            for start, end in intervals
        ),
    }


# ------------------------------------------------------------
# Extract audio for each group
# ------------------------------------------------------------

def group_audio(group, audio):

    # Concatenate only the speech intervals.
    # Do not include long silences.

    pieces = []

    for start, end in group["intervals"]:

        left = max(
            0,
            int(start * SAMPLE_RATE)
        )

        right = min(
            len(audio),
            int(end * SAMPLE_RATE)
        )

        if right > left:
            pieces.append(
                audio[left:right]
            )

    if not pieces:
        return np.array([], dtype=np.float32)

    combined = np.concatenate(pieces)

    # Keep the first 20 seconds to avoid
    # oversized embedding inputs.

    return combined[:20 * SAMPLE_RATE]


# ------------------------------------------------------------
# Speaker profile similarity
# ------------------------------------------------------------

def profile_similarity(embedding, profile):

    # Compare with multiple stored examples.
    # Average the best two matches when possible.

    scores = sorted(
        (
            cosine_similarity(embedding, item)
            for item in profile["embeddings"]
        ),
        reverse=True
    )

    if not scores:
        return -1.0

    return (
        float(np.mean(scores[:2]))
        if len(scores) >= 2
        else scores[0]
    )


# ------------------------------------------------------------
# Match groups against speaker profiles
# ------------------------------------------------------------

def match_groups(
    groups,
    audio,
    model,
    match_threshold=MATCH_THRESHOLD
):

    profiles = []
    decisions = []

    for index, group in enumerate(groups):

        sample = group_audio(
            group,
            audio
        )

        duration = len(sample) / SAMPLE_RATE

        record = {
            "group": index,

            "chunk": group["chunk"],

            "local_speaker": group["local_speaker"],

            "start": round(group["start"], 3),

            "end": round(group["end"], 3),

            "speech_duration": round(duration, 3),

            "speaker": "UNKNOWN",

            "best_similarity": None,

            "second_similarity": None,

            "decision": "TOO_SHORT",
        }

        if duration < MIN_MATCH_SECONDS:

            decisions.append(record)
            continue

        embedding = extract_embedding(
            model,
            sample
        )

        scored = [
            (
                profile_similarity(embedding, p),
                i
            )
            for i, p in enumerate(profiles)
        ]

        scored.sort(reverse=True)

        best_score = (
            scored[0][0]
            if scored
            else -1.0
        )

        second_score = (
            scored[1][0]
            if len(scored) > 1
            else -1.0
        )

        record["best_similarity"] = (
            round(best_score, 4)
            if scored
            else None
        )

        record["second_similarity"] = (
            round(second_score, 4)
            if len(scored) > 1
            else None
        )

        confident_match = (
            bool(scored)
            and best_score >= match_threshold
            and (
                len(scored) == 1
                or best_score - second_score >= MATCH_MARGIN
            )
        )

        if confident_match:

            profile_index = scored[0][1]

            profile = profiles[profile_index]

            record["speaker"] = profile["speaker"]

            record["decision"] = "MATCHED"

            # Only update a profile with sufficiently
            # long, confidently matched speech.

            if (
                duration >= MIN_PROFILE_SECONDS
                and len(profile["embeddings"])
                < MAX_EMBEDDINGS_PER_PROFILE
            ):

                profile["embeddings"].append(
                    embedding
                )

                profile["samples"].append({
                    "start": group["start"],
                    "end": group["end"],
                    "duration": duration
                })

        elif (
            duration >= MIN_PROFILE_SECONDS
            and len(profiles) < MAX_SPEAKERS
        ):

            speaker_id = (
                f"SPEAKER_{len(profiles):02d}"
            )

            profiles.append({
                "speaker": speaker_id,

                "embeddings": [embedding],

                "samples": [{
                    "start": group["start"],
                    "end": group["end"],
                    "duration": duration
                }],
            })

            record["speaker"] = speaker_id

            record["decision"] = "NEW_PROFILE"

        else:

            record["decision"] = (
                "UNCERTAIN_OR_LIMIT_REACHED"
            )

        decisions.append(record)

        print(
            f"{group['start']:7.2f}-{group['end']:7.2f}s | "
            f"chunk {group['chunk']:02d} "
            f"{group['local_speaker']:<12} | "
            f"{duration:5.1f}s | "
            f"{record['speaker']:<10} | "
            f"{record['decision']} | "
            f"sim={record['best_similarity']}"
        )

    return profiles, decisions


# ------------------------------------------------------------
# Reconstruct final speaker turns
# ------------------------------------------------------------

def reconstruct_turns(
    checkpoints,
    decisions,
    audio_duration
):

    # Each profile decision belongs to one
    # (chunk, local speaker, temporal group).

    group_map = {}

    for d in decisions:

        key = (
            d["chunk"],
            d["local_speaker"]
        )

        group_map.setdefault(
            key,
            []
        ).append(d)

    all_turns = []

    for chunk_index, checkpoint in enumerate(checkpoints):

        owner_start = chunk_index * STEP_DURATION

        owner_end = min(
            (chunk_index + 1) * STEP_DURATION,
            audio_duration
        )

        if chunk_index == len(checkpoints) - 1:
            owner_end = audio_duration

        for turn in checkpoint.get("turns", []):

            start = max(
                float(turn["start"]),
                owner_start,
                0.0
            )

            end = min(
                float(turn["end"]),
                owner_end,
                audio_duration
            )

            if end <= start:
                continue

            candidates = group_map.get(
                (
                    chunk_index,
                    str(turn["speaker"])
                ),
                []
            )

            assigned = "UNKNOWN"

            # Assign using the temporal group
            # that overlaps this turn the most.

            best_overlap = 0.0

            for candidate in candidates:

                overlap = max(
                    0.0,
                    min(end, candidate["end"])
                    - max(start, candidate["start"])
                )

                if overlap > best_overlap:

                    best_overlap = overlap

                    assigned = candidate["speaker"]

            all_turns.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": assigned,
            })

    all_turns.sort(
        key=lambda x: (
            x["start"],
            x["end"]
        )
    )

    return all_turns


# ------------------------------------------------------------
# Main speaker matching function
# ------------------------------------------------------------

def run(
    audio_path,
    match_threshold=MATCH_THRESHOLD
):

    recording_name = (
        audio_path.stem.replace("_processed", "")
    )

    checkpoint_dir = (
        Path("diarization")
        / f"{recording_name}_chunks"
    )

    output_path = (
        Path("diarization")
        / f"{recording_name}.json"
    )

    diagnostics_path = (
        Path("diarization")
        / f"{recording_name}_profile_diagnostics.json"
    )

    audio = load_audio(audio_path)

    duration = len(audio) / SAMPLE_RATE

    checkpoints = load_checkpoints(
        checkpoint_dir
    )

    print(
        f"Loaded {len(checkpoints)} checkpoints; "
        f"audio duration {duration:.1f}s"
    )

    print("Loading ECAPA model on CPU...")

    model = EncoderClassifier.from_hparams(
        source=MODEL_SOURCE,
        savedir=MODEL_DIR,
        local_strategy=LocalStrategy.COPY,
        run_opts={"device": "cpu"},
    )

    groups = build_groups(
        checkpoints,
        duration
    )

    print(
        f"Built {len(groups)} temporal speech groups"
    )

    profiles, decisions = match_groups(
        groups,
        audio,
        model,
        match_threshold
    )

    turns = reconstruct_turns(
        checkpoints,
        decisions,
        duration
    )

    speakers = [
        p["speaker"]
        for p in profiles
    ]

    output = {
        "audio_file": str(audio_path),

        "number_of_speakers": len(speakers),

        "number_of_turns": len(turns),

        "speakers": speakers,

        "turns": turns,
    }

    # Never silently overwrite an existing
    # diarization result.

    if output_path.exists():

        backup_path = output_path.with_name(
            output_path.stem
            + "_before_profile_matcher.json"
        )

        if not backup_path.exists():

            shutil.copy2(
                output_path,
                backup_path
            )

            print(
                f"Backed up existing result to {backup_path}"
            )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2
        )

    diagnostics = {

        "configuration": {

            "max_speakers": MAX_SPEAKERS,

            "group_gap_seconds": GROUP_GAP_SECONDS,

            "min_profile_seconds": MIN_PROFILE_SECONDS,

            "min_match_seconds": MIN_MATCH_SECONDS,

            "match_threshold": match_threshold,

            "match_margin": MATCH_MARGIN,
        },

        "profiles": [

            {
                "speaker": p["speaker"],

                "sample_count": len(p["samples"]),

                "samples": p["samples"],
            }

            for p in profiles
        ],

        "groups": decisions,
    }

    with diagnostics_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            diagnostics,
            f,
            indent=2
        )

    print("\n========== RESULT ==========")

    print(
        f"Profiles created: {len(profiles)} "
        f"(maximum {MAX_SPEAKERS})"
    )

    print(
        f"Final turns: {len(turns)}"
    )

    print(
        "UNKNOWN turns: "
        f"{sum(t['speaker'] == 'UNKNOWN' for t in turns)}"
    )

    print(
        f"Saved diarization: {output_path}"
    )

    print(
        f"Saved diagnostics: {diagnostics_path}"
    )

    print("===========================")


# ------------------------------------------------------------
# Backend-compatible entry point
# ------------------------------------------------------------

def match_speakers(
    audio_path,
    cosine_threshold=MATCH_THRESHOLD
):

    """Backend-compatible entry point; uses the original profile matcher."""

    return run(
        Path(audio_path),
        cosine_threshold
    )


# ------------------------------------------------------------
# Command-line execution
# ------------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) not in (2, 3):

        print(
            "Usage: python .\\scripts\\match_speakers.py "
            "<audio_path> [match_threshold]"
        )

        sys.exit(1)

    try:

        threshold = (
            float(sys.argv[2])
            if len(sys.argv) == 3
            else MATCH_THRESHOLD
        )

        if not 0.6 <= threshold <= 0.8:

            raise ValueError(
                "Match threshold must be between 0.60 and 0.80."
            )

        run(
            Path(sys.argv[1]),
            threshold
        )

    except Exception as exc:

        print(
            f"\nSpeaker matching failed: {exc}"
        )

        raise