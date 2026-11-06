
from pathlib import Path
import subprocess
import sys


# --------------------------------------------------
# Project directories
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


# --------------------------------------------------
# Run an individual processing stage
# --------------------------------------------------

def run_stage(stage_name, command, update_status):

    update_status(stage_name, "running", "")

    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        logs = result.stdout or ""

        if result.stderr:
            logs += "\n" + result.stderr

        update_status(
            stage_name,
            "completed",
            logs,
        )

    except subprocess.CalledProcessError as error:

        logs = error.stdout or ""
        logs += "\n" + (error.stderr or "")

        update_status(
            stage_name,
            "failed",
            logs,
        )

        raise RuntimeError(
            f"Stage '{stage_name}' failed.\n{logs}"
        ) from error


# --------------------------------------------------
# Process complete meeting
# --------------------------------------------------

def process_meeting(
    audio_path,
    update_status,
    cosine_threshold=0.72,
):

    audio_path = Path(audio_path).resolve()

    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    # Validate cosine similarity threshold.

    if not 0.6 <= cosine_threshold <= 0.8:
        raise ValueError(
            "Cosine threshold must be between 0.6 and 0.8."
        )

    meeting_name = audio_path.stem

    processed_audio = (
        PROJECT_ROOT
        / "processed_audio"
        / f"{meeting_name}_processed.wav"
    )

    python = sys.executable

    # --------------------------------------------------
    # Processing stages
    # --------------------------------------------------

    stages = [

        (
            "Preprocessing",
            [
                python,
                str(SCRIPTS_DIR / "preprocess_audio.py"),
                str(audio_path),
            ],
        ),

        (
            "Speaker diarization",
            [
                python,
                str(SCRIPTS_DIR / "diarize.py"),
                str(processed_audio),
            ],
        ),

        (
            "Transcription",
            [
                python,
                str(SCRIPTS_DIR / "transcribe.py"),
                str(processed_audio),
            ],
        ),

        (
            "Speaker matching",
            [
                python,
                str(SCRIPTS_DIR / "match_speakers.py"),
                str(processed_audio),
                str(cosine_threshold),
            ],
        ),

        (
            "Transcript merging",
            [
                python,
                str(SCRIPTS_DIR / "merge_transcript.py"),
                meeting_name,
            ],
        ),

        (
            "Summary generation",
            [
                python,
                str(SCRIPTS_DIR / "generate_summary.py"),
                meeting_name,
            ],
        ),
    ]

    # --------------------------------------------------
    # Execute stages sequentially
    # --------------------------------------------------

    total_stages = len(stages)

    for index, (stage_name, command) in enumerate(
        stages,
        start=1,
    ):

        update_status(
            stage_name,
            "running",
            f"Starting stage {index}/{total_stages}",
        )

        run_stage(
            stage_name,
            command,
            update_status,
        )

    # --------------------------------------------------
    # Verify generated output files
    # --------------------------------------------------

    expected_files = [

        PROJECT_ROOT
        / "transcripts"
        / f"{meeting_name}_final.txt",

        PROJECT_ROOT
        / "reports"
        / f"{meeting_name}_transcript.pdf",

        PROJECT_ROOT
        / "reports"
        / f"{meeting_name}_summary.txt",

        PROJECT_ROOT
        / "reports"
        / f"{meeting_name}_summary.pdf",
    ]

    missing_files = [
        str(path)
        for path in expected_files
        if not path.exists()
    ]

    if missing_files:

        raise RuntimeError(
            "Processing finished, but some output files are missing:\n"
            + "\n".join(missing_files)
        )

    # --------------------------------------------------
    # Return generated file paths
    # --------------------------------------------------

    return {
        "transcript": str(expected_files[0]),
        "transcript_pdf": str(expected_files[1]),
        "summary": str(expected_files[2]),
        "summary_pdf": str(expected_files[3]),
    }
