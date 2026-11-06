from pathlib import Path
from threading import Lock
from uuid import uuid4
import re

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.jobs import (
    create_job,
    start_job,
    get_job,
    update_job_results,
)

from scripts.merge_transcript import build_pdf
from scripts.generate_summary import generate_summary_from_transcript


# --------------------------------------------------
# Project directories
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIO_DIR = PROJECT_ROOT / "audio"
REPORTS_DIR = PROJECT_ROOT / "reports"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="AI Meeting Assistant API",
    description="Backend for audio transcription and meeting summaries",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

uploaded_audio_paths = {}
uploaded_lock = Lock()

ALLOWED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
}


# --------------------------------------------------
# Request models
# --------------------------------------------------

class TranscriptUpdateRequest(BaseModel):
    speaker_names: dict[str, str] = Field(default_factory=dict)
    unknown_assignments: dict[int, str] = Field(default_factory=dict)
    paragraph_speakers: dict[int, str] = Field(default_factory=dict)


class ProcessMeetingRequest(BaseModel):
    cosine_threshold: float = Field(
        default=0.72,
        ge=0.6,
        le=0.8,
    )


# --------------------------------------------------
# Transcript parsing
# --------------------------------------------------

SEGMENT_PATTERN = re.compile(
    r"^\[(\d+(?:\.\d+)?)s\s*-\s*(\d+(?:\.\d+)?)s\]\s*([^:\n]+):\s*(.*)$"
)

OVERLAP_PATTERN = re.compile(
    r"^\[(\d+(?:\.\d+)?)s\s*-\s*(\d+(?:\.\d+)?)s\]\s*(.+?)\s+and\s+(.+?)\s*$"
)


def parse_transcript(transcript_text: str):

    if "TRANSCRIPT" not in transcript_text:
        raise ValueError("TRANSCRIPT section not found.")

    transcript_section = transcript_text.split(
        "TRANSCRIPT",
        1,
    )[1]

    transcript_section = transcript_section.split(
        "OVERLAPPING SPEECH",
        1,
    )[0]

    segments = []

    for line in transcript_section.splitlines():

        match = SEGMENT_PATTERN.match(line.strip())

        if not match:
            continue

        segments.append({
            "start": float(match.group(1)),
            "end": float(match.group(2)),
            "speaker": match.group(3).strip(),
            "text": match.group(4).strip(),
        })

    if not segments:
        raise ValueError("No transcript segments could be parsed.")

    return segments


def parse_overlaps(transcript_text: str, speaker_names: dict[str, str]):

    overlaps_found = []

    if "OVERLAPPING SPEECH" not in transcript_text:
        return overlaps_found

    overlap_section = transcript_text.split(
        "OVERLAPPING SPEECH",
        1,
    )[1]

    for line in overlap_section.splitlines():

        match = OVERLAP_PATTERN.match(line.strip())

        if not match:
            continue

        speaker1 = match.group(3).strip()
        speaker2 = match.group(4).strip()

        speaker1 = speaker_names.get(speaker1, "").strip() or speaker1
        speaker2 = speaker_names.get(speaker2, "").strip() or speaker2

        overlaps_found.append({
            "start": float(match.group(1)),
            "end": float(match.group(2)),
            "speaker1": speaker1,
            "speaker2": speaker2,
        })

    return overlaps_found


def get_recording_name(transcript_text: str, meeting_id: str):

    first_line = (
        transcript_text.splitlines()[0].strip()
        if transcript_text else ""
    )

    prefix = "Meeting Transcript:"

    if first_line.startswith(prefix):

        recording_name = first_line[len(prefix):].strip()

        if recording_name:
            return recording_name

    return meeting_id


# --------------------------------------------------
# Corrected transcript helpers
# --------------------------------------------------

def get_speaker_metadata(segments):

    speakers = sorted({
        str(segment["speaker"]).strip()
        for segment in segments
        if str(segment["speaker"]).strip().upper() != "UNKNOWN"
    })

    duration = max(
        (float(segment["end"]) for segment in segments),
        default=0.0,
    )

    unknown_turns = sum(
        1 for segment in segments
        if str(segment["speaker"]).strip().upper() == "UNKNOWN"
    )

    return {
        "speakers": speakers,
        "speaker_count": len(speakers),
        "turn_count": len(segments),
        "duration": duration,
        "unknown_turns": unknown_turns,
    }


def merge_consecutive_segments(segments, max_pause=1.5):

    if not segments:
        return []

    merged = []

    for segment in segments:

        current = segment.copy()

        if not merged:
            merged.append(current)
            continue

        previous = merged[-1]

        same_speaker = previous["speaker"] == current["speaker"]
        gap = current["start"] - previous["end"]

        if same_speaker and 0 <= gap <= max_pause:

            previous["end"] = current["end"]

            previous["text"] = (
                previous["text"].rstrip()
                + " "
                + current["text"].lstrip()
            )

        else:
            merged.append(current)

    return merged


def build_corrected_transcript_text(
    original_text,
    segments,
    recording_name,
):

    lines = [
        f"Meeting Transcript: {recording_name}",
        "",
        "TRANSCRIPT",
        "----------",
    ]

    for segment in segments:

        lines.append(
            f"[{segment['start']:.2f}s - {segment['end']:.2f}s] "
            f"{segment['speaker']}: {segment['text']}"
        )

    lines.extend([
        "",
        "OVERLAPPING SPEECH",
        "------------------",
        "Overlap information is not recalculated after manual speaker corrections.",
    ])

    return "\n".join(lines)


# --------------------------------------------------
# Updated transcript PDF
# --------------------------------------------------

def generate_updated_transcript_pdf(
    meeting_id,
    segments,
    transcript_text,
    speaker_names,
    output_path,
):

    recording_name = get_recording_name(
        transcript_text,
        meeting_id,
    )

    overlaps_found = parse_overlaps(
        transcript_text,
        speaker_names,
    )

    unknown_count = sum(
        len(segment["text"].split())
        for segment in segments
        if str(segment["speaker"]).strip().upper() == "UNKNOWN"
    )

    duration = max(
        (segment["end"] for segment in segments),
        default=0.0,
    )

    metadata = get_speaker_metadata(segments)

    diarization = {
        "number_of_speakers": metadata["speaker_count"]
    }

    build_pdf(
        recording_name=recording_name,
        diarization=diarization,
        segments=segments,
        overlaps_found=overlaps_found,
        unknown_count=unknown_count,
        pdf_path=output_path,
        duration=duration,
    )


# --------------------------------------------------
# Home
# --------------------------------------------------

@app.get("/")
def home():
    return {"message": "AI Meeting Assistant API is running"}


# --------------------------------------------------
# Upload audio
# --------------------------------------------------

@app.post("/meetings/upload")
async def upload_meeting(file: UploadFile = File(...)):

    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format.",
        )

    meeting_id = str(uuid4())

    audio_path = AUDIO_DIR / f"{meeting_id}{extension}"

    try:

        with audio_path.open("wb") as output:

            while chunk := await file.read(1024 * 1024):
                output.write(chunk)

    except Exception:

        audio_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail="Could not save uploaded audio.",
        )

    finally:
        await file.close()

    job_id = create_job(audio_path)

    with uploaded_lock:
        uploaded_audio_paths[job_id] = audio_path

    return {
        "meeting_id": job_id,
        "filename": file.filename,
        "status": "uploaded",
        "message": "Audio uploaded successfully.",
    }
    # --------------------------------------------------
# Start processing
# --------------------------------------------------

@app.post("/meetings/{meeting_id}/process")
def start_meeting_processing(
    meeting_id: str,
    request: ProcessMeetingRequest = ProcessMeetingRequest(),
):

    job = get_job(meeting_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Meeting not found.",
        )

    if job["status"] != "queued":
        raise HTTPException(
            status_code=409,
            detail=f"Meeting cannot be started. Current status: {job['status']}",
        )

    with uploaded_lock:
        audio_path = uploaded_audio_paths.get(meeting_id)

    if audio_path is None or not audio_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Uploaded audio file not found.",
        )

    start_job(
        meeting_id,
        audio_path,
        request.cosine_threshold,
    )

    return {
        "meeting_id": meeting_id,
        "status": "processing",
        "cosine_threshold": request.cosine_threshold,
        "message": "Meeting processing has started.",
    }


# --------------------------------------------------
# Processing status
# --------------------------------------------------

@app.get("/meetings/{meeting_id}/status")
def meeting_status(meeting_id: str):

    job = get_job(meeting_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Meeting not found.",
        )

    return job


# --------------------------------------------------
# Completed results
# --------------------------------------------------

@app.get("/meetings/{meeting_id}/results")
def meeting_results(meeting_id: str):

    job = get_job(meeting_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Meeting not found.",
        )

    if job["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail="Meeting processing is not completed yet.",
        )

    return {
        "meeting_id": meeting_id,
        "results": job["results"],
    }


# --------------------------------------------------
# Generate corrected transcript AND corrected summary
# --------------------------------------------------

@app.post("/meetings/{meeting_id}/transcript/update")
def update_transcript(
    meeting_id: str,
    request: TranscriptUpdateRequest,
):

    job = get_job(meeting_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Meeting not found.",
        )

    if job["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail="Meeting processing is not completed.",
        )

    results = job.get("results", {})
    transcript_value = results.get("transcript")

    if not transcript_value:
        raise HTTPException(
            status_code=404,
            detail="Original transcript path is missing from meeting results.",
        )

    transcript_path = Path(transcript_value)

    if not transcript_path.is_absolute():
        transcript_path = PROJECT_ROOT / transcript_path

    if not transcript_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Original transcript file not found.",
        )

    try:
        original_text = transcript_path.read_text(encoding="utf-8")
        original_segments = parse_transcript(original_text)

        available_speakers = {
            segment["speaker"]
            for segment in original_segments
            if segment["speaker"].upper() != "UNKNOWN"
        }

        # Validate individual paragraph changes.

        for index, speaker in request.paragraph_speakers.items():

            if index < 0 or index >= len(original_segments):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid paragraph index: {index}",
                )

            if speaker not in available_speakers:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid speaker ID: {speaker}",
                )

        # Validate UNKNOWN assignments.

        for index, speaker in request.unknown_assignments.items():

            if index < 0 or index >= len(original_segments):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid paragraph index: {index}",
                )

            if original_segments[index]["speaker"].upper() != "UNKNOWN":
                raise HTTPException(
                    status_code=400,
                    detail=f"Paragraph {index} is not UNKNOWN.",
                )

            if speaker not in available_speakers:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid speaker ID: {speaker}",
                )

        # Apply corrections to a new copy.

        corrected_segments = []

        for index, segment in enumerate(original_segments):

            updated_segment = segment.copy()
            final_speaker = segment["speaker"]

            # Individual paragraph modification takes priority.

            if index in request.paragraph_speakers:
                final_speaker = request.paragraph_speakers[index]

            # Apply manual assignment to UNKNOWN paragraphs.

            elif final_speaker.upper() == "UNKNOWN":

                assigned_speaker = request.unknown_assignments.get(index)

                if assigned_speaker:
                    final_speaker = assigned_speaker

            # Replace speaker ID with entered participant name.

            entered_name = request.speaker_names.get(
                final_speaker,
                "",
            ).strip()

            updated_segment["speaker"] = entered_name or final_speaker

            corrected_segments.append(updated_segment)

        # Merge consecutive segments with the same final identity.

        corrected_segments = merge_consecutive_segments(
            corrected_segments,
            max_pause=1.5,
        )

        recording_name = get_recording_name(
            original_text,
            meeting_id,
        )

        corrected_text = build_corrected_transcript_text(
            original_text,
            corrected_segments,
            recording_name,
        )

        # Save corrected transcript as a separate TXT file.

        updated_transcript_txt = (
            REPORTS_DIR / f"{meeting_id}_updated_transcript.txt"
        )

        updated_transcript_txt.write_text(
            corrected_text,
            encoding="utf-8",
        )

        # Generate corrected transcript PDF.

        updated_transcript_pdf = (
            REPORTS_DIR / f"{meeting_id}_updated_transcript.pdf"
        )

        generate_updated_transcript_pdf(
            meeting_id=meeting_id,
            segments=corrected_segments,
            transcript_text=original_text,
            speaker_names=request.speaker_names,
            output_path=updated_transcript_pdf,
        )

        # Build metadata from the corrected transcript itself.

        corrected_metadata = get_speaker_metadata(
            corrected_segments,
        )

        # Generate a new summary from the corrected transcript.

        updated_summary_txt = (
            REPORTS_DIR / f"{meeting_id}_updated_summary.txt"
        )

        updated_summary_pdf = (
            REPORTS_DIR / f"{meeting_id}_updated_summary.pdf"
        )

        summary_text = generate_summary_from_transcript(
            transcript_text=corrected_text,
            metadata=corrected_metadata,
            meeting_name=recording_name,
            output_txt=updated_summary_txt,
            output_pdf=updated_summary_pdf,
        )

        # Verify that all expected files were actually generated.

        expected_files = [
            updated_transcript_txt,
            updated_transcript_pdf,
            updated_summary_txt,
            updated_summary_pdf,
        ]

        missing_files = [
            str(path)
            for path in expected_files
            if not path.is_file()
        ]

        if missing_files:
            raise RuntimeError(
                "The following updated report files were not generated: "
                + ", ".join(missing_files)
            )

        # IMPORTANT FIX:
        # get_job() returns a copy of the job.
        # Therefore, do not assign paths through job["results"].
        # update_job_results() updates the original stored job.

        update_job_results(
            meeting_id,
            {
                "updated_transcript_txt": str(updated_transcript_txt),
                "updated_transcript_pdf": str(updated_transcript_pdf),
                "updated_summary": str(updated_summary_txt),
                "updated_summary_pdf": str(updated_summary_pdf),
            },
        )

        return {
            "meeting_id": meeting_id,
            "message": "Updated transcript and summary generated successfully.",
            "updated_transcript_pdf": str(updated_transcript_pdf),
            "updated_summary_pdf": str(updated_summary_pdf),
            "updated_summary": summary_text,
            "paragraphs_before": len(original_segments),
            "paragraphs_after": len(corrected_segments),
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not generate updated transcript and summary: "
                f"{str(error)}"
            ),
        )
        # --------------------------------------------------
# Download files
# --------------------------------------------------

@app.get("/meetings/{meeting_id}/download/{file_type}")
def download_result(meeting_id: str, file_type: str):

    job = get_job(meeting_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Meeting not found.",
        )

    if job["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail="Results are not ready yet.",
        )

    file_mapping = {
        "transcript": (
            "transcript",
            "text/plain",
        ),

        "transcript_pdf": (
            "transcript_pdf",
            "application/pdf",
        ),

        "summary": (
            "summary",
            "text/plain",
        ),

        "summary_pdf": (
            "summary_pdf",
            "application/pdf",
        ),

        "updated_transcript_txt": (
            "updated_transcript_txt",
            "text/plain",
        ),

        "updated_transcript_pdf": (
            "updated_transcript_pdf",
            "application/pdf",
        ),

        "updated_summary": (
            "updated_summary",
            "text/plain",
        ),

        "updated_summary_pdf": (
            "updated_summary_pdf",
            "application/pdf",
        ),
    }

    if file_type not in file_mapping:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type.",
        )

    result_key, media_type = file_mapping[file_type]

    result_value = job.get("results", {}).get(result_key)

    if not result_value:
        raise HTTPException(
            status_code=404,
            detail=f"Result '{file_type}' has not been generated yet.",
        )

    file_path = Path(result_value)

    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Result file not found.",
        )

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=file_path.name,
    )