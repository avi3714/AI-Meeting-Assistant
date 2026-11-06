
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from uuid import uuid4

from backend.pipeline import process_meeting


# --------------------------------------------------
# Background worker
# --------------------------------------------------

executor = ThreadPoolExecutor(max_workers=1)

jobs = {}
jobs_lock = Lock()


# --------------------------------------------------
# Create a new job
# --------------------------------------------------

def create_job(audio_path):
    job_id = str(uuid4())

    job = {
        "meeting_id": job_id,
        "audio_path": str(audio_path),
        "status": "queued",
        "current_stage": None,
        "stages": {},
        "results": {},
        "error": None,
        "cosine_threshold": None,
    }

    with jobs_lock:
        jobs[job_id] = job

    return job_id


# --------------------------------------------------
# Update processing stage
# --------------------------------------------------

def update_stage(job_id, stage_name, status, logs=""):
    with jobs_lock:
        job = jobs.get(job_id)

        if job is None:
            return

        job["current_stage"] = stage_name

        job["stages"][stage_name] = {
            "status": status,
            "logs": logs,
        }


# --------------------------------------------------
# Run processing in background
# --------------------------------------------------

def run_job(job_id, audio_path, cosine_threshold=0.72):

    try:
        # Update overall job status.

        with jobs_lock:
            job = jobs.get(job_id)

            if job is None:
                return

            job["status"] = "processing"
            job["cosine_threshold"] = cosine_threshold

        # Callback used by pipeline.py to update stage progress.

        def stage_callback(stage_name, status, logs=""):
            update_stage(
                job_id,
                stage_name,
                status,
                logs,
            )

        # Run the complete meeting pipeline.

        results = process_meeting(
            audio_path=audio_path,
            update_status=stage_callback,
            cosine_threshold=cosine_threshold,
        )

        # Store completed results.

        with jobs_lock:
            job = jobs.get(job_id)

            if job is None:
                return

            job["status"] = "completed"
            job["current_stage"] = None
            job["results"] = results
            job["error"] = None

    except Exception as error:

        with jobs_lock:
            job = jobs.get(job_id)

            if job is None:
                return

            job["status"] = "failed"
            job["error"] = str(error)


# --------------------------------------------------
# Start a background job
# --------------------------------------------------

def start_job(job_id, audio_path, cosine_threshold=0.72):

    with jobs_lock:
        job = jobs.get(job_id)

        if job is None:
            raise ValueError("Job not found.")

        if job["status"] != "queued":
            raise ValueError(
                f"Job cannot be started. Current status: {job['status']}"
            )

        if not 0.6 <= cosine_threshold <= 0.8:
            raise ValueError(
                "Cosine threshold must be between 0.6 and 0.8."
            )

        job["cosine_threshold"] = cosine_threshold

    executor.submit(
        run_job,
        job_id,
        audio_path,
        cosine_threshold,
    )


# --------------------------------------------------
# Retrieve job information
# --------------------------------------------------

def get_job(job_id):

    with jobs_lock:
        job = jobs.get(job_id)

        if job is None:
            return None

        # Return a copy so external code cannot directly
        # modify the stored job dictionary.

        return {
            **job,
            "stages": {
                name: stage.copy()
                for name, stage in job["stages"].items()
            },
            "results": job["results"].copy(),
        }
# --------------------------------------------------
# Update job results safely
# --------------------------------------------------

def update_job_results(job_id, new_results):

    with jobs_lock:
        job = jobs.get(job_id)

        if job is None:
            raise ValueError("Job not found.")

        job["results"].update(new_results)

        return job["results"].copy()