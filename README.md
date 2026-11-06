# AI Meeting Assistant

An AI-powered meeting assistant that converts meeting audio into speaker-attributed transcripts, structured summaries, and PDF reports.

## Features

- Audio preprocessing using FFmpeg
- Speaker diarization using Pyannote Audio
- Speech-to-text transcription using Faster-Whisper
- Speaker matching using SpeechBrain ECAPA embeddings
- Transcript merging with speaker labels and timestamps
- AI-powered summarization using Ollama and Qwen 2.5
- PDF generation for transcripts and meeting summaries
- FastAPI backend
- Next.js frontend

## Technology Stack

- Frontend: Next.js, React
- Backend: Python, FastAPI
- Audio processing: FFmpeg
- Transcription: Faster-Whisper
- Speaker diarization: Pyannote Audio
- Speaker matching: SpeechBrain ECAPA
- Summarization: Ollama, Qwen 2.5
- PDF generation: ReportLab

## Project Structure

- `backend/` - API and meeting-processing pipeline
- `frontend/` - Web application
- `scripts/` - Audio processing, transcription, diarization, and summarization scripts
- `requirements.txt` - Python dependencies

Generated directories such as `audio/`, `processed_audio/`, `diarization/`, `transcriptions/`, `transcripts/`, and `reports/` are created automatically when needed.

## Prerequisites

- Python 3.12
- Node.js and npm
- FFmpeg
- Ollama
- Hugging Face account with access to the required Pyannote model

## Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
