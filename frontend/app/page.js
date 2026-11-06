"use client";

import { useState } from "react";

const API_URL = "http://127.0.0.1:8000";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export default function Home() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);

  // Cosine similarity threshold
  const [cosineThreshold, setCosineThreshold] = useState(0.72);

  const [message, setMessage] = useState("");
  const [meetingId, setMeetingId] = useState(null);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState("");

  // Original transcript states
  const [transcript, setTranscript] = useState([]);
  const [speakerNames, setSpeakerNames] = useState({});
  const [unknownAssignments, setUnknownAssignments] = useState({});
  const [paragraphSpeakers, setParagraphSpeakers] = useState({});
  const [transcriptLoading, setTranscriptLoading] = useState(false);

  // Updated transcript and summary states
  const [generatingUpdatedTranscript, setGeneratingUpdatedTranscript] =
    useState(false);

  const [updatedTranscriptReady, setUpdatedTranscriptReady] =
    useState(false);

  const [updatedSummary, setUpdatedSummary] = useState("");
  const [updatedSummaryReady, setUpdatedSummaryReady] = useState(false);

  // --------------------------------------------------
  // Parse transcript text
  // --------------------------------------------------

  function parseTranscript(transcriptText) {
    if (!transcriptText.includes("TRANSCRIPT")) {
      throw new Error("The TRANSCRIPT section was not found.");
    }

    const transcriptSection = transcriptText
      .split("TRANSCRIPT", 2)[1]
      .split("OVERLAPPING SPEECH", 1)[0];

    // Expected format:
    // [0.00s - 4.80s] SPEAKER_00: spoken text
    const segmentPattern =
      /^\[(\d+(?:\.\d+)?)s\s*-\s*(\d+(?:\.\d+)?)s\]\s*([^:\n]+):\s*(.*)$/;

    return transcriptSection
      .split(/\r?\n/)
      .map((line) => line.trim())
      .map((line) => {
        const match = line.match(segmentPattern);

        if (!match) return null;

        return {
          start: Number(match[1]),
          end: Number(match[2]),
          speaker: match[3].trim(),
          text: match[4].trim(),
        };
      })
      .filter(Boolean);
  }

  // --------------------------------------------------
  // Load original transcript
  // --------------------------------------------------

  async function loadTranscript(id) {
    setTranscriptLoading(true);

    try {
      const response = await fetch(
        `${API_URL}/meetings/${id}/download/transcript`
      );

      if (!response.ok) {
        throw new Error("Could not load the original transcript.");
      }

      const transcriptText = await response.text();
      const segments = parseTranscript(transcriptText);

      if (segments.length === 0) {
        throw new Error(
          "No transcript segments could be extracted. Please check the transcript format."
        );
      }

      const speakers = [
        ...new Set(
          segments
            .map((segment) => segment.speaker)
            .filter((speaker) => speaker.toUpperCase() !== "UNKNOWN")
        ),
      ];

      const initialNames = {};
      const initialParagraphSpeakers = {};

      speakers.forEach((speaker) => {
        initialNames[speaker] = "";
      });

      segments.forEach((segment, index) => {
        if (segment.speaker.toUpperCase() !== "UNKNOWN") {
          initialParagraphSpeakers[index] = segment.speaker;
        }
      });

      setTranscript(segments);
      setSpeakerNames(initialNames);
      setParagraphSpeakers(initialParagraphSpeakers);
      setUnknownAssignments({});

      setUpdatedTranscriptReady(false);
      setUpdatedSummary("");
      setUpdatedSummaryReady(false);
    } catch (error) {
      setError(error.message || "Could not load transcript.");
    } finally {
      setTranscriptLoading(false);
    }
  }

  // --------------------------------------------------
  // Reset meeting states
  // --------------------------------------------------

  function resetMeeting() {
    setMeetingId(null);
    setStatus(null);
    setMessage("");
    setError("");

    setTranscript([]);
    setSpeakerNames({});
    setUnknownAssignments({});
    setParagraphSpeakers({});

    setUpdatedTranscriptReady(false);
    setUpdatedSummary("");
    setUpdatedSummaryReady(false);
  }

  // --------------------------------------------------
  // Upload and process meeting
  // --------------------------------------------------

  async function handleUpload() {
    if (!file) return;

    resetMeeting();
    setUploading(true);
    setMessage("Uploading audio...");

    try {
      const formData = new FormData();
      formData.append("file", file);

      // Upload first so the backend can return a meeting ID.
      const uploadResponse = await fetch(`${API_URL}/meetings/upload`, {
        method: "POST",
        body: formData,
      });

      if (!uploadResponse.ok) {
        const data = await uploadResponse.json().catch(() => ({}));
        throw new Error(data.detail || "Upload failed.");
      }

      const uploadData = await uploadResponse.json();
      const id = uploadData.meeting_id;

      if (!id) {
        throw new Error("The upload response did not contain a meeting_id.");
      }

      setMeetingId(id);
      setMessage("Upload successful. Starting processing...");

      // Start processing with the selected cosine similarity threshold.
      const processResponse = await fetch(`${API_URL}/meetings/${id}/process`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          cosine_threshold: cosineThreshold,
        }),
      });

      if (!processResponse.ok) {
        const data = await processResponse.json().catch(() => ({}));
        throw new Error(data.detail || "Could not start processing.");
      }

      setMessage("Meeting is being processed.");

      // Poll processing status until the backend completes or fails.
      while (true) {
        await sleep(2000);

        const statusResponse = await fetch(
          `${API_URL}/meetings/${id}/status`,
          { cache: "no-store" }
        );

        if (!statusResponse.ok) {
          throw new Error("Could not retrieve processing status.");
        }

        const statusData = await statusResponse.json();
        setStatus(statusData);

        if (statusData.status === "completed") {
          setMessage("Meeting processing completed successfully!");
          await loadTranscript(id);
          break;
        }

        if (statusData.status === "failed") {
          throw new Error(statusData.error || "Meeting processing failed.");
        }
      }
    } catch (error) {
      setError(error.message || "Something went wrong.");
      setMessage("");
    } finally {
      setUploading(false);
    }
  }

  // --------------------------------------------------
  // Download files
  // --------------------------------------------------

  function downloadFile(fileType) {
    if (!meetingId) return;

    window.open(
      `${API_URL}/meetings/${meetingId}/download/${fileType}`,
      "_blank"
    );
  }

  // --------------------------------------------------
  // Speaker updates
  // --------------------------------------------------

  function updateSpeakerName(speaker, name) {
    setSpeakerNames((previous) => ({
      ...previous,
      [speaker]: name,
    }));

    setUpdatedTranscriptReady(false);
    setUpdatedSummary("");
    setUpdatedSummaryReady(false);
  }

  function updateUnknownAssignment(index, speaker) {
    setUnknownAssignments((previous) => ({
      ...previous,
      [index]: speaker,
    }));

    setUpdatedTranscriptReady(false);
    setUpdatedSummary("");
    setUpdatedSummaryReady(false);
  }

  function updateParagraphSpeaker(index, speaker) {
    setParagraphSpeakers((previous) => ({
      ...previous,
      [index]: speaker,
    }));

    setUpdatedTranscriptReady(false);
    setUpdatedSummary("");
    setUpdatedSummaryReady(false);
  }

  // --------------------------------------------------
  // Generate updated transcript AND updated summary
  // --------------------------------------------------

  async function generateUpdatedTranscript() {
    if (!meetingId) return;

    setGeneratingUpdatedTranscript(true);
    setError("");
    setMessage("");

    setUpdatedTranscriptReady(false);
    setUpdatedSummary("");
    setUpdatedSummaryReady(false);

    try {
      const response = await fetch(
        `${API_URL}/meetings/${meetingId}/transcript/update`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            speaker_names: speakerNames,
            unknown_assignments: unknownAssignments,
            paragraph_speakers: paragraphSpeakers,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Could not generate updated transcript."
        );
      }

      // The backend should return the updated summary text.
      if (
        typeof data.updated_summary !== "string" ||
        !data.updated_summary.trim()
      ) {
        throw new Error(
          "The transcript was processed, but the backend did not return updated_summary. Check the transcript-update endpoint in main.py."
        );
      }

      setUpdatedTranscriptReady(true);
      setUpdatedSummary(data.updated_summary);
      setUpdatedSummaryReady(true);

      setMessage("Updated transcript and summary generated successfully!");
    } catch (error) {
      setError(
        error.message ||
          "Failed to generate updated transcript and summary."
      );
    } finally {
      setGeneratingUpdatedTranscript(false);
    }
  }

  // --------------------------------------------------
  // Derived values
  // --------------------------------------------------

  const speakerList = Object.keys(speakerNames);
  const completedSegments = transcript.length;

  const unknownCount = transcript.filter(
    (segment) => segment.speaker.toUpperCase() === "UNKNOWN"
  ).length;

  const detectedSpeakerCount = speakerList.length;
  const progressStages = Object.entries(status?.stages || {});

  return (
        <main className="min-h-screen bg-[#070b18] text-white">
      {/* NAVBAR */}
      <nav className="sticky top-0 z-30 border-b border-slate-800/80 bg-[#070b18]/90 px-5 py-4 backdrop-blur-xl sm:px-8 lg:px-12">
        <div className="mx-auto flex max-w-[1700px] items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-600 text-xl font-bold shadow-lg shadow-blue-900/30">
              M
            </div>

            <div>
              <h1 className="text-xl font-bold tracking-tight">
                Meet<span className="text-blue-400">Mind</span>
              </h1>

              <p className="text-[11px] text-slate-500">
                AI Meeting Assistant
              </p>
            </div>
          </div>

          <div className="hidden items-center gap-2 rounded-full border border-slate-700 bg-slate-900/80 px-4 py-2 text-xs text-slate-300 sm:flex">
            <span className="h-2 w-2 rounded-full bg-green-400" />
            Local meeting processing
          </div>
        </div>
      </nav>

      <div className="mx-auto max-w-[1700px] px-5 sm:px-8 lg:px-12">
        {/* HERO */}
        <section className="grid items-center gap-10 py-12 md:grid-cols-[1.2fr_0.8fr] lg:py-16">
          <div>
            <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-blue-500/30 bg-blue-500/10 px-4 py-2 text-sm text-blue-300">
              <span>✦</span>
              AI-powered meeting intelligence
            </div>

            <div className="mt-8 max-w-2xl rounded-2xl border border-slate-700 bg-slate-900/80 p-6 sm:p-8">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <label
                  htmlFor="cosine-threshold"
                  className="text-lg font-semibold text-white"
                >
                  Adjust the cosine similarity parameter
                </label>

                <span className="rounded-lg border border-blue-500/30 bg-blue-500/10 px-4 py-2 font-mono text-lg font-bold text-blue-400">
                  {cosineThreshold.toFixed(2)}
                </span>
              </div>

              <input
                id="cosine-threshold"
                type="range"
                min="0.6"
                max="0.8"
                step="0.01"
                value={cosineThreshold}
                onChange={(event) =>
                  setCosineThreshold(Number(event.target.value))
                }
                disabled={uploading}
                className="mt-7 w-full cursor-pointer accent-blue-500 disabled:cursor-not-allowed"
              />

              <div className="mt-2 flex justify-between text-xs text-slate-500">
                <span>0.60</span>
                <span>0.70</span>
                <span>0.80</span>
              </div>

              <div className="mt-6 rounded-xl border border-green-500/20 bg-green-500/5 p-4">
                <p className="text-sm font-semibold text-green-400">
                  Recommended value: 0.72
                </p>

                <p className="mt-1 text-sm leading-6 text-slate-400">
                  Higher values make speaker matching stricter,
                  while lower values make it more permissive.
                  The default recommended and tested threshold is 0.72.
                </p>

                <button
                  type="button"
                  onClick={() => setCosineThreshold(0.72)}
                  disabled={uploading}
                  className="mt-3 text-xs font-semibold text-blue-400 transition hover:text-blue-300 disabled:opacity-50"
                >
                  Reset to recommended value
                </button>
              </div>
            </div>

            <p className="mt-6 max-w-2xl text-base leading-8 text-slate-400 sm:text-lg">
              Turn your meeting recordings into structured transcripts,
              identify participants, and generate useful AI-powered
              meeting summaries.
            </p>

            <div className="mt-8 flex flex-wrap gap-3 text-sm text-slate-400">
              <span className="rounded-lg border border-slate-800 bg-slate-900 px-3 py-2">
                🎙️ Speech recognition
              </span>

              <span className="rounded-lg border border-slate-800 bg-slate-900 px-3 py-2">
                👥 Speaker identification
              </span>

              <span className="rounded-lg border border-slate-800 bg-slate-900 px-3 py-2">
                📄 PDF reports
              </span>
            </div>
          </div>

          {/* UPLOAD CARD */}
          <div className="rounded-2xl border border-slate-700/80 bg-slate-900/80 p-6 shadow-2xl shadow-black/20 sm:p-8">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-xl font-semibold">
                  Upload a meeting
                </h3>

                <p className="mt-1 text-sm text-slate-400">
                  Start by selecting an audio recording.
                </p>
              </div>

              <div className="hidden h-12 w-12 items-center justify-center rounded-xl bg-blue-500/10 text-2xl sm:flex">
                🎧
              </div>
            </div>

            <label className="mt-7 flex min-h-[190px] cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-700 bg-[#080e1e] px-5 py-8 text-center transition hover:border-blue-500 hover:bg-blue-500/5">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-blue-500/10 text-3xl">
                🎙️
              </div>

              <p className="mt-4 font-medium text-slate-200">
                Click to select an audio file
              </p>

              <p className="mt-2 text-xs text-slate-500">
                WAV, MP3, M4A, FLAC or OGG
              </p>

              <input
                type="file"
                accept=".wav,.mp3,.m4a,.flac,.ogg,audio/*"
                className="hidden"
                disabled={uploading}
                onChange={(event) => {
                  setFile(event.target.files?.[0] || null);
                  resetMeeting();
                }}
              />
            </label>

            {file && (
              <div className="mt-4 flex items-center gap-3 rounded-xl border border-slate-700 bg-slate-950 p-4">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-blue-500/10 text-xl">
                  🎵
                </div>

                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium text-slate-200">
                    {file.name}
                  </p>

                  <p className="mt-1 text-xs text-slate-500">
                    {(file.size / (1024 * 1024)).toFixed(2)} MB
                  </p>
                </div>

                <button
                  type="button"
                  onClick={() => {
                    setFile(null);
                    resetMeeting();
                  }}
                  disabled={uploading}
                  className="rounded-lg px-3 py-2 text-xs text-slate-400 hover:bg-slate-800 hover:text-white"
                >
                  Remove
                </button>
              </div>
            )}

            <button
              onClick={handleUpload}
              disabled={!file || uploading}
              className="mt-5 w-full rounded-xl bg-blue-600 px-6 py-4 font-semibold shadow-lg shadow-blue-900/20 transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {uploading
                ? "Processing your meeting..."
                : "Upload and process meeting →"}
            </button>

            {message && (
              <p className="mt-4 text-center text-sm text-blue-300">
                {message}
              </p>
            )}

            {error && (
              <p className="mt-4 whitespace-pre-wrap rounded-lg border border-red-500/30 bg-red-950/30 p-3 text-sm text-red-400">
                {error}
              </p>
            )}

            {meetingId && (
              <div className="mt-4 rounded-lg border border-slate-800 bg-slate-950 p-3">
                <p className="text-[11px] uppercase tracking-wider text-slate-500">
                  Meeting ID
                </p>

                <p className="mt-1 break-all font-mono text-xs text-slate-300">
                  {meetingId}
                </p>
              </div>
            )}
          </div>
        </section>

        {/* PROCESSING PROGRESS */}
        {status && status.status !== "completed" && (
          <section className="mb-10 rounded-2xl border border-slate-700 bg-slate-900/80 p-6 sm:p-8">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <h3 className="text-xl font-semibold">
                  Processing your meeting
                </h3>

                <p className="mt-1 text-sm text-slate-400">
                  MeetMind is working through your recording.
                </p>
              </div>

              <span className="rounded-full border border-blue-500/30 bg-blue-500/10 px-4 py-2 text-sm text-blue-300">
                {status.status}
              </span>
            </div>

            <p className="mt-6 text-sm text-blue-300">
              Current stage: {status.current_stage || "Waiting"}
            </p>

            <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {progressStages.map(([stage, details]) => (
                <div
                  key={stage}
                  className="flex items-center justify-between gap-3 rounded-xl border border-slate-800 bg-slate-950 p-4"
                >
                  <span className="text-sm text-slate-300">
                    {stage}
                  </span>

                  <span
                    className={
                      details.status === "completed"
                        ? "text-xs font-medium text-green-400"
                        : details.status === "failed"
                          ? "text-xs font-medium text-red-400"
                          : "text-xs font-medium text-blue-400"
                    }
                  >
                    {details.status}
                  </span>
                </div>
              ))}
            </div>
          </section>
        )}
                {/* COMPLETED RESULTS */}
        {status?.status === "completed" && (
          <section className="pb-20">
            {/* RESULTS HEADING */}
            <div className="mb-8 flex flex-wrap items-end justify-between gap-4 border-b border-slate-800 pb-6">
              <div>
                <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-green-500/30 bg-green-500/10 px-3 py-1.5 text-xs font-medium text-green-400">
                  <span className="h-2 w-2 rounded-full bg-green-400" />
                  Processing completed
                </div>

                <h2 className="text-3xl font-bold tracking-tight sm:text-4xl">
                  Meeting results
                </h2>

                <p className="mt-2 text-sm text-slate-400">
                  Review your transcript, identify speakers and
                  generate your final report.
                </p>
              </div>

              <div className="text-sm text-slate-500">
                {completedSegments} transcript segments
              </div>
            </div>

            {/* SUMMARY STATISTICS */}
            <div className="mb-8 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              <StatCard
                label="Transcript segments"
                value={completedSegments}
                icon="📝"
              />

              <StatCard
                label="Detected speakers"
                value={detectedSpeakerCount}
                icon="👥"
              />

              <StatCard
                label="Unknown segments"
                value={unknownCount}
                icon="⚠️"
                danger={unknownCount > 0}
              />

              <StatCard
                label="Meeting status"
                value="Completed"
                icon="✓"
                success
              />
            </div>

            {/* ORIGINAL DOWNLOADS */}
            <div className="mb-8 rounded-2xl border border-green-700/40 bg-gradient-to-r from-green-950/40 to-slate-900 p-6 sm:p-8">
              <div className="flex flex-col justify-between gap-6 lg:flex-row lg:items-center">
                <div>
                  <h3 className="text-xl font-semibold text-green-400">
                    Original meeting reports
                  </h3>

                  <p className="mt-2 text-sm text-slate-400">
                    These files contain the original transcript and
                    original AI-generated summary.
                  </p>
                </div>

                <div className="grid gap-3 sm:grid-cols-2 lg:min-w-[460px]">
                  <button
                    onClick={() => downloadFile("transcript_pdf")}
                    className="rounded-xl bg-blue-600 px-5 py-4 text-sm font-semibold transition hover:bg-blue-500"
                  >
                    ↓ Download Original Transcript PDF
                  </button>

                  <button
                    onClick={() => downloadFile("summary_pdf")}
                    className="rounded-xl bg-slate-700 px-5 py-4 text-sm font-semibold transition hover:bg-slate-600"
                  >
                    ↓ Download Original Summary PDF
                  </button>
                </div>
              </div>
            </div>

            {/* FULL-WIDTH TRANSCRIPT */}
            <div className="mb-8 overflow-hidden rounded-2xl border border-slate-700/80 bg-slate-900/80 shadow-xl shadow-black/10">
              <div className="border-b border-slate-800 px-5 py-6 sm:px-8">
                <div className="flex flex-wrap items-center justify-between gap-4">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-500/10 text-xl">
                      📝
                    </div>

                    <div>
                      <h3 className="text-xl font-semibold sm:text-2xl">
                        Meeting Transcript
                      </h3>

                      <p className="mt-1 text-xs text-slate-500">
                        Original transcript with detected speaker labels
                      </p>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-2 text-xs">
                    <span className="rounded-full border border-blue-500/30 bg-blue-500/10 px-3 py-1.5 text-blue-300">
                      {detectedSpeakerCount} speakers
                    </span>

                    {unknownCount > 0 && (
                      <span className="rounded-full border border-red-500/30 bg-red-500/10 px-3 py-1.5 text-red-400">
                        {unknownCount} unknown
                      </span>
                    )}
                  </div>
                </div>

                <p className="mt-5 max-w-4xl text-sm leading-6 text-slate-400">
                  Enter participant names, change individual paragraph
                  identities using Modify, and assign UNKNOWN segments
                  to detected speakers. These changes are applied only
                  when generating the updated reports.
                </p>

                <div className="mt-5 flex flex-wrap gap-5 text-xs text-slate-400">
                  <div className="flex items-center gap-2">
                    <span className="h-2.5 w-2.5 rounded-full bg-blue-400" />
                    Identified speaker
                  </div>

                  <div className="flex items-center gap-2">
                    <span className="h-2.5 w-2.5 rounded-full bg-red-400" />
                    Unknown speaker
                  </div>
                </div>
              </div>

              {/* TRANSCRIPT SEGMENTS */}
              <div className="p-4 sm:p-6 lg:p-8">
                {transcriptLoading ? (
                  <div className="py-16 text-center text-blue-400">
                    Loading transcript...
                  </div>
                ) : transcript.length === 0 ? (
                  <div className="py-16 text-center text-slate-400">
                    No transcript segments available.
                  </div>
                ) : (
                  <div className="space-y-4">
                    {transcript.map((segment, index) => {
                      const isUnknown =
                        segment.speaker.trim().toUpperCase() === "UNKNOWN";

                      const currentSpeaker =
                        paragraphSpeakers[index] || segment.speaker;

                      const displayedSpeaker =
                        speakerNames[currentSpeaker]?.trim() ||
                        currentSpeaker;

                      return (
                        <article
                          key={index}
                          className={
                            isUnknown
                              ? "rounded-xl border border-red-500/50 bg-red-950/20 p-5 transition hover:border-red-400/70 sm:p-6"
                              : "rounded-xl border border-slate-700/80 bg-[#080e1e] p-5 transition hover:border-slate-600 sm:p-6"
                          }
                        >
                          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                            <div className="flex flex-wrap items-center gap-3">
                              <span className="rounded-md border border-slate-700 bg-slate-900 px-3 py-1.5 font-mono text-xs text-slate-400">
                                {segment.start.toFixed(2)}s –{" "}
                                {segment.end.toFixed(2)}s
                              </span>

                              <span className="text-xs text-slate-600">
                                Segment {index + 1}
                              </span>
                            </div>

                            <div className="flex flex-wrap items-center gap-3">
                              <span
                                className={
                                  isUnknown
                                    ? "text-sm font-bold tracking-wide text-red-400"
                                    : "text-sm font-bold tracking-wide text-blue-400"
                                }
                              >
                                {isUnknown ? "UNKNOWN" : displayedSpeaker}
                              </span>

                              {/* INDIVIDUAL PARAGRAPH MODIFY */}
                              {!isUnknown && (
                                <div className="flex items-center gap-2">
                                  <label
                                    htmlFor={`speaker-${index}`}
                                    className="text-xs text-slate-500"
                                  >
                                    Modify
                                  </label>

                                  <select
                                    id={`speaker-${index}`}
                                    value={currentSpeaker}
                                    onChange={(event) =>
                                      updateParagraphSpeaker(
                                        index,
                                        event.target.value
                                      )
                                    }
                                    className="min-w-[170px] max-w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2.5 text-sm text-white outline-none transition focus:border-blue-500"
                                  >
                                    {speakerList.map((speaker) => (
                                      <option key={speaker} value={speaker}>
                                        {speaker}
                                        {speakerNames[speaker]
                                          ? ` — ${speakerNames[speaker]}`
                                          : ""}
                                      </option>
                                    ))}
                                  </select>
                                </div>
                              )}
                            </div>
                          </div>

                          {/* TRANSCRIPT TEXT */}
                          <p className="mt-5 whitespace-pre-wrap break-words text-[15px] leading-8 text-slate-200 sm:text-base">
                            {segment.text}
                          </p>

                          {/* UNKNOWN ASSIGNMENT */}
                          {isUnknown && (
                            <div className="mt-5 border-t border-red-500/20 pt-4">
                              <label
                                htmlFor={`unknown-${index}`}
                                className="mb-2 block text-sm font-medium text-red-300"
                              >
                                Assign this segment to a speaker
                              </label>

                              <select
                                id={`unknown-${index}`}
                                value={unknownAssignments[index] || ""}
                                onChange={(event) =>
                                  updateUnknownAssignment(
                                    index,
                                    event.target.value
                                  )
                                }
                                className="w-full rounded-lg border border-red-500/40 bg-[#0b1020] p-3 text-sm text-white outline-none transition focus:border-red-400 sm:max-w-md"
                              >
                                <option value="">
                                  Select a speaker
                                </option>

                                {speakerList.map((speaker) => (
                                  <option key={speaker} value={speaker}>
                                    {speaker}
                                    {speakerNames[speaker]
                                      ? ` — ${speakerNames[speaker]}`
                                      : ""}
                                  </option>
                                ))}
                              </select>

                              {unknownAssignments[index] && (
                                <p className="mt-2 text-xs text-green-400">
                                  Assigned to{" "}
                                  {speakerNames[
                                    unknownAssignments[index]
                                  ]?.trim() || unknownAssignments[index]}
                                </p>
                              )}
                            </div>
                          )}
                        </article>
                      );
                    })}
                  </div>
                )}
              </div>

              <div className="border-t border-slate-800 bg-slate-950/50 px-5 py-4 sm:px-8">
                <p className="text-xs text-slate-500">
                  Showing {transcript.length} transcript segments.
                  Original transcript remains unchanged.
                </p>
              </div>
            </div>
                        {/* SPEAKER IDENTIFICATION */}
            <div className="mb-8 rounded-2xl border border-slate-700/80 bg-slate-900/80 p-5 sm:p-8">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-purple-500/10 text-xl">
                      👥
                    </div>

                    <h3 className="text-xl font-semibold sm:text-2xl">
                      Identify Participants
                    </h3>
                  </div>

                  <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">
                    Enter the actual names of detected speakers.
                    Their names will be used in the updated transcript
                    and updated meeting summary.
                  </p>
                </div>

                <span className="rounded-full border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-400">
                  {detectedSpeakerCount} detected
                </span>
              </div>

              {speakerList.length === 0 ? (
                <p className="mt-6 text-sm text-slate-400">
                  No identified speakers available.
                </p>
              ) : (
                <div className="mt-7 grid gap-5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                  {speakerList.map((speaker, index) => (
                    <div
                      key={speaker}
                      className="rounded-xl border border-slate-700 bg-[#080e1e] p-5"
                    >
                      <div className="mb-4 flex items-center gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-500/10 font-semibold text-blue-300">
                          {index + 1}
                        </div>

                        <div>
                          <p className="text-sm font-semibold text-blue-300">
                            {speaker}
                          </p>

                          <p className="text-xs text-slate-500">
                            Detected speaker
                          </p>
                        </div>
                      </div>

                      <label
                        htmlFor={`name-${speaker}`}
                        className="mb-2 block text-xs text-slate-400"
                      >
                        Participant name
                      </label>

                      <input
                        id={`name-${speaker}`}
                        type="text"
                        placeholder="Enter participant name"
                        value={speakerNames[speaker] || ""}
                        onChange={(event) =>
                          updateSpeakerName(speaker, event.target.value)
                        }
                        className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-3 text-sm text-white outline-none transition placeholder:text-slate-600 focus:border-blue-500"
                      />
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* GENERATE UPDATED TRANSCRIPT AND SUMMARY */}
            <div className="rounded-2xl border border-blue-500/30 bg-gradient-to-r from-blue-950/40 to-slate-900 p-6 sm:p-8">
              <div className="flex flex-col justify-between gap-6 lg:flex-row lg:items-center">
                <div className="max-w-2xl">
                  <div className="flex items-center gap-3">
                    <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-blue-500/10 text-xl">
                      📄
                    </div>

                    <h3 className="text-xl font-semibold">
                      Generate Updated Reports
                    </h3>
                  </div>

                  <p className="mt-3 text-sm leading-7 text-slate-400">
                    Apply participant names, individual paragraph
                    modifications and UNKNOWN assignments. MeetMind
                    will generate a corrected transcript PDF and
                    regenerate the AI summary using that corrected
                    transcript.
                  </p>

                  <p className="mt-2 text-xs text-slate-500">
                    Original transcript and original summary files
                    will not be overwritten.
                  </p>
                </div>

                <div className="w-full lg:max-w-sm">
                  <button
                    onClick={generateUpdatedTranscript}
                    disabled={
                      generatingUpdatedTranscript ||
                      transcriptLoading ||
                      transcript.length === 0
                    }
                    className="w-full rounded-xl bg-blue-600 px-6 py-4 font-semibold shadow-lg shadow-blue-900/20 transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {generatingUpdatedTranscript
                      ? "Generating updated reports..."
                      : "Generate Updated Transcript & Summary →"}
                  </button>

                  {generatingUpdatedTranscript && (
                    <p className="mt-3 text-center text-xs text-blue-300">
                      Applying your changes and generating the new summary.
                      This may take a few minutes.
                    </p>
                  )}

                  {updatedTranscriptReady && (
                    <div className="mt-4 rounded-xl border border-green-600/40 bg-green-950/30 p-4">
                      <p className="font-semibold text-green-400">
                        Updated reports are ready!
                      </p>

                      <p className="mt-1 text-xs leading-5 text-slate-400">
                        Your original files remain unchanged.
                      </p>

                      <button
                        onClick={() =>
                          downloadFile("updated_transcript_pdf")
                        }
                        className="mt-4 w-full rounded-lg bg-green-600 px-4 py-3 text-sm font-semibold transition hover:bg-green-500"
                      >
                        ↓ Download Updated Transcript PDF
                      </button>

                      <button
                        onClick={() =>
                          downloadFile("updated_summary_pdf")
                        }
                        className="mt-3 w-full rounded-lg bg-blue-600 px-4 py-3 text-sm font-semibold transition hover:bg-blue-500"
                      >
                        ↓ Download Updated Summary PDF
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* UPDATED SUMMARY PREVIEW */}
            {updatedSummaryReady && (
              <div className="mt-8 overflow-hidden rounded-2xl border border-purple-500/30 bg-slate-900/80 shadow-xl">
                <div className="border-b border-slate-800 px-5 py-6 sm:px-8">
                  <div className="flex flex-wrap items-center justify-between gap-4">
                    <div className="flex items-center gap-3">
                      <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-purple-500/10 text-xl">
                        ✨
                      </div>

                      <div>
                        <h3 className="text-xl font-semibold sm:text-2xl">
                          Updated Meeting Summary
                        </h3>

                        <p className="mt-1 text-xs text-slate-500">
                          Generated from your corrected transcript
                        </p>
                      </div>
                    </div>

                    <span className="rounded-full border border-green-500/30 bg-green-500/10 px-3 py-2 text-xs text-green-400">
                      Generated
                    </span>
                  </div>
                </div>

                <div className="p-5 sm:p-8">
                  <div className="whitespace-pre-wrap break-words text-sm leading-8 text-slate-300 sm:text-base">
                    {updatedSummary}
                  </div>
                </div>

                <div className="flex flex-col justify-between gap-4 border-t border-slate-800 bg-slate-950/50 px-5 py-5 sm:flex-row sm:items-center sm:px-8">
                  <p className="text-xs text-slate-500">
                    This summary reflects your corrected speaker identities.
                  </p>

                  <button
                    onClick={() =>
                      downloadFile("updated_summary_pdf")
                    }
                    className="rounded-xl bg-purple-600 px-5 py-3 text-sm font-semibold transition hover:bg-purple-500"
                  >
                    ↓ Download Updated Summary PDF
                  </button>
                </div>
              </div>
            )}
          </section>
        )}

        {/* FEATURES */}
        {!status?.status && (
          <section className="grid gap-5 pb-16 md:grid-cols-3">
            <Feature
              icon="📝"
              title="Smart Transcription"
              description="Convert meeting recordings into readable transcripts with timestamps."
            />

            <Feature
              icon="👥"
              title="Speaker Identification"
              description="Detect different speakers and correct their identities."
            />

            <Feature
              icon="📄"
              title="Meeting Reports"
              description="Generate downloadable transcript and AI summary PDFs."
            />
          </section>
        )}
      </div>

      {/* FOOTER */}
      <footer className="border-t border-slate-800 px-5 py-6 sm:px-8 lg:px-12">
        <div className="mx-auto flex max-w-[1700px] flex-col items-center justify-between gap-2 text-xs text-slate-500 sm:flex-row">
          <p>
            Meet<span className="text-blue-400">Mind</span> · AI Meeting Assistant
          </p>

          <p>
            Transforming conversations into useful insights.
          </p>
        </div>
      </footer>
    </main>
  );
}

// --------------------------------------------------
// STATISTIC CARD
// --------------------------------------------------

function StatCard({ label, value, icon, danger, success }) {
  return (
    <div className="rounded-xl border border-slate-700/80 bg-slate-900/80 p-5">
      <div className="flex items-center justify-between">
        <p className="text-sm text-slate-400">
          {label}
        </p>

        <span className="text-xl">
          {icon}
        </span>
      </div>

      <p
        className={
          danger
            ? "mt-4 text-3xl font-bold text-red-400"
            : success
              ? "mt-4 text-2xl font-bold text-green-400"
              : "mt-4 text-3xl font-bold text-white"
        }
      >
        {value}
      </p>
    </div>
  );
}

// --------------------------------------------------
// FEATURE CARD
// --------------------------------------------------

function Feature({ icon, title, description }) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-6 transition hover:border-slate-700">
      <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-blue-500/10 text-2xl">
        {icon}
      </div>

      <h3 className="mt-5 text-lg font-semibold">
        {title}
      </h3>

      <p className="mt-2 text-sm leading-7 text-slate-400">
        {description}
      </p>
    </div>
  );
}