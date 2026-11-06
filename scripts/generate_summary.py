
import argparse
import json
import re
import urllib.request
import urllib.error
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    HRFlowable,
    Table,
    TableStyle,
    KeepTogether,
)


# ==================================================
# CONFIGURATION
# ==================================================

MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MAX_CHUNK_CHARS = 8000

NAVY = colors.HexColor("#17365D")
BLUE = colors.HexColor("#2874A6")
LIGHT_BLUE = colors.HexColor("#EAF1F8")
LIGHT_GREY = colors.HexColor("#F3F5F7")
DARK_GREY = colors.HexColor("#444444")


# ==================================================
# OLLAMA
# ==================================================

def ask_ollama(prompt):

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 8192
        }
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

        return result["response"].strip()

    except urllib.error.URLError as error:
        raise RuntimeError(
            "Could not connect to Ollama. "
            "Please make sure Ollama is running.\n"
            f"Details: {error}"
        )


# ==================================================
# TRANSCRIPT EXTRACTION
# ==================================================

def extract_transcript_text(content):
    """
    Extract only the TRANSCRIPT section.

    If no TRANSCRIPT heading exists, treat the complete
    supplied content as the transcript.
    """

    match = re.search(
        r"TRANSCRIPT\s*-+\s*(.*?)(?=\n\s*OVERLAPPING SPEECH|\Z)",
        content,
        flags=re.DOTALL | re.IGNORECASE
    )

    transcript = (
        match.group(1).strip()
        if match
        else content.strip()
    )

    if not transcript:
        raise ValueError("Transcript is empty.")

    return transcript


def load_transcript(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Transcript not found: {path}"
        )

    content = path.read_text(encoding="utf-8")

    transcript = extract_transcript_text(content)

    return content, transcript


# ==================================================
# DIARIZATION METADATA
# Used only for the ORIGINAL summary workflow.
# ==================================================

def load_diarization(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Diarization file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    turns = data.get("turns", [])

    speakers = data.get("speakers", [])

    if not speakers:
        speakers = sorted({
            turn.get("speaker")
            for turn in turns
            if turn.get("speaker")
            and turn.get("speaker") != "UNKNOWN"
        })

    duration = max(
        (
            float(turn.get("end", 0))
            for turn in turns
        ),
        default=0
    )

    unknown_turns = sum(
        1 for turn in turns
        if turn.get("speaker") == "UNKNOWN"
    )

    return {
        "speakers": speakers,
        "speaker_count": data.get(
            "number_of_speakers",
            len(speakers)
        ),
        "turn_count": data.get(
            "number_of_turns",
            len(turns)
        ),
        "duration": duration,
        "unknown_turns": unknown_turns
    }


# ==================================================
# METADATA FROM CORRECTED TRANSCRIPT
# Used for the UPDATED summary workflow.
# ==================================================

def build_metadata_from_transcript(transcript):

    """
    Derive speaker labels, number of turns, duration
    and UNKNOWN turns from the corrected transcript.

    This deliberately does not read the original
    diarization JSON.
    """

    segment_pattern = re.compile(
        r"^\[(\d+(?:\.\d+)?)s\s*-\s*"
        r"(\d+(?:\.\d+)?)s\]\s*(.+?):\s*(.*)$"
    )

    speakers = set()
    turn_count = 0
    unknown_turns = 0
    duration = 0.0

    for line in transcript.splitlines():

        match = segment_pattern.match(line.strip())

        if not match:
            continue

        start = float(match.group(1))
        end = float(match.group(2))
        speaker = match.group(3).strip()

        turn_count += 1
        duration = max(duration, end)

        if speaker.upper() == "UNKNOWN":
            unknown_turns += 1
        else:
            speakers.add(speaker)

    speaker_list = sorted(speakers)

    return {
        "speakers": speaker_list,
        "speaker_count": len(speaker_list),
        "turn_count": turn_count,
        "duration": duration,
        "unknown_turns": unknown_turns
    }


# ==================================================
# TRANSCRIPT CHUNKING
# ==================================================

def split_transcript(transcript):

    lines = transcript.splitlines()

    chunks = []
    current_chunk = ""

    for line in lines:

        if len(current_chunk) + len(line) > MAX_CHUNK_CHARS:

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            current_chunk = ""

        if len(line) > MAX_CHUNK_CHARS:

            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""

            for start in range(0, len(line), MAX_CHUNK_CHARS):
                chunks.append(
                    line[start:start + MAX_CHUNK_CHARS]
                )

        else:
            current_chunk += line + "\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ==================================================
# CHUNK SUMMARIZATION
# ==================================================

def summarize_chunks(chunks):

    summaries = []

    total = len(chunks)

    for index, chunk in enumerate(chunks, start=1):

        print(
            f"\nProcessing transcript chunk {index}/{total}..."
        )

        prompt = f"""
You are a professional meeting analysis assistant.

Analyze the following portion of a meeting transcript.

Your task is to extract useful and accurate meeting information.

IMPORTANT RULES:

1. Do not invent information.
2. Preserve the actual meaning of what speakers said.
3. Identify the main subjects being discussed.
4. Capture important explanations, examples and viewpoints.
5. Identify agreements only when speakers clearly agree.
6. Distinguish opinions, suggestions and confirmed decisions.
7. Identify action items only when an actual task is assigned.
8. Never invent an owner or deadline.
9. Use the speaker labels exactly as they appear in the transcript.
10. Ignore transcription errors where the intended meaning
    cannot be determined confidently.
11. Do not treat casual statements as formal decisions.
12. Do not replace participant names with anonymous labels.
13. If the transcript says UNKNOWN, do not guess the identity.

Analyze the transcript under these headings:

MAIN TOPICS:

Explain the important subjects discussed.

IMPORTANT DISCUSSION:

Describe the key arguments, explanations, examples
and different viewpoints.

SPEAKER CONTRIBUTIONS:

Explain what each identifiable speaker contributed.
Preserve the exact speaker labels.

AGREEMENTS:

Mention any clearly established agreements.

DECISIONS:

Mention confirmed decisions only.

ACTION ITEMS:

Mention explicitly assigned tasks, owners and deadlines.

OPEN QUESTIONS:

Mention questions or matters left unresolved.

If a category has no relevant information, say so.

TRANSCRIPT:

{chunk}
"""

        summary = ask_ollama(prompt)

        summaries.append(summary)

        print(f"Chunk {index} completed.")

    return summaries


# ==================================================
# FINAL SUMMARY GENERATION
# ==================================================

def generate_final_summary(chunk_summaries, metadata):

    combined_notes = "\n\n".join(
        f"TRANSCRIPT PART {index}:\n{summary}"
        for index, summary in enumerate(
            chunk_summaries,
            start=1
        )
    )

    speaker_labels = ", ".join(metadata["speakers"])

    prompt = f"""
You are a professional meeting documentation assistant.

Create a detailed, well-organized and professional
meeting report using the supplied transcript analysis.

MEETING INFORMATION:

Detected speakers: {metadata["speaker_count"]}
Speaker labels: {speaker_labels}
Speaker turns: {metadata["turn_count"]}
Unknown turns: {metadata["unknown_turns"]}

MEETING ANALYSIS:

{combined_notes}

YOUR RESPONSIBILITIES:

Create a comprehensive report that accurately represents
the discussion.

CRITICAL ACCURACY RULES:

1. Do not introduce facts that are not supported by the notes.
2. Do not invent participant names or identities.
3. Preserve speaker names and labels exactly as provided.
4. If a participant is named in the transcript, use that name.
5. Do not replace participant names with SPEAKER_00 or
   other anonymous labels.
6. If a speaker is labelled UNKNOWN, keep that uncertainty.
7. Do not invent meeting dates, locations or titles.
8. Do not invent decisions, agreements or deadlines.
9. A suggestion is not automatically a decision.
10. An opinion is not automatically an agreement.
11. Do not invent action items to fill a section.
12. Combine repeated discussion points instead of repeating them.
13. Preserve important examples and explanations.
14. Use professional, neutral language.
15. Do not make the report unnecessarily verbose.
16. If information is unavailable, explicitly state that it
    was not identified in the available transcript.

Use the following exact report structure.

## 1. Executive Summary

Provide a meaningful overview of the meeting.

Explain the central subject, the most important discussion
points, the different viewpoints and the overall outcome.

Do not simply repeat the meeting title.

## 2. Topics Discussed

Identify the major topics.

For each topic:

- Give it a descriptive title.
- Explain what was discussed.
- Include important arguments and examples.
- Mention the relevant speakers where identifiable.
- Preserve the exact participant names from the transcript.

## 3. Detailed Discussion

Present a coherent account of how the discussion developed.

Explain the important viewpoints, examples, supporting
arguments and contrasting opinions.

Avoid repeating the same information from the previous section.

## 4. Speaker-wise Contributions

For each detected speaker:

- Summarize their main contributions.
- Identify their viewpoints and examples.
- Explain any important responses to other speakers.
- Use the exact speaker name or label from the transcript.

Do not invent identities or attribute statements to the wrong speaker.

## 5. Key Takeaways

List the most important insights from the meeting.

These should reflect the actual discussion rather than
generic statements.

## 6. Decisions and Agreements

List only explicitly confirmed decisions or agreements.

If none were identified, write:

"No confirmed decisions or agreements were identified
in the available transcript."

## 7. Action Items

List tasks that were explicitly assigned.

Include the responsible speaker and deadline only when
the transcript supports that information.

If no tasks were assigned, write:

"No explicit action items were identified in the
available transcript."

## 8. Open Questions and Follow-ups

List questions, unresolved issues or follow-ups that
were actually raised.

Do not invent future work.

If none were identified, say so.

## 9. Conclusion

Provide a concise closing summary of the meeting's
overall outcome and significance.

FORMATTING:

- Use clear, professional language.
- Use short paragraphs.
- Use bullet points where appropriate.
- Use **bold** for important labels and terms.
- Do not include Markdown tables.
- Do not add greetings or conversational commentary.
- Do not include information about the AI model.

"""

    print("\nGenerating final professional report...")

    return ask_ollama(prompt)


# ==================================================
# COMMON SUMMARY GENERATION PIPELINE
# ==================================================

def generate_summary_content(transcript, metadata):

    """
    Shared summarization pipeline.

    Both original and updated summaries use this function.
    """

    if not transcript.strip():
        raise ValueError("Cannot summarize an empty transcript.")

    chunks = split_transcript(transcript)

    if not chunks:
        raise ValueError("No transcript chunks were generated.")

    print(f"\nTranscript chunks: {len(chunks)}")

    chunk_summaries = summarize_chunks(chunks)

    final_summary = generate_final_summary(
        chunk_summaries,
        metadata
    )

    if not final_summary.strip():
        raise RuntimeError("The model returned an empty summary.")

    return final_summary


# ==================================================
# UPDATED SUMMARY ENTRY POINT
# Called by main.py after transcript corrections.
# ==================================================

def generate_summary_from_transcript(
    transcript_text,
    metadata,
    meeting_name,
    output_txt,
    output_pdf
):

    """
    Generate an updated summary from the corrected transcript.

    Parameters:
        transcript_text:
            Corrected transcript text received from main.py.

        metadata:
            Metadata derived from the corrected transcript.

        meeting_name:
            Meeting ID or recording name.

        output_txt:
            Destination for the updated summary text.

        output_pdf:
            Destination for the updated summary PDF.

    Returns:
        The generated summary as a string.

    IMPORTANT:
        This function does not read the original transcript,
        original diarization JSON, or original summary files.
    """

    print("\n" + "=" * 55)
    print("       UPDATED MEETING SUMMARY GENERATOR")
    print("=" * 55)

    # Extract only the corrected transcript section.
    transcript = extract_transcript_text(transcript_text)

    # Generate summary using the corrected transcript.
    final_summary = generate_summary_content(
        transcript,
        metadata
    )

    # Convert output paths to Path objects.
    output_txt = Path(output_txt)
    output_pdf = Path(output_pdf)

    # Save updated summary text.
    output_txt.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_txt.write_text(
        final_summary,
        encoding="utf-8"
    )

    # Generate updated summary PDF using the same styling.
    create_pdf(
        final_summary,
        metadata,
        output_pdf,
        meeting_name
    )

    print("\n" + "=" * 55)
    print("       UPDATED SUMMARY COMPLETED")
    print("=" * 55)

    print(f"\nUpdated PDF: {output_pdf}")
    print(f"Updated Text: {output_txt}")

    return final_summary


# ==================================================
# TEXT FORMATTING FOR PDF
# ==================================================

def format_inline_markdown(text):

    # Escape XML/HTML characters first.
    text = escape(text)

    # Convert Markdown bold to ReportLab bold.
    text = re.sub(
        r"\*\*(.+?)\*\*",
        r"<b>\1</b>",
        text
    )

    # Convert Markdown italic to ReportLab italic.
    text = re.sub(
        r"(?<!\*)\*([^*]+)\*(?!\*)",
        r"<i>\1</i>",
        text
    )

    return text


def format_duration(seconds):

    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)

    return f"{minutes} min {remaining_seconds} sec"


# ==================================================
# PDF STYLES
# ==================================================

def create_styles():

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=23,
        leading=29,
        alignment=TA_CENTER,
        textColor=NAVY,
        spaceAfter=8
    ))

    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        alignment=TA_CENTER,
        textColor=DARK_GREY,
        spaceAfter=18
    ))

    styles.add(ParagraphStyle(
        name="SectionHeadingCustom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=18,
        textColor=NAVY,
        spaceBefore=15,
        spaceAfter=7,
        keepWithNext=True
    ))

    styles.add(ParagraphStyle(
        name="SubheadingCustom",
        parent=styles["Heading3"],
        fontSize=10.5,
        leading=14,
        textColor=BLUE,
        spaceBefore=8,
        spaceAfter=5,
        keepWithNext=True
    ))

    styles.add(ParagraphStyle(
        name="BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#252525"),
        spaceAfter=7
    ))

    styles.add(ParagraphStyle(
        name="BulletCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.3,
        leading=14,
        leftIndent=16,
        firstLineIndent=-10,
        textColor=colors.HexColor("#252525"),
        spaceAfter=6
    ))

    styles.add(ParagraphStyle(
        name="MetadataCustom",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=DARK_GREY
    ))

    styles.add(ParagraphStyle(
        name="SmallNote",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#666666")
    ))

    return styles


# ==================================================
# PDF CONTENT
# ==================================================

def add_metadata_table(story, metadata, meeting_name, styles):

    speaker_names = ", ".join(metadata["speakers"])

    rows = [
        [
            Paragraph("<b>Meeting</b>", styles["MetadataCustom"]),
            Paragraph(escape(meeting_name), styles["MetadataCustom"]),
            Paragraph("<b>Duration</b>", styles["MetadataCustom"]),
            Paragraph(
                format_duration(metadata["duration"]),
                styles["MetadataCustom"]
            )
        ],
        [
            Paragraph("<b>Detected speakers</b>", styles["MetadataCustom"]),
            Paragraph(
                str(metadata["speaker_count"]),
                styles["MetadataCustom"]
            ),
            Paragraph("<b>Speaker turns</b>", styles["MetadataCustom"]),
            Paragraph(
                str(metadata["turn_count"]),
                styles["MetadataCustom"]
            )
        ],
        [
            Paragraph("<b>Speaker labels</b>", styles["MetadataCustom"]),
            Paragraph(
                escape(speaker_names),
                styles["MetadataCustom"]
            ),
            Paragraph("<b>Unknown turns</b>", styles["MetadataCustom"]),
            Paragraph(
                str(metadata["unknown_turns"]),
                styles["MetadataCustom"]
            )
        ]
    ]

    table = Table(
        rows,
        colWidths=[
            1.05 * inch,
            1.75 * inch,
            1.0 * inch,
            2.0 * inch
        ]
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BLUE),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#CFDCE8")),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D6E0E8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))

    story.append(table)


def add_report_content(story, summary, styles):

    lines = summary.splitlines()

    for line in lines:

        line = line.strip()

        if not line:
            story.append(Spacer(1, 3))
            continue

        # Main section headings.
        if line.startswith("## "):

            heading = escape(line[3:].strip())

            story.append(Spacer(1, 5))

            story.append(KeepTogether([
                Paragraph(
                    heading,
                    styles["SectionHeadingCustom"]
                ),
                HRFlowable(
                    width="100%",
                    thickness=0.6,
                    color=colors.HexColor("#D4DCE4"),
                    spaceAfter=8
                )
            ]))

        # Smaller headings.
        elif line.startswith("### "):

            heading = escape(line[4:].strip())

            story.append(Paragraph(
                heading,
                styles["SubheadingCustom"]
            ))

        # Bullet points.
        elif line.startswith(("- ", "* ")):

            content = format_inline_markdown(line[2:].strip())

            bullet = Paragraph(
                f"&bull; {content}",
                styles["BulletCustom"]
            )

            story.append(KeepTogether([bullet]))

        # Numbered points.
        elif re.match(r"^\d+[.)]\s", line):

            content = format_inline_markdown(line)

            story.append(KeepTogether([
                Paragraph(
                    content,
                    styles["BulletCustom"]
                )
            ]))

        # Regular paragraphs.
        else:

            content = format_inline_markdown(line)

            story.append(Paragraph(
                content,
                styles["BodyCustom"]
            ))


# ==================================================
# PDF GENERATION
# ==================================================

def create_pdf(
    summary,
    metadata,
    output_path,
    meeting_name
):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    styles = create_styles()

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.7 * inch,
        title=f"Meeting Summary - {meeting_name}",
        author="AI Meeting Assistant"
    )

    story = []

    # Title.
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "MEETING SUMMARY",
        styles["ReportTitle"]
    ))

    story.append(Paragraph(
        f"Meeting Report | {escape(meeting_name)}",
        styles["ReportSubtitle"]
    ))

    story.append(HRFlowable(
        width="100%",
        thickness=1.3,
        color=NAVY,
        spaceAfter=16
    ))

    # Meeting overview.
    story.append(Paragraph(
        "MEETING OVERVIEW",
        styles["SectionHeadingCustom"]
    ))

    add_metadata_table(
        story,
        metadata,
        meeting_name,
        styles
    )

    story.append(Spacer(1, 12))

    # Report content.
    add_report_content(
        story,
        summary,
        styles
    )

    # Footer.
    def add_footer(canvas, doc):

        canvas.saveState()

        page_width, _ = A4

        canvas.setStrokeColor(
            colors.HexColor("#D6DCE2")
        )

        canvas.setLineWidth(0.5)

        canvas.line(
            0.65 * inch,
            0.52 * inch,
            page_width - 0.65 * inch,
            0.52 * inch
        )

        canvas.setFont("Helvetica", 8)

        canvas.setFillColor(
            colors.HexColor("#777777")
        )

        canvas.drawString(
            0.65 * inch,
            0.35 * inch,
            "AI Meeting Assistant"
        )

        canvas.drawRightString(
            page_width - 0.65 * inch,
            0.35 * inch,
            f"Page {doc.page}"
        )

        canvas.restoreState()

    document.build(
        story,
        onFirstPage=add_footer,
        onLaterPages=add_footer
    )

    print(f"\nPDF created: {output_path}")


# ==================================================
# ORIGINAL SUMMARY WORKFLOW
# ==================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate a professional meeting summary "
            "using a local Ollama model."
        )
    )

    parser.add_argument(
        "meeting_name",
        help="Meeting filename without extension, e.g. audio3"
    )

    args = parser.parse_args()

    meeting_name = args.meeting_name

    transcript_path = (
        PROJECT_ROOT
        / "transcripts"
        / f"{meeting_name}_final.txt"
    )

    diarization_path = (
        PROJECT_ROOT
        / "diarization"
        / f"{meeting_name}.json"
    )

    output_pdf = (
        PROJECT_ROOT
        / "reports"
        / f"{meeting_name}_summary.pdf"
    )

    output_text = (
        PROJECT_ROOT
        / "reports"
        / f"{meeting_name}_summary.txt"
    )

    print("=" * 55)
    print("       AI MEETING SUMMARY GENERATOR")
    print("=" * 55)

    print("\nLoading transcript...")

    _, transcript = load_transcript(
        transcript_path
    )

    print("Loading diarization information...")

    metadata = load_diarization(
        diarization_path
    )

    print("\nMeeting Information")
    print("-" * 35)

    print(f"Detected speakers: {metadata['speaker_count']}")
    print(f"Speaker turns: {metadata['turn_count']}")
    print(f"Duration: {format_duration(metadata['duration'])}")
    print(f"Unknown turns: {metadata['unknown_turns']}")

    # Generate the original summary.
    final_summary = generate_summary_content(
        transcript,
        metadata
    )

    # Save original text output.
    output_text.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_text.write_text(
        final_summary,
        encoding="utf-8"
    )

    # Generate original PDF.
    create_pdf(
        final_summary,
        metadata,
        output_pdf,
        meeting_name
    )

    print("\n" + "=" * 55)
    print("       REPORT GENERATION COMPLETED")
    print("=" * 55)

    print(f"\nPDF: {output_pdf}")
    print(f"Text: {output_text}")


if __name__ == "__main__":
    main()
