
import argparse
import json
import re
import urllib.request
import urllib.error
import os
import requests
from dotenv import load_dotenv
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Flowable
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables from the project-root .env file.
load_dotenv(PROJECT_ROOT / ".env")

# Ollama configuration.
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b").strip()

REMOTE_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "").strip()
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()

if REMOTE_OLLAMA_URL:
    # Accept either the server's base URL or its complete generate endpoint.
    REMOTE_OLLAMA_URL = REMOTE_OLLAMA_URL.rstrip("/")

    if REMOTE_OLLAMA_URL.endswith("/api/generate"):
        OLLAMA_URL = REMOTE_OLLAMA_URL
    else:
        OLLAMA_URL = f"{REMOTE_OLLAMA_URL}/api/generate"

    print("Ollama configuration: Remote API")
else:
    OLLAMA_URL = "http://localhost:11434/api/generate"
    print("Ollama configuration: Local device")

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

    # Prepare request headers.
    headers = {
        "Content-Type": "application/json"
    }

    # Add API key only when one has been configured.
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"

    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers=headers,
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=600) as response:

            result = json.loads(
                response.read().decode("utf-8")
            )

        return result["response"].strip()

    except urllib.error.HTTPError as error:

        error_details = error.read().decode(
            "utf-8",
            errors="replace"
        )

        raise RuntimeError(
            f"Ollama API returned HTTP {error.code}.\n"
            f"Details: {error_details}"
        ) from error

    except urllib.error.URLError as error:

        if REMOTE_OLLAMA_URL:
            raise RuntimeError(
                "Could not connect to the configured remote Ollama API.\n"
                "Please check the API URL, server availability and network.\n"
                f"Details: {error}"
            ) from error

        raise RuntimeError(
            "Could not connect to local Ollama.\n"
            "Please make sure Ollama is installed and running on "
            "http://localhost:11434.\n"
            f"Details: {error}"
        ) from error

    except (KeyError, json.JSONDecodeError) as error:

        raise RuntimeError(
            "Ollama returned an unexpected response.\n"
            f"Details: {error}"
        ) from error

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
# CLEAN MEETING NOTES PDF DESIGN
# ==================================================

class Checkbox(Flowable):
    """Draw an empty checkbox for action items."""

    def __init__(self, size=9):
        super().__init__()
        self.size = size
        self.width = size
        self.height = size

    def draw(self):
        self.canv.setStrokeColor(colors.HexColor("#333333"))
        self.canv.setLineWidth(0.9)
        self.canv.rect(0, 0, self.size, self.size)



# ==================================================
# CLEAN MEETING NOTES PDF DESIGN
# ==================================================

class Checkbox(Flowable):
    """Draw an empty checkbox for action items."""

    def __init__(self, size=9):
        super().__init__()
        self.size = size
        self.width = size
        self.height = size

    def draw(self):
        self.canv.setStrokeColor(colors.HexColor("#333333"))
        self.canv.setLineWidth(0.9)
        self.canv.rect(0, 0, self.size, self.size)


# ==================================================
# PDF STYLES
# ==================================================

def create_styles():

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="MeetingTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=27,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#222222"),
        spaceAfter=17
    ))

    styles.add(ParagraphStyle(
        name="ExecutiveSummary",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#333333"),
        spaceAfter=15
    ))

    styles.add(ParagraphStyle(
        name="SectionHeadingNew",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#292929"),
        spaceBefore=13,
        spaceAfter=6,
        keepWithNext=True
    ))

    styles.add(ParagraphStyle(
        name="TopicHeadingNew",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=9.7,
        leading=13,
        textColor=colors.HexColor("#333333"),
        spaceBefore=10,
        spaceAfter=3,
        keepWithNext=True
    ))

    styles.add(ParagraphStyle(
        name="BodyNew",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.3,
        leading=13.5,
        textColor=colors.HexColor("#333333"),
        spaceAfter=6
    ))

    styles.add(ParagraphStyle(
        name="BulletNew",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.3,
        leading=13.5,
        leftIndent=18,
        firstLineIndent=-9,
        textColor=colors.HexColor("#333333"),
        spaceAfter=2
    ))

    styles.add(ParagraphStyle(
        name="NestedBulletNew",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=13,
        leftIndent=34,
        firstLineIndent=-9,
        textColor=colors.HexColor("#333333"),
        spaceAfter=2
    ))

    return styles


# ==================================================
# MARKDOWN FORMATTING
# ==================================================

def clean_heading(text):

    text = re.sub(
        r"^\d+(?:\.\d+)*[.)]?\s*",
        "",
        text.strip()
    )

    return text


def add_bullet(story, content, styles, nested=False):

    content = format_inline_markdown(content)

    bullet_symbol = "•"

    if nested:
        bullet_symbol = "–"

    style = (
        styles["NestedBulletNew"]
        if nested
        else styles["BulletNew"]
    )

    story.append(Paragraph(
        f"{bullet_symbol} {content}",
        style
    ))


def add_action_item(story, content, styles):

    content = format_inline_markdown(content)

    checkbox = Checkbox(size=9)

    paragraph = Paragraph(
        content,
        styles["BulletNew"]
    )

    row = Table(
        [[checkbox, paragraph]],
        colWidths=[15, 470]
    )

    row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3)
    ]))

    story.append(row)


# ==================================================
# SUMMARY CONTENT RENDERER
# ==================================================


def add_report_content(story, summary, styles):

    lines = summary.splitlines()

    current_section = ""
    executive_summary = True

    for raw_line in lines:

        # Preserve indentation to identify nested bullets.
        indentation = len(raw_line) - len(raw_line.lstrip())

        line = raw_line.strip()

        if not line:
            continue

        # --------------------------------------------------
        # MAIN SECTION HEADINGS
        # --------------------------------------------------

        if line.startswith("## "):

            heading = clean_heading(line[3:])

            current_section = heading.lower()

            # Executive Summary becomes the introductory
            # paragraph rather than a separate heading.
            if current_section == "executive summary":
                executive_summary = True
                continue

            executive_summary = False

            # Topics Discussed is a container heading.
            # Its individual topic headings are displayed.
            if current_section == "topics discussed":
                continue

            # Rename Action Items to Next steps.
            if current_section == "action items":
                heading = "Next steps"

            story.append(Paragraph(
                escape(heading),
                styles["SectionHeadingNew"]
            ))

        # --------------------------------------------------
        # SUBHEADINGS / TOPICS
        # --------------------------------------------------

        elif line.startswith("### "):

            heading = clean_heading(line[4:])

            story.append(Paragraph(
                escape(heading),
                styles["TopicHeadingNew"]
            ))

            executive_summary = False

        # --------------------------------------------------
        # BULLET POINTS
        # --------------------------------------------------

        elif line.startswith(("- ", "* ")):

            content = line[2:].strip()

            # Action items become empty checkboxes.
            if current_section == "action items":

                add_action_item(
                    story,
                    content,
                    styles
                )

            else:

                add_bullet(
                    story,
                    content,
                    styles,
                    nested=(indentation >= 2)
                )

            executive_summary = False

        # --------------------------------------------------
        # REGULAR PARAGRAPHS
        # --------------------------------------------------

        else:

            content = format_inline_markdown(line)

            selected_style = (
                styles["ExecutiveSummary"]
                if executive_summary
                else styles["BodyNew"]
            )

            story.append(Paragraph(
                content,
                selected_style
            ))

            executive_summary = False


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
        rightMargin=0.85 * inch,
        leftMargin=0.85 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.65 * inch,
        title="Meeting Summary",
        author="AI Meeting Assistant"
    )

    story = []

    # Main title.
    story.append(Paragraph(
        "Meeting Summary",
        styles["MeetingTitle"]
    ))

    # Render the LLM-generated summary.
    add_report_content(
        story,
        summary,
        styles
    )

    document.build(story)

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
