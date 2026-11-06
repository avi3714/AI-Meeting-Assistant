from pathlib import Path
import json
import sys
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

MAX_PAUSE = 1.5
MIN_OVERLAP = 0.05
AMBIGUITY_RATIO = 0.80
MIN_OVERLAP_DURATION = 0.20
BOUNDARY_TOLERANCE = 0.15


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_timestamp(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_pdf(
    recording_name,
    diarization,
    segments,
    overlaps_found,
    unknown_count,
    pdf_path,
    duration
):
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=26,
        textColor=colors.HexColor("#172554"),
        alignment=TA_CENTER,
        spaceAfter=5,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#64748B"),
        alignment=TA_CENTER,
        spaceAfter=16,
    )

    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#172554"),
        spaceBefore=12,
        spaceAfter=8,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "TranscriptBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#1E293B"),
        alignment=TA_LEFT,
        spaceAfter=0,
        wordWrap="LTR",
    )

    meta_style = ParagraphStyle(
        "Meta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#475569"),
    )

    story = []

    story.append(Paragraph("MEETING TRANSCRIPT", title_style))
    story.append(Paragraph(escape(recording_name), subtitle_style))

    metadata = [
        [
            Paragraph("<b>Duration</b><br/>" + format_timestamp(duration), meta_style),
            Paragraph(
                "<b>Speakers</b><br/>" +
                str(diarization.get("number_of_speakers", "Unknown")),
                meta_style
            ),
            Paragraph("<b>Transcript sections</b><br/>" + str(len(segments)), meta_style),
            Paragraph("<b>Unknown words</b><br/>" + str(unknown_count), meta_style),
        ]
    ]

    meta_table = Table(
        metadata,
        colWidths=[40 * mm, 32 * mm, 47 * mm, 41 * mm],
        hAlign="LEFT"
    )

    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))

    story.append(meta_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Transcript", section_style))

    # Assign one consistent colour to each speaker, based on first appearance.
    speaker_palette = [
        ("#DBEAFE", "#1E40AF"),
        ("#DCFCE7", "#166534"),
        ("#FCE7F3", "#9D174D"),
        ("#FEF3C7", "#92400E"),
        ("#EDE9FE", "#5B21B6"),
        ("#CFFAFE", "#155E75"),
    ]

    speaker_styles = {}

    for segment in segments:
        speaker_name = str(segment["speaker"]).strip()

        if speaker_name.upper() == "UNKNOWN" or speaker_name in speaker_styles:
            continue

        palette_index = len(speaker_styles) % len(speaker_palette)
        header_hex, label_hex = speaker_palette[palette_index]

        speaker_styles[speaker_name] = (
            colors.HexColor(header_hex),
            colors.HexColor(label_hex),
        )

    content_width = A4[0] - (50 * mm)
    header_col_widths = [content_width * 0.72, content_width * 0.28]

    for index, segment in enumerate(segments):

        speaker_name = str(segment["speaker"]).strip()
        speaker = escape(speaker_name)

        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])

        text = (
            escape(str(segment["text"]))
            .replace("\n\n", "<br/><br/>")
            .replace("\n", "<br/>")
        )

        is_unknown = speaker_name.upper() == "UNKNOWN"

        if is_unknown:
            header_color = colors.HexColor("#FECACA")
            body_color = colors.HexColor("#FEF2F2")
            label_color = colors.HexColor("#991B1B")
        else:
            header_color, label_color = speaker_styles[speaker_name]
            body_color = colors.HexColor("#F8FAFC")

        timestamp = Paragraph(
            f"<b>{start} – {end}</b>",
            ParagraphStyle(
                f"Timestamp{index}",
                parent=meta_style,
                fontName="Helvetica-Bold",
                textColor=colors.HexColor("#334155"),
            ),
        )

        speaker_label = Paragraph(
            f"<b>{speaker}</b>",
            ParagraphStyle(
                f"Speaker{index}",
                parent=meta_style,
                alignment=TA_RIGHT,
                textColor=label_color,
            ),
        )

        # Header strip
        header = Table(
            [[timestamp, speaker_label]],
            colWidths=header_col_widths,
            hAlign="LEFT",
        )

        header.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), header_color),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))

        # Body background without a visible border.
        card_body_style = ParagraphStyle(
    f"CardBody{index}",
    parent=body_style,
    backColor=body_color,
    borderWidth=0,
    borderPadding=0,
    leftIndent=9,
    rightIndent=9,
    spaceBefore=8,
    spaceAfter=8,
)

        body = Paragraph(text, card_body_style)

        story.append(header)
        story.append(body)
        story.append(Spacer(1, 8))

    story.append(Paragraph("Overlapping Speech", section_style))

    if not overlaps_found:
        story.append(
            Paragraph("No overlapping speech detected.", body_style)
        )

    else:
        overlap_rows = [[
            Paragraph("<b>Time interval</b>", meta_style),
            Paragraph("<b>Speakers</b>", meta_style),
        ]]

        for overlap in overlaps_found:

            interval = (
                f"{format_timestamp(overlap['start'])} – "
                f"{format_timestamp(overlap['end'])}"
            )

            pair = (
                f"{escape(str(overlap['speaker1']))} and "
                f"{escape(str(overlap['speaker2']))}"
            )

            overlap_rows.append([
                Paragraph(interval, body_style),
                Paragraph(pair, body_style),
            ])

        overlap_table = Table(
            overlap_rows,
            colWidths=[55 * mm, 102 * mm],
            repeatRows=1
        )

        overlap_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                colors.white,
                colors.HexColor("#F8FAFC")
            ]),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))

        story.append(overlap_table)

    def add_page_number(canvas, doc):
        canvas.saveState()

        width, height = A4

        canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
        canvas.setLineWidth(0.5)

        canvas.line(
            25 * mm,
            16 * mm,
            width - 25 * mm,
            16 * mm
        )

        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748B"))

        canvas.drawString(
            25 * mm,
            10 * mm,
            "AI Meeting Assistant"
        )

        canvas.drawRightString(
            width - 25 * mm,
            10 * mm,
            f"Page {doc.page}"
        )

        canvas.restoreState()

    pdf = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=25 * mm,
        leftMargin=25 * mm,
        topMargin=20 * mm,
        bottomMargin=23 * mm,
        title=f"Meeting Transcript - {recording_name}",
        author="AI Meeting Assistant",
    )

    pdf.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number
    )


def merge_transcript(audio_path):

    recording_name = Path(audio_path).stem

    diarization_file = Path("diarization") / f"{recording_name}.json"
    transcription_file = Path("transcriptions") / f"{recording_name}.json"

    if not diarization_file.exists():
        raise FileNotFoundError(
            f"Diarization file not found: {diarization_file}"
        )

    if not transcription_file.exists():
        raise FileNotFoundError(
            f"Transcription file not found: {transcription_file}"
        )

    diarization = load_json(diarization_file)
    transcription = load_json(transcription_file)

    turns = sorted(
        diarization["turns"],
        key=lambda x: x["start"]
    )

    words = []

    for segment in transcription["segments"]:
        words.extend(segment.get("words", []))

    assigned_words = []
    ambiguous_count = 0
    unknown_count = 0

    for word in words:

        start = word["start"]
        end = word["end"]
        text = word["word"].strip()

        if not text:
            continue

        overlaps = []

        for turn in turns:

            overlap = max(
                0,
                min(end, turn["end"]) - max(start, turn["start"])
            )

            if overlap > 0:
                overlaps.append((overlap, turn["speaker"]))

        overlaps.sort(reverse=True)

        speaker = "UNKNOWN"
        ambiguous = False

        if overlaps and overlaps[0][0] >= MIN_OVERLAP:

            best_overlap, best_speaker = overlaps[0]

            if (
                len(overlaps) > 1
                and overlaps[1][0] >= best_overlap * AMBIGUITY_RATIO
                and overlaps[1][1] != best_speaker
            ):
                ambiguous = True

            else:
                speaker = best_speaker

        else:

            distances = []

            for turn in turns:

                if end < turn["start"]:
                    distance = turn["start"] - end

                elif start > turn["end"]:
                    distance = start - turn["end"]

                else:
                    distance = 0

                if distance <= BOUNDARY_TOLERANCE:
                    distances.append((distance, turn["speaker"]))

            distances.sort()

            if distances:

                if (
                    len(distances) > 1
                    and distances[1][0] - distances[0][0] < 0.05
                    and distances[1][1] != distances[0][1]
                ):
                    ambiguous = True

                else:
                    speaker = distances[0][1]

        if ambiguous:
            speaker = "UNKNOWN"
            ambiguous_count += 1

        if speaker == "UNKNOWN":
            unknown_count += 1

        assigned_words.append({
            "start": start,
            "end": end,
            "word": text,
            "speaker": speaker
        })

    # Keep consecutive words from the same speaker in one card.
    # A pause longer than MAX_PAUSE starts a new paragraph inside that card.

    segments = []

    for word in assigned_words:

        if segments and segments[-1]["speaker"] == word["speaker"]:

            pause = word["start"] - segments[-1]["end"]

            separator = "\n\n" if pause > MAX_PAUSE else " "

            segments[-1]["end"] = word["end"]
            segments[-1]["text"] += separator + word["word"]

        else:

            segments.append({
                "start": word["start"],
                "end": word["end"],
                "speaker": word["speaker"],
                "text": word["word"]
            })

    # Detect overlapping diarization turns from different speakers.

    overlaps_found = []

    for i, first in enumerate(turns):

        for second in turns[i + 1:]:

            if second["start"] >= first["end"]:
                break

            overlap_start = max(
                first["start"],
                second["start"]
            )

            overlap_end = min(
                first["end"],
                second["end"]
            )

            if (
                first["speaker"] != second["speaker"]
                and overlap_end - overlap_start >= MIN_OVERLAP_DURATION
            ):

                overlaps_found.append({
                    "start": overlap_start,
                    "end": overlap_end,
                    "speaker1": first["speaker"],
                    "speaker2": second["speaker"]
                })

    # Save TXT transcript.

    output_file = (
        Path("transcripts") /
        f"{recording_name}_final.txt"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_file, "w", encoding="utf-8") as f:

        f.write(f"Meeting Transcript: {recording_name}\n")
        f.write("=" * 60 + "\n\n")

        f.write(
            f"Global speakers: "
            f"{diarization.get('number_of_speakers', 'Unknown')}\n"
        )

        f.write(f"Speaker turns: {len(turns)}\n")
        f.write(f"Transcript segments: {len(segments)}\n")
        f.write(f"Unknown/ambiguous words: {unknown_count}\n\n")

        f.write("TRANSCRIPT\n")
        f.write("-" * 60 + "\n")

        for segment in segments:

            f.write(
                f"[{segment['start']:.2f}s - {segment['end']:.2f}s] "
                f"{segment['speaker']}: {segment['text']}\n"
            )

        f.write("\nOVERLAPPING SPEECH\n")
        f.write("-" * 60 + "\n")

        if not overlaps_found:

            f.write("No overlapping speech detected.\n")

        else:

            for overlap in overlaps_found:

                f.write(
                    f"[{overlap['start']:.2f}s - {overlap['end']:.2f}s] "
                    f"{overlap['speaker1']} and {overlap['speaker2']}\n"
                )

    # Generate PDF report.

    pdf_dir = Path("reports")
    pdf_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf_file = pdf_dir / f"{recording_name}_transcript.pdf"

    duration = float(transcription.get("duration", 0))

    if duration <= 0 and turns:
        duration = max(turn["end"] for turn in turns)

    build_pdf(
        recording_name=recording_name,
        diarization=diarization,
        segments=segments,
        overlaps_found=overlaps_found,
        unknown_count=unknown_count,
        pdf_path=pdf_file,
        duration=duration,
    )

    print("\nTranscript merging and PDF generation completed!")
    print(f"Segments: {len(segments)}")
    print(f"Unknown/ambiguous words: {unknown_count}")
    print(f"Overlapping intervals: {len(overlaps_found)}")
    print(f"TXT saved: {output_file.resolve()}")
    print(f"PDF saved: {pdf_file.resolve()}")


if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage: python scripts/merge_transcript.py "
            "<audio_file_or_recording_name>"
        )

        sys.exit(1)

    try:
        merge_transcript(sys.argv[1])

    except Exception as e:

        print(f"\nMerging failed: {e}")
        sys.exit(1)