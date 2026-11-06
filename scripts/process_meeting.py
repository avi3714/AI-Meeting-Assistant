
        )

    if not processed_path.exists():
        raise RuntimeError(
            "Preprocessing did not produce the expected WAV file."
        )

    # STAGE 2: DIARIZATION

    if diarization_path.exists() and not force:
        print("\n[SKIP] Diarization results already exist.")
    else:
        run_stage(
            "[2/4] SPEAKER DIARIZATION",
            [
                python,
                "scripts/diarize.py",
                str(processed_path),
            ]
        )

    if not diarization_path.exists():
        raise RuntimeError(
            "Diarization did not produce the expected JSON file."
        )

    # STAGE 3: TRANSCRIPTION

    if transcription_path.exists() and not force:
        print("\n[SKIP] Transcription results already exist.")
    else:
        run_stage(
            "[3/4] SPEECH TRANSCRIPTION",
            [
                python,
                "scripts/transcribe.py",
                str(processed_path),
            ]
        )

    if not transcription_path.exists():
        raise RuntimeError(
            "Transcription did not produce the expected JSON file."
        )

    # STAGE 4: MERGING + PDF GENERATION
    # Re-run this stage if either output is missing.

    if (
        final_path.exists()
        and pdf_path.exists()
        and not force
    ):
        print("\n[SKIP] TXT transcript and PDF report already exist.")
    else:
        run_stage(
            "[4/4] TRANSCRIPT MERGING + PDF GENERATION",
            [
                python,
                "scripts/merge_transcript.py",
                recording_name,
            ]
        )

    if not final_path.exists():
        raise RuntimeError(
            "Final transcript was not generated."
        )

    if not pdf_path.exists():
        raise RuntimeError(
            "PDF report was not generated."
        )

    print("\n" + "=" * 55)
    print("MEETING PROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 55)

    print(f"\nFinal transcript: {final_path.resolve()}")
    print(f"PDF report: {pdf_path.resolve()}")


def main():

    parser = argparse.ArgumentParser(
        description="AI Meeting Assistant processing pipeline"
    )

    parser.add_argument(
        "audio",
        help="Path to the input meeting audio"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun all processing stages"
    )

    args = parser.parse_args()

    try:
        process_meeting(
            Path(args.audio),
            force=args.force
        )

    except subprocess.CalledProcessError as error:
        print(
            f"\nPipeline stopped because a stage failed "
            f"(exit code: {error.returncode})."
        )
        sys.exit(error.returncode)

    except Exception as error:
        print(f"\nProcessing failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()


