import whisper
import json
import os

def transcribe_audio(filepath: str, model_size: str = "base") -> dict:
    """
    Transcribe an audio/video file using Whisper.
    Returns a dict with text, segments (with timestamps), and duration.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)

    print(f"Transcribing: {filepath}")
    result = model.transcribe(filepath)

    output = {
        "text": result["text"].strip(),
        "segments": [
            {
                "start": seg["start"],
                "end":   seg["end"],
                "text":  seg["text"].strip()
            }
            for seg in result["segments"]
        ],
        "duration": result["segments"][-1]["end"] if result["segments"] else 0
    }

    return output


def save_transcript(result: dict, output_path: str):
    """Save transcript result to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Transcript saved to: {output_path}")


if __name__ == "__main__":
    audio_path  = "data/recordings/Demo.mp4"
    output_path = "data/transcripts/Demo.json"

    result = transcribe_audio(audio_path)
    save_transcript(result, output_path)

    print("\n--- TRANSCRIPT ---")
    print(result["text"])
    print("\n--- SEGMENTS (first 3) ---")
    for seg in result["segments"][:3]:
        print(f"  [{seg['start']:.1f}s → {seg['end']:.1f}s] {seg['text']}")
    print(f"\nDuration: {result['duration']:.1f} seconds")