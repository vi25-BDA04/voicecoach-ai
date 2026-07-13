import json

FILLER_WORDS = [
    "um", "uh", "like", "you know", "basically",
    "actually", "literally", "right", "so", "anyway"
]

def analyze_metrics(transcript: dict) -> dict:
    text     = transcript["text"].lower()
    duration = transcript["duration"]

    # Filler word counts
    filler_breakdown = {}
    for word in FILLER_WORDS:
        count = text.split().count(word)
        if count > 0:
            filler_breakdown[word] = count

    total_fillers = sum(filler_breakdown.values())

    # Words per minute
    word_count       = len(transcript["text"].split())
    duration_minutes = duration / 60
    wpm              = round(word_count / duration_minutes) if duration_minutes > 0 else 0

    # Confidence score (0-100)
    filler_rate  = total_fillers / word_count if word_count > 0 else 0
    filler_score = max(0, 100 - (filler_rate * 300))
    pace_penalty = abs(wpm - 140) / 140 * 100
    pace_score   = max(0, 100 - pace_penalty)
    confidence_score = round((filler_score * 0.6) + (pace_score * 0.4))

    return {
        "word_count":       word_count,
        "duration_sec":     round(duration, 1),
        "wpm":              wpm,
        "filler_breakdown": filler_breakdown,
        "total_fillers":    total_fillers,
        "confidence_score": confidence_score
    }


if __name__ == "__main__":
    with open("data/transcripts/Demo.json") as f:
        transcript = json.load(f)

    metrics = analyze_metrics(transcript)

    print("--- METRICS ---")
    print(f"Word count:       {metrics['word_count']}")
    print(f"Duration:         {metrics['duration_sec']}s")
    print(f"Pace:             {metrics['wpm']} WPM")
    print(f"Total fillers:    {metrics['total_fillers']}")
    print(f"Filler breakdown: {metrics['filler_breakdown']}")
    print(f"Confidence score: {metrics['confidence_score']}/100")