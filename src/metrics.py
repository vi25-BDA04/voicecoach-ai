import json
import re
import statistics

FILLER_WORDS = [
    "um", "uh", "like", "you know", "basically",
    "actually", "literally", "right", "so", "anyway"
]

NATURAL_MARKERS = [
    "i mean", "wait", "actually", "well", "so basically",
    "you know what", "kind of", "sort of", "i think", "i guess"
]


def analyze_metrics(transcript: dict) -> dict:
    text     = transcript["text"].lower()
    duration = transcript["duration"]
    segments = transcript["segments"]

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

    # Naturalness Score
    naturalness = analyze_naturalness(transcript, filler_breakdown, wpm)

    # Authenticity Indicators
    authenticity = analyze_authenticity(transcript, wpm, filler_breakdown)

    return {
        "word_count":        word_count,
        "duration_sec":      round(duration, 1),
        "wpm":               wpm,
        "filler_breakdown":  filler_breakdown,
        "total_fillers":     total_fillers,
        "confidence_score":  confidence_score,
        "naturalness":       naturalness,
        "authenticity":      authenticity
    }


def analyze_naturalness(transcript: dict, filler_breakdown: dict, wpm: int) -> dict:
    """
    Estimates how natural/spontaneous the speech sounds vs scripted.
    Based entirely on signals already in the transcript — no extra API needed.
    """
    text       = transcript["text"].lower()
    segments   = transcript["segments"]
    words      = text.split()
    word_count = len(words)

    signals     = {}
    score_parts = []

    # Signal 1: Pace variance across segments
    # Natural speech varies in speed — scripted reading stays flat
    if len(segments) >= 3:
        seg_wpms = []
        for seg in segments:
            seg_words = len(seg["text"].split())
            seg_dur   = seg["end"] - seg["start"]
            if seg_dur > 0:
                seg_wpms.append(seg_words / (seg_dur / 60))
        if len(seg_wpms) >= 2:
            variance = statistics.stdev(seg_wpms)
            variance_score = min(100, (variance / 30) * 100)
            signals["pace_variance"] = round(variance, 1)
            score_parts.append(variance_score)
        else:
            signals["pace_variance"] = 0
            score_parts.append(20)
    else:
        signals["pace_variance"] = 0
        score_parts.append(20)

    # Signal 2: Filler word presence
    # Real spontaneous speech almost always has some fillers
    total_fillers = sum(filler_breakdown.values())
    filler_rate   = total_fillers / word_count if word_count > 0 else 0
    if filler_rate == 0:
        filler_score = 10
    elif filler_rate < 0.02:
        filler_score = 60
    elif filler_rate < 0.08:
        filler_score = 100
    else:
        filler_score = 70
    signals["filler_rate_pct"] = round(filler_rate * 100, 1)
    score_parts.append(filler_score)

    # Signal 3: Natural speech markers
    marker_count = sum(1 for m in NATURAL_MARKERS if m in text)
    marker_score = min(100, marker_count * 20)
    signals["natural_markers_found"] = marker_count
    score_parts.append(marker_score)

    # Signal 4: Sentence length variation
    sentences = [s.strip() for s in re.split(r'[.!?]', transcript["text"]) if s.strip()]
    if len(sentences) >= 3:
        lengths      = [len(s.split()) for s in sentences]
        length_std   = statistics.stdev(lengths)
        length_score = min(100, (length_std / 8) * 100)
        signals["sentence_length_variance"] = round(length_std, 1)
        score_parts.append(length_score)
    else:
        signals["sentence_length_variance"] = 0
        score_parts.append(50)

    # Signal 5: Word repetition
    unique_words = len(set(words))
    repeat_ratio = 1 - (unique_words / word_count) if word_count > 0 else 0
    repeat_score = min(100, repeat_ratio * 300)
    signals["word_repetition_pct"] = round(repeat_ratio * 100, 1)
    score_parts.append(repeat_score)

    naturalness_score = round(sum(score_parts) / len(score_parts))

    if naturalness_score >= 70:
        label  = "Spontaneous"
        detail = "Speech patterns strongly suggest genuine, unscripted delivery."
        color  = "#22C55E"
    elif naturalness_score >= 45:
        label  = "Partially Scripted"
        detail = "Some patterns suggest prepared content — natural elements also present."
        color  = "#F5A623"
    else:
        label  = "Likely Scripted"
        detail = "Speech patterns closely match scripted or read content."
        color  = "#EF4444"

    return {
        "score":   naturalness_score,
        "label":   label,
        "detail":  detail,
        "color":   color,
        "signals": signals
    }


def analyze_authenticity(transcript: dict, wpm: int, filler_breakdown: dict) -> dict:
    """
    Shows authenticity indicators — signals that distinguish human speech
    from AI-generated or synthetic audio.

    Important: This does NOT make a binary claim of 'AI' or 'Human'.
    It shows which signals are present and lets the user interpret them.
    No detection system can claim 100% accuracy — this shows honest indicators.
    """
    text       = transcript["text"].lower()
    segments   = transcript["segments"]
    words      = text.split()
    word_count = len(words)

    indicators    = []
    human_signals = 0
    total_signals = 0

    # Indicator 1: Pace variation
    if len(segments) >= 3:
        seg_wpms = []
        for seg in segments:
            seg_words = len(seg["text"].split())
            seg_dur   = seg["end"] - seg["start"]
            if seg_dur > 0:
                seg_wpms.append(seg_words / (seg_dur / 60))
        if len(seg_wpms) >= 2:
            variance = statistics.stdev(seg_wpms)
            is_human = variance > 15
            indicators.append({
                "name":     "Pace variation across segments",
                "value":    f"±{variance:.1f} WPM variation",
                "human":    is_human,
                "note":     "Natural human speech varies in speed throughout — your pace variation is consistent with genuine spontaneous delivery." if is_human else "Very uniform pace throughout — human speakers typically vary speed more. Could indicate scripted reading or synthetic audio."
            })
            if is_human: human_signals += 1
            total_signals += 1

    # Indicator 2: Filler words
    total_fillers = sum(filler_breakdown.values())
    has_fillers   = total_fillers > 0
    indicators.append({
        "name":  "Filler words present",
        "value": f"{total_fillers} detected ({', '.join(filler_breakdown.keys()) if filler_breakdown else 'none'})",
        "human": has_fillers,
        "note":  f"Filler words like {', '.join(list(filler_breakdown.keys())[:3])} are a strong signal of genuine human speech — AI audio tools rarely include them naturally." if has_fillers else "No filler words detected. While this can mean excellent delivery, it may also indicate scripted, rehearsed, or AI-generated speech."
    })
    if has_fillers: human_signals += 1
    total_signals += 1

    # Indicator 3: Self-corrections and restarts
    restart_patterns = [
        r'\b(i mean|wait|actually|no wait|sorry|let me|i think|i guess)\b',
        r'\b(\w+)\s+\1\b',
    ]
    restarts     = sum(len(re.findall(p, text)) for p in restart_patterns)
    has_restarts = restarts > 0
    indicators.append({
        "name":  "Self-corrections / restarts",
        "value": f"{restarts} instance{'s' if restarts != 1 else ''} detected",
        "human": has_restarts,
        "note":  "Mid-speech corrections and restarts are natural human behaviour — people think while they speak." if has_restarts else "No mid-speech corrections detected. AI audio is typically smooth and correction-free, which can be a synthetic audio signal."
    })
    if has_restarts: human_signals += 1
    total_signals += 1

    # Indicator 4: Sentence completeness
    sentences        = [s.strip() for s in re.split(r'[.!?]', transcript["text"]) if s.strip()]
    incomplete       = sum(1 for s in sentences if len(s.split()) < 4)
    incomplete_ratio = incomplete / len(sentences) if sentences else 0
    has_incomplete   = incomplete_ratio > 0.1
    indicators.append({
        "name":  "Sentence completeness",
        "value": f"{100 - incomplete_ratio*100:.0f}% complete sentences",
        "human": has_incomplete,
        "note":  "Short and trailing sentences are typical of natural conversational speech." if has_incomplete else "Your sentences were well-formed and complete throughout — this is a mark of a prepared, confident speaker. Note: complete sentences also appear in scripted content, so this signal alone is not conclusive."
    })
    if has_incomplete: human_signals += 1
    total_signals += 1

    # Indicator 5: Common word usage
    common_words      = {"the","a","and","i","you","it","is","was","that","this","to","of","in","my","we"}
    common_ratio      = sum(1 for w in words if w in common_words) / word_count if word_count > 0 else 0
    is_natural_vocab  = common_ratio > 0.25
    indicators.append({
        "name":  "Natural vocabulary pattern",
        "value": f"{common_ratio*100:.0f}% common everyday words",
        "human": is_natural_vocab,
        "note":  "High use of everyday common words is typical of casual human speech and conversation." if is_natural_vocab else "Lower common word ratio — may suggest more formal, written, or AI-generated content."
    })
    if is_natural_vocab: human_signals += 1
    total_signals += 1

    # Overall verdict — reframed as indicators, not a binary claim
    human_pct = (human_signals / total_signals * 100) if total_signals > 0 else 50

    if human_pct >= 70:
        verdict        = "Strong Human Indicators"
        verdict_detail = f"{human_signals} out of {total_signals} signals match natural human speech patterns. The one signal that didn't match is common in prepared speakers and is not conclusive on its own."
        verdict_color  = "#22C55E"
        verdict_badge  = "✅"
    elif human_pct >= 40:
        verdict        = "Mixed Indicators"
        verdict_detail = f"{human_signals} out of {total_signals} signals match human speech. Mixed results — could be a rehearsed human speaker or high-quality AI audio."
        verdict_color  = "#F5A623"
        verdict_badge  = "⚠️"
    else:
        verdict        = "Synthetic Audio Signals Present"
        verdict_detail = f"Only {human_signals} out of {total_signals} signals match natural human speech. Multiple indicators suggest this may be AI-generated or heavily synthesized audio."
        verdict_color  = "#EF4444"
        verdict_badge  = "🔴"

    return {
        "verdict":        verdict,
        "verdict_detail": verdict_detail,
        "verdict_color":  verdict_color,
        "verdict_badge":  verdict_badge,
        "human_signals":  human_signals,
        "total_signals":  total_signals,
        "human_pct":      round(human_pct),
        "indicators":     indicators,
        "disclaimer":     "Note: No text-based detection system can claim 100% accuracy. These are signals and indicators — not a definitive verdict. Even leading AI detection tools (GPTZero, Turnitin) report false positive rates of 20–30%."
    }


if __name__ == "__main__":
    with open("data/transcripts/Demo.json") as f:
        transcript = json.load(f)

    metrics = analyze_metrics(transcript)

    print("--- METRICS ---")
    print(f"WPM: {metrics['wpm']}, Fillers: {metrics['total_fillers']}, Score: {metrics['confidence_score']}/100")

    n = metrics["naturalness"]
    print(f"\n--- NATURALNESS: {n['score']}/100 — {n['label']} ---")
    print(n["detail"])
    for k, v in n["signals"].items():
        print(f"  {k}: {v}")

    a = metrics["authenticity"]
    print(f"\n--- AUTHENTICITY: {a['verdict']} ({a['human_pct']}% human signals) ---")
    print(a["verdict_detail"])
    for ind in a["indicators"]:
        status = "✓" if ind["human"] else "✗"
        print(f"  {status} {ind['name']}: {ind['value']}")
    print(f"\n  {a['disclaimer']}")