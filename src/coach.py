from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def call_gpt(system_prompt: str, user_prompt: str) -> str:
    """Single LLM call — returns the response text."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


def draft_feedback(transcript: dict, metrics: dict) -> str:
    """Call 1 — Coach persona drafts feedback."""
    system_prompt = """You are a warm, supportive communication coach helping 
    someone build public speaking confidence. Based on their transcript and 
    speech metrics, give 2-3 specific, encouraging, actionable pieces of 
    feedback. Reference actual things from their speech. Never be harsh."""

    user_prompt = f"""
    Transcript: {transcript['text']}

    Metrics:
    - Words per minute: {metrics['wpm']}
    - Total filler words: {metrics['total_fillers']}
    - Filler breakdown: {metrics['filler_breakdown']}
    - Confidence score: {metrics['confidence_score']}/100
    - Duration: {metrics['duration_sec']} seconds
    """
    return call_gpt(system_prompt, user_prompt)


def review_feedback(draft: str, transcript: dict, metrics: dict) -> str:
    """Call 2 — Critic persona reviews and improves the draft."""
    system_prompt = """You are reviewing coaching feedback before it is shown 
    to a user. Check it against these three criteria:
    1. Specific — does it reference actual things from the transcript or metrics?
    2. Encouraging — is the tone warm and supportive, not harsh or critical?
    3. Actionable — does it give a clear next step the person can act on?

    If it meets all three, return it unchanged.
    If not, rewrite it so it does.
    Return ONLY the final feedback text, nothing else."""

    user_prompt = f"""
    Draft feedback: {draft}

    Original transcript: {transcript['text']}
    Metrics: {metrics}
    """
    return call_gpt(system_prompt, user_prompt)


def get_coaching_feedback(transcript: dict, metrics: dict) -> dict:
    """
    Full agentic loop:
    Draft feedback → review and improve → return both for transparency.
    """
    print("Generating draft feedback...")
    draft = draft_feedback(transcript, metrics)

    print("Running self-review...")
    final = review_feedback(draft, transcript, metrics)

    return {
        "draft": draft,
        "final": final
    }


if __name__ == "__main__":
    import json
    import sys
    sys.path.append(".")

    with open("data/transcripts/Demo.json") as f:
        transcript = json.load(f)

    from src.metrics import analyze_metrics
    metrics = analyze_metrics(transcript)

    result = get_coaching_feedback(transcript, metrics)

    print("\n--- DRAFT FEEDBACK (before self-review) ---")
    print(result["draft"])
    print("\n--- FINAL FEEDBACK (after self-review) ---")
    print(result["final"])
