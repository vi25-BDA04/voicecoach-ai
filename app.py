import streamlit as st
import os
import sys
import tempfile

sys.path.append(".")
from src.transcribe import transcribe_audio
from src.metrics import analyze_metrics
from src.coach import get_coaching_feedback
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="VoiceCoach AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sora:wght@400;600;700;800&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: #090C14;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 4rem !important; max-width: 900px !important; margin: auto; }

/* ── HERO ── */
.hero-wrap {
    text-align: center;
    padding: 4rem 1rem 2.5rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    background: rgba(245,166,35,0.12);
    border: 1px solid rgba(245,166,35,0.3);
    color: #F5A623;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 100px;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-family: 'Sora', sans-serif !important;
    font-size: 3.6rem;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1.08;
    letter-spacing: -0.04em;
    margin-bottom: 1rem;
}
.hero-title span { color: #F5A623; }
.hero-desc {
    font-size: 1.05rem;
    color: #6B7280;
    max-width: 480px;
    margin: 0 auto 2.5rem;
    line-height: 1.65;
}

/* Animated gradient orb behind hero */
.orb {
    position: absolute;
    top: 20%;
    left: 50%;
    transform: translateX(-50%);
    width: 600px;
    height: 300px;
    background: radial-gradient(ellipse, rgba(245,166,35,0.07) 0%, transparent 70%);
    pointer-events: none;
    animation: pulse 4s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 0.6; transform: translateX(-50%) scale(1); }
    50% { opacity: 1; transform: translateX(-50%) scale(1.08); }
}

/* Stats bar */
.stats-bar {
    display: flex;
    justify-content: center;
    gap: 2.5rem;
    padding: 1.2rem 2rem;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    margin-bottom: 2.5rem;
}
.stat-item { text-align: center; }
.stat-val {
    font-family: 'Sora', sans-serif !important;
    font-size: 1.4rem;
    font-weight: 700;
    color: #F5A623;
}
.stat-lbl {
    font-size: 0.72rem;
    color: #4B5563;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 2px;
}

/* ── UPLOAD CARD ── */
.upload-card {
    background: linear-gradient(135deg, #111827 0%, #0F172A 100%);
    border: 1.5px dashed rgba(245,166,35,0.35);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    transition: border-color 0.3s;
    margin-bottom: 1rem;
    animation: fadeUp 0.6s ease both;
}
.upload-card:hover { border-color: rgba(245,166,35,0.7); }
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.upload-icon {
    font-size: 2.8rem;
    margin-bottom: 0.8rem;
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
}
.upload-title {
    font-family: 'Sora', sans-serif !important;
    font-size: 1.15rem;
    font-weight: 700;
    color: #F9FAFB;
    margin-bottom: 0.4rem;
}
.upload-sub {
    font-size: 0.83rem;
    color: #4B5563;
    margin-bottom: 1.5rem;
}
.format-pills {
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    color: #6B7280;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 0.25rem 0.7rem;
    border-radius: 100px;
    letter-spacing: 0.06em;
}

/* ── BUTTON ── */
div.stButton > button {
    background: linear-gradient(135deg, #F5A623 0%, #E8941A 100%) !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2.5rem !important;
    width: 100% !important;
    letter-spacing: -0.01em !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 24px rgba(245,166,35,0.25) !important;
}
div.stButton > button:hover {
    box-shadow: 0 6px 32px rgba(245,166,35,0.45) !important;
    transform: translateY(-1px) !important;
}

/* ── SECTION LABELS ── */
.section-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #F5A623;
    margin-bottom: 1rem;
    margin-top: 2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(245,166,35,0.15);
}

/* ── METRIC CARDS ── */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin-bottom: 1.5rem;
    animation: fadeUp 0.5s ease both;
}
.m-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
    transition: border-color 0.3s, transform 0.2s;
}
.m-card:hover {
    border-color: rgba(245,166,35,0.3);
    transform: translateY(-2px);
}
.m-val {
    font-family: 'Sora', sans-serif !important;
    font-size: 1.9rem;
    font-weight: 700;
    color: #F5A623;
    line-height: 1;
}
.m-lbl {
    font-size: 0.72rem;
    color: #4B5563;
    margin-top: 0.35rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.m-hint {
    font-size: 0.65rem;
    color: #374151;
    margin-top: 0.25rem;
}

/* ── SCORE RING CARD ── */
.score-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    animation: fadeUp 0.6s ease both;
}

/* ── FEEDBACK CARD ── */
.feedback-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-left: 3px solid #F5A623;
    border-radius: 0 16px 16px 0;
    padding: 1.6rem 1.8rem;
    font-size: 0.96rem;
    line-height: 1.78;
    color: #D1D5DB;
    animation: fadeUp 0.7s ease both;
}

/* ── CHAT ── */
.chat-header-wrap {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 16px 16px 0 0;
    padding: 1rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.chat-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #F5A623;
    animation: blink 2s ease infinite;
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}
.chat-title-text {
    font-family: 'Sora', sans-serif !important;
    font-size: 0.95rem;
    font-weight: 600;
    color: #F9FAFB;
}
.chat-sub-text {
    font-size: 0.78rem;
    color: #4B5563;
    margin-left: auto;
}

[data-testid="stChatMessage"] {
    background: #111827 !important;
    border: 1px solid #1F2937 !important;
    border-radius: 12px !important;
    margin-bottom: 0.5rem !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: #111827 !important;
    border: 1px solid #1F2937 !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    color: #6B7280 !important;
    font-size: 0.85rem !important;
}

/* File uploader — hide the default ugly box */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploader"] > div {
    background: transparent !important;
    border: none !important;
}
section[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1.5px dashed rgba(245,166,35,0.3) !important;
    border-radius: 12px !important;
}
section[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(245,166,35,0.7) !important;
    background: rgba(245,166,35,0.03) !important;
}
section[data-testid="stFileUploaderDropzone"] span {
    color: #6B7280 !important;
}
section[data-testid="stFileUploaderDropzone"] button {
    background: rgba(245,166,35,0.1) !important;
    color: #F5A623 !important;
    border: 1px solid rgba(245,166,35,0.3) !important;
    border-radius: 8px !important;
}

/* Success */
[data-testid="stAlert"] {
    background: rgba(34, 197, 94, 0.08) !important;
    border: 1px solid rgba(34, 197, 94, 0.2) !important;
    border-radius: 10px !important;
    color: #86EFAC !important;
}

/* Audio player */
audio {
    width: 100%;
    border-radius: 10px;
    margin: 0.5rem 0;
}

/* Spinner */
.stSpinner > div { border-top-color: #F5A623 !important; }

/* hr */
hr { border-color: #1F2937 !important; margin: 2rem 0 !important; }

/* Chat input */
[data-testid="stChatInput"] {
    background: #111827 !important;
    border: 1px solid #1F2937 !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    color: #F9FAFB !important;
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──
for key in ["transcript","metrics","feedback","messages","analysed"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "messages" else (False if key == "analysed" else None)

# ══════════════════
# HERO
# ══════════════════
st.markdown("""
<div class="hero-wrap">
    <div class="orb"></div>
    <div class="hero-badge">🎙️ AI-Powered Speech Analysis</div>
    <div class="hero-title">Speak better,<br><span>every time.</span></div>
    <div class="hero-desc">Upload any recording and get instant, personalised coaching — filler word analysis, pacing feedback, and an AI coach ready to chat.</div>
</div>
""", unsafe_allow_html=True)

# Stats bar
st.markdown("""
<div class="stats-bar">
    <div class="stat-item"><div class="stat-val">4</div><div class="stat-lbl">AI Stages</div></div>
    <div class="stat-item"><div class="stat-val">~2min</div><div class="stat-lbl">Analysis time</div></div>
    <div class="stat-item"><div class="stat-val">Free</div><div class="stat-lbl">No cost</div></div>
    <div class="stat-item"><div class="stat-val">100%</div><div class="stat-lbl">Private</div></div>
</div>
""", unsafe_allow_html=True)

# ══════════════════
# UPLOAD
# ══════════════════
if not st.session_state.analysed:

    st.markdown("""
    <div class="upload-icon">🎤</div>
    <div class="upload-title">Drop your recording here</div>
    <div class="upload-sub">Audio or video file · up to 200MB</div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload", type=["mp3","wav","mp4","m4a","mov"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <div class="format-pills">
        <span class="pill">MP3</span><span class="pill">WAV</span>
        <span class="pill">MP4</span><span class="pill">M4A</span><span class="pill">MOV</span>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file:
        st.markdown("<br>", unsafe_allow_html=True)
        st.audio(uploaded_file)
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("✨  Analyse my speech"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            with st.spinner("Transcribing your recording with Whisper..."):
                transcript = transcribe_audio(tmp_path)
            os.unlink(tmp_path)

            with st.spinner("Analysing speech patterns..."):
                metrics = analyze_metrics(transcript)

            with st.spinner("Generating coaching feedback (agentic loop)..."):
                feedback = get_coaching_feedback(transcript, metrics)

            st.session_state.transcript = transcript
            st.session_state.metrics    = metrics
            st.session_state.feedback   = feedback
            st.session_state.analysed   = True
            st.rerun()

# ══════════════════
# RESULTS
# ══════════════════
if st.session_state.analysed:
    transcript = st.session_state.transcript
    metrics    = st.session_state.metrics
    feedback   = st.session_state.feedback
    score      = metrics["confidence_score"]

    st.success("✅  Analysis complete — your coaching report is ready.")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Score + Metrics ──
    col_score, col_right = st.columns([1, 2], gap="large")

    with col_score:
        st.markdown('<div class="section-label">Confidence Score</div>', unsafe_allow_html=True)
        ring_color = "#22C55E" if score >= 75 else "#F5A623" if score >= 50 else "#EF4444"
        fig_ring = go.Figure(go.Pie(
            values=[score, 100-score],
            hole=0.78,
            marker=dict(colors=[ring_color, "#1F2937"]),
            textinfo="none", hoverinfo="skip", showlegend=False,
            direction="clockwise", sort=False
        ))
        fig_ring.update_layout(
            margin=dict(l=10,r=10,t=10,b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=200,
            annotations=[dict(
                text=f"<b>{score}</b>",
                x=0.5, y=0.48,
                font=dict(size=44, color=ring_color, family="Sora"),
                showarrow=False
            )]
        )
        st.plotly_chart(fig_ring, use_container_width=True, config={"displayModeBar":False})
        st.markdown(f'<div style="text-align:center;color:#4B5563;font-size:0.75rem;margin-top:-1.5rem">out of 100</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-label">Speech Metrics</div>', unsafe_allow_html=True)
        wpm_hint = "✓ Good pace" if 120 <= metrics['wpm'] <= 150 else "↑ Too fast" if metrics['wpm'] > 150 else "↓ Too slow"
        st.markdown(f"""
        <div class="metrics-grid">
            <div class="m-card">
                <div class="m-val">{metrics['wpm']}</div>
                <div class="m-lbl">Words/min</div>
                <div class="m-hint">{wpm_hint}</div>
            </div>
            <div class="m-card">
                <div class="m-val">{metrics['total_fillers']}</div>
                <div class="m-lbl">Fillers</div>
                <div class="m-hint">total count</div>
            </div>
            <div class="m-card">
                <div class="m-val">{metrics['word_count']}</div>
                <div class="m-lbl">Words</div>
                <div class="m-hint">spoken</div>
            </div>
            <div class="m-card">
                <div class="m-val">{int(metrics['duration_sec'])}s</div>
                <div class="m-lbl">Duration</div>
                <div class="m-hint">recording</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Filler Chart ──
    if metrics["filler_breakdown"]:
        st.markdown('<div class="section-label">Filler Word Breakdown</div>', unsafe_allow_html=True)
        fig_bar = px.bar(
            x=list(metrics["filler_breakdown"].keys()),
            y=list(metrics["filler_breakdown"].values()),
            labels={"x":"","y":"Count"},
            color=list(metrics["filler_breakdown"].values()),
            color_continuous_scale=[[0,"#1F2937"],[0.5,"#D97706"],[1,"#F5A623"]],
            text=list(metrics["filler_breakdown"].values()),
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#6B7280", family="Inter", size=12),
            coloraxis_showscale=False,
            margin=dict(l=0,r=0,t=10,b=0), height=240,
            xaxis=dict(gridcolor="#1F2937", tickfont=dict(color="#9CA3AF", size=13)),
            yaxis=dict(gridcolor="#1F2937", tickfont=dict(color="#6B7280")),
        )
        fig_bar.update_traces(
            marker_line_width=0,
            textposition="outside",
            textfont=dict(color="#F5A623", size=13),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar":False})

    # ── Transcript ──
    with st.expander("📄  View full transcript"):
        st.markdown(f'<div style="color:#9CA3AF;font-size:0.92rem;line-height:1.8;padding:0.5rem 0">{transcript["text"]}</div>', unsafe_allow_html=True)

    # ── Coaching Feedback ──
    st.markdown('<div class="section-label">Coaching Feedback</div>', unsafe_allow_html=True)

    with st.expander("🔍  Show draft (before self-review)"):
        st.markdown(f'<div style="color:#6B7280;font-size:0.88rem;line-height:1.75">{feedback["draft"]}</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="feedback-card">{feedback["final"]}</div>', unsafe_allow_html=True)

    # ── Chat ──
    st.markdown("---")
    st.markdown("""
    <div class="chat-header-wrap">
        <div class="chat-dot"></div>
        <div class="chat-title-text">Ask your coach</div>
        <div class="chat-sub-text">Remembers your full session</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div style="color:#4B5563;font-size:0.82rem;margin:0.75rem 0 1rem">Your coach has full context of your speech and feedback. Ask anything — it remembers the whole conversation.</div>', unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if user_question := st.chat_input("How can I reduce my filler words?  •  What was my strongest moment?"):
        st.session_state.messages.append({"role":"user","content":user_question})
        with st.chat_message("user"):
            st.write(user_question)

        system = f"""You are a warm, expert communication coach in an ongoing coaching session.

Full speech context:
- Transcript: {transcript['text']}
- WPM: {metrics['wpm']} (ideal: 120-150), Fillers: {metrics['filler_breakdown']}, Score: {metrics['confidence_score']}/100
- Coaching feedback already given: {feedback['final']}

Be conversational, specific, and encouraging. Reference the actual transcript when relevant.
Keep responses concise — 2-4 sentences unless more detail is genuinely needed."""

        messages_for_api = [{"role":"system","content":system}]
        for msg in st.session_state.messages:
            messages_for_api.append({"role":msg["role"],"content":msg["content"]})

        with st.spinner(""):
            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages_for_api,
                max_tokens=500,
                temperature=0.7
            )
            reply = response.choices[0].message.content.strip()

        st.session_state.messages.append({"role":"assistant","content":reply})
        with st.chat_message("assistant"):
            st.write(reply)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("↩  Analyse a new recording"):
        for key in ["transcript","metrics","feedback","messages","analysed"]:
            del st.session_state[key]
        st.rerun()
