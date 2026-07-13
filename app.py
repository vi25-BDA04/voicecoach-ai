import streamlit as st
import streamlit.components.v1 as components
import os
import re
import sys
import json
import time
import html
import sqlite3
import tempfile
from datetime import datetime

sys.path.append(".")
from src.transcribe import transcribe_audio
from src.metrics import analyze_metrics
from src.coach import get_coaching_feedback
import plotly.express as px
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="VoiceCoach AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════════
# HISTORY STORE — SQLite so past sessions survive app restarts,
# unlike st.session_state which only lasts the current browser tab.
# ══════════════════════════════════════════════════════════════════
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voicecoach_history.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT NOT NULL,
            filename TEXT,
            created_at TEXT NOT NULL,
            wpm REAL,
            total_fillers INTEGER,
            word_count INTEGER,
            duration_sec REAL,
            confidence_score INTEGER
        )
    """)
    conn.commit()
    conn.close()


def save_session_to_history(user_name: str, filename: str, metrics: dict):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO sessions
           (user_name, filename, created_at, wpm, total_fillers, word_count, duration_sec, confidence_score)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            user_name, filename, datetime.now().isoformat(timespec="seconds"),
            metrics.get("wpm"), metrics.get("total_fillers"), metrics.get("word_count"),
            metrics.get("duration_sec"), metrics.get("confidence_score"),
        ),
    )
    conn.commit()
    conn.close()


def get_user_history(user_name: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM sessions WHERE user_name = ? ORDER BY created_at DESC", (user_name,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_user_history(user_name: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM sessions WHERE user_name = ?", (user_name,))
    conn.commit()
    conn.close()


init_db()

# ══════════════════════════════════════════════════════════════════
# STYLE
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sora:wght@400;600;700;800&display=swap');

:root {
    --bg: #090C14;
    --bg-card: #111827;
    --bg-card-2: #0F172A;
    --border: #1F2937;
    --accent: #F5A623;
    --accent-dim: rgba(245,166,35,0.14);
    --accent-2: #38BDF8;
    --success: #22C55E;
    --danger: #EF4444;
    --text-hi: #F9FAFB;
    --text-mid: #B4BCC8;
    --text-lo: #8A93A3;
    --text-dim: #667085;
    --text-faint: #4B5563;
}

*:not([data-testid="stIconMaterial"]),
html, body,
[class*="css"]:not([data-testid="stIconMaterial"]) { font-family: 'Inter', sans-serif !important; }

/* Restore Streamlit's icon-ligature font explicitly — this is what actually renders
   expander arrows and chat avatars as glyphs instead of literal text like "face" or
   "smart_toy". Without this exact font-feature-settings/liga combo the browser shows
   the raw icon name as text and it overlaps everything around it. */
[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
    font-weight: normal !important;
    font-style: normal !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
    direction: ltr !important;
    -webkit-font-feature-settings: 'liga' !important;
    font-feature-settings: 'liga' !important;
    -webkit-font-smoothing: antialiased !important;
}

html, body, #root {
    background: var(--bg) !important;
}

.stApp {
    background:
        radial-gradient(ellipse 900px 500px at 50% -10%, rgba(245,166,35,0.06) 0%, transparent 60%),
        linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px),
        var(--bg);
    background-size: auto, 42px 42px, 42px 42px, auto;
    background-position: center;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 4rem !important; max-width: 900px !important; margin: auto; }

::selection { background: rgba(245,166,35,0.25); color: #fff; }

/* custom scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1F2937; border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: #2b3646; }

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: 0.001ms !important; animation-iteration-count: 1 !important; transition-duration: 0.001ms !important; }
}

/* ── WELCOME / NAME GATE ── */
.welcome-wrap {
    text-align: center; padding: 3.5rem 1rem 2rem; position: relative; overflow: hidden;
    border-radius: 24px;
    background:
        radial-gradient(ellipse 500px 340px at 50% 30%, rgba(245,166,35,0.09) 0%, transparent 65%),
        linear-gradient(160deg, rgba(255,255,255,0.02) 0%, rgba(255,255,255,0.005) 100%);
}
.welcome-orb { position: absolute; border-radius: 50%; filter: blur(46px); pointer-events: none; z-index: 0; animation: orbDrift ease-in-out infinite; }
.welcome-orb.o1 { width: 220px; height: 220px; background: rgba(245,166,35,0.22); top: -50px; left: 6%; animation-duration: 9s; }
.welcome-orb.o2 { width: 170px; height: 170px; background: rgba(56,189,248,0.16); bottom: -30px; right: 8%; animation-duration: 11s; animation-delay: -3s; }
.welcome-orb.o3 { width: 130px; height: 130px; background: rgba(245,166,35,0.16); top: 35%; right: 20%; animation-duration: 8s; animation-delay: -5.5s; }
.welcome-orb.o4 { width: 110px; height: 110px; background: rgba(56,189,248,0.12); top: 15%; left: 22%; animation-duration: 10s; animation-delay: -2s; }
@keyframes orbDrift { 0%, 100% { transform: translate(0, 0) scale(1); } 50% { transform: translate(16px, -20px) scale(1.1); } }
.welcome-sparkle { position: absolute; width: 4px; height: 4px; border-radius: 50%; background: #FFD08A; z-index: 0; animation: sparkleTwinkle ease-in-out infinite; }
@keyframes sparkleTwinkle { 0%, 100% { opacity: 0; transform: scale(0.4); } 50% { opacity: 1; transform: scale(1.4); } }
.welcome-wrap > *:not(.welcome-orb):not(.welcome-sparkle) { position: relative; z-index: 1; }
.welcome-title {
    font-family: 'Sora', sans-serif !important; font-size: 1.9rem; font-weight: 800; color: var(--text-hi);
    margin-bottom: 0.6rem; letter-spacing: -0.02em; min-height: 2.4rem;
}
.welcome-sub {
    font-size: 0.95rem; color: var(--text-lo); margin-bottom: 2rem;
    opacity: 0; animation: fadeUp 0.6s ease 2.9s both;
}

/* Typewriter greeting */
.typewriter {
    display: inline-block; overflow: hidden; white-space: nowrap; vertical-align: bottom;
    border-right: 3px solid var(--accent);
    width: 0;
    animation: typing 2s steps(33, end) 1s forwards, caretBlink 0.7s step-end infinite 1s;
}
@keyframes typing { from { width: 0; } to { width: 33ch; } }
@keyframes caretBlink { 50% { border-color: transparent; } }

/* Robot mascot */
.robot-wrap {
    display: flex; justify-content: center; margin-bottom: 0.5rem;
    opacity: 0; animation: robotEnter 0.75s cubic-bezier(.34,1.56,.64,1) 0.05s both;
}
@keyframes robotEnter { 0% { opacity: 0; transform: translateY(-45px) scale(0.5) rotate(-10deg); } 100% { opacity: 1; transform: translateY(0) scale(1) rotate(0); } }
.robot-float { animation: robotFloat 3.2s ease-in-out infinite; transform-origin: center; }
@keyframes robotFloat { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-9px); } }
.robot-eye { animation: robotBlink 4.2s ease-in-out infinite; transform-origin: center; }
.robot-eye.right { animation-delay: 0.05s; }
@keyframes robotBlink { 0%, 90%, 100% { transform: scaleY(1); } 94% { transform: scaleY(0.1); } }
.robot-antenna-dot { animation: blink 1.4s ease infinite; }
.robot-arm-wave { transform-origin: 35px 65px; animation: robotWave 2.1s ease-in-out infinite; }
@keyframes robotWave { 0%, 100% { transform: rotate(0deg); } 20% { transform: rotate(24deg); } 40% { transform: rotate(4deg); } 60% { transform: rotate(24deg); } 80% { transform: rotate(0deg); } }
.robot-glow { animation: robotGlowPulse 2.6s ease-in-out infinite; transform-origin: center; }
@keyframes robotGlowPulse { 0%, 100% { opacity: 0.35; transform: scale(1); } 50% { opacity: 0.65; transform: scale(1.15); } }

/* ── HEADER / NAV BAR ── */
.nav-bar {
    background: linear-gradient(160deg, var(--bg-card) 0%, var(--bg-card-2) 100%);
    border: 1px solid var(--border); border-radius: 14px;
    padding: 0.7rem 1.4rem; margin-bottom: 1.5rem;
}
.nav-bar [data-testid="stHorizontalBlock"] { gap: 0.6rem; }
.nav-bar-greeting { color: var(--text-lo); font-size: 0.9rem; line-height: 1; }

/* ── HERO ── */
.hero-wrap { text-align: center; padding: 4rem 1rem 1.5rem; position: relative; }
.hero-badge {
    display: inline-flex; align-items: center; gap: 0.45rem;
    background: var(--accent-dim);
    border: 1px solid rgba(245,166,35,0.3);
    color: var(--accent);
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase;
    padding: 0.4rem 1.1rem 0.4rem 0.9rem; border-radius: 100px;
    margin-bottom: 1.75rem;
    opacity: 0; animation: fadeUp 0.6s ease 0.05s both;
}
.hero-badge-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); animation: blink 1.8s ease infinite; }
.hero-title {
    font-family: 'Sora', sans-serif !important;
    font-size: 3.7rem; font-weight: 800; color: var(--text-hi);
    line-height: 1.06; letter-spacing: -0.04em; margin-bottom: 1.1rem;
    opacity: 0; animation: fadeUp 0.7s ease 0.15s both;
}
.hero-title .grad {
    background: linear-gradient(100deg, var(--accent) 20%, #FFD08A 45%, var(--accent) 70%);
    background-size: 220% auto;
    -webkit-background-clip: text; background-clip: text; color: transparent;
    animation: sheen 5s linear infinite;
}
@keyframes sheen { to { background-position: -220% center; } }
.hero-desc {
    font-size: 1.06rem; color: var(--text-lo); max-width: 490px; margin: 0 auto 2.5rem; line-height: 1.68;
    opacity: 0; animation: fadeUp 0.7s ease 0.28s both;
}

/* Waveform signature */
.wave { display:flex; align-items:center; justify-content:center; gap:4px; height:34px; margin: 0 auto 2.25rem; opacity:0; animation: fadeUp 0.7s ease 0.4s both; }
.wave span {
    width: 3.5px; border-radius: 3px;
    background: linear-gradient(180deg, var(--accent), rgba(245,166,35,0.25));
    animation: sound 1.6s ease-in-out infinite;
}
.wave span:nth-child(1){height:10px;animation-delay:-1.1s}
.wave span:nth-child(2){height:22px;animation-delay:-0.9s}
.wave span:nth-child(3){height:14px;animation-delay:-1.4s}
.wave span:nth-child(4){height:30px;animation-delay:-0.3s}
.wave span:nth-child(5){height:18px;animation-delay:-0.7s}
.wave span:nth-child(6){height:34px;animation-delay:-1.6s}
.wave span:nth-child(7){height:16px;animation-delay:-0.2s}
.wave span:nth-child(8){height:26px;animation-delay:-1.0s}
.wave span:nth-child(9){height:12px;animation-delay:-0.5s}
.wave span:nth-child(10){height:24px;animation-delay:-1.3s}
.wave span:nth-child(11){height:9px;animation-delay:-0.6s}
@keyframes sound { 0%,100% { transform: scaleY(0.35); opacity:0.6; } 50% { transform: scaleY(1); opacity:1; } }

@keyframes fadeUp { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
@keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }

/* Stats bar */
.stats-bar {
    display: flex; justify-content: center; gap: 2.5rem; padding: 1.2rem 2rem;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; margin-bottom: 2.75rem; position: relative; overflow: hidden;
    opacity: 0; animation: fadeUp 0.7s ease 0.5s both;
}
.stats-bar::before {
    content:''; position:absolute; top:0; left:-30%; width:30%; height:1px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    animation: scan 3.5s ease-in-out infinite;
}
@keyframes scan { 0%{left:-30%;} 100%{left:100%;} }
.stat-item { text-align: center; }
.stat-val { font-family: 'Sora', sans-serif !important; font-size: 1.4rem; font-weight: 700; color: var(--accent); }
.stat-lbl { font-size: 0.72rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 2px; }

/* ── UPLOAD CARD ── */
.upload-icon {
    font-size: 2.6rem; margin-bottom: 0.8rem; text-align:center;
    animation: float 3s ease-in-out infinite;
}
@keyframes float { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-8px); } }
.upload-title { font-family: 'Sora', sans-serif !important; font-size: 1.18rem; font-weight: 700; color: var(--text-hi); text-align:center; margin-bottom: 0.35rem; }
.upload-sub { font-size: 0.85rem; color: var(--text-dim); text-align:center; margin-bottom: 1.4rem; }
.format-pills { display: flex; justify-content: center; gap: 0.5rem; flex-wrap: wrap; margin-top: 1.1rem; }
.pill {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.09);
    color: var(--text-lo); font-size: 0.72rem; font-weight: 500;
    padding: 0.28rem 0.75rem; border-radius: 100px; letter-spacing: 0.06em;
    transition: all 0.25s;
}
.pill:hover { border-color: rgba(245,166,35,0.4); color: var(--accent); transform: translateY(-1px); }

/* ── BUTTON ── */
div.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, #E8941A 100%) !important;
    color: #000 !important; border: none !important; border-radius: 12px !important;
    font-family: 'Sora', sans-serif !important; font-weight: 700 !important; font-size: 1rem !important;
    padding: 0.8rem 2.5rem !important; width: 100% !important; letter-spacing: -0.01em !important;
    transition: all 0.25s cubic-bezier(.2,.8,.2,1) !important;
    box-shadow: 0 4px 24px rgba(245,166,35,0.25) !important;
    position: relative !important; overflow: hidden !important;
}
div.stButton > button:hover { box-shadow: 0 8px 36px rgba(245,166,35,0.5) !important; transform: translateY(-2px) !important; }
div.stButton > button:active { transform: translateY(0) !important; }
div.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.03) !important; color: var(--text-mid) !important;
    border: 1px solid var(--border) !important; box-shadow: none !important;
}
div.stButton > button[kind="secondary"]:hover {
    border-color: rgba(245,166,35,0.4) !important; color: var(--accent) !important;
    box-shadow: 0 4px 20px rgba(245,166,35,0.12) !important; transform: translateY(-2px) !important;
}

/* ── SECTION LABELS ── */
.section-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--accent); margin-bottom: 1rem; margin-top: 2.25rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, rgba(245,166,35,0.2), transparent); }

/* ── METRIC CARDS ── */
.metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; margin-bottom: 1.5rem; }
.m-card {
    background: linear-gradient(160deg, var(--bg-card) 0%, var(--bg-card-2) 100%);
    border: 1px solid var(--border); border-radius: 14px; padding: 1.25rem 1rem; text-align: center;
    transition: border-color 0.3s, transform 0.25s, box-shadow 0.3s;
    opacity: 0; animation: fadeUp 0.5s ease both;
}
.m-card:nth-child(1){animation-delay:.05s} .m-card:nth-child(2){animation-delay:.12s}
.m-card:nth-child(3){animation-delay:.19s} .m-card:nth-child(4){animation-delay:.26s}
.m-card:hover { border-color: rgba(245,166,35,0.35); transform: translateY(-3px); box-shadow: 0 10px 28px rgba(0,0,0,0.35); }
.m-val { font-family: 'Sora', sans-serif !important; font-size: 1.95rem; font-weight: 700; color: var(--accent); line-height: 1; }
.m-lbl { font-size: 0.72rem; color: var(--text-dim); margin-top: 0.4rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.08em; }
.m-hint { font-size: 0.66rem; color: var(--text-faint); margin-top: 0.3rem; }

/* ── ANALYSIS PIPELINE ── */
.pipeline { display: flex; align-items: flex-start; justify-content: center; gap: 0; margin: 2.25rem 0 1.5rem; }
.pipe-step { display: flex; flex-direction: column; align-items: center; gap: 0.6rem; width: 118px; }
.pipe-icon {
    width: 54px; height: 54px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
    font-size: 1.35rem; background: var(--bg-card); border: 2px solid var(--border);
    transition: all 0.4s cubic-bezier(.2,.8,.2,1); position: relative;
}
.pipe-active .pipe-icon { border-color: var(--accent); box-shadow: 0 0 0 7px rgba(245,166,35,0.1); animation: pulseIcon 1.3s ease-in-out infinite; }
.pipe-done .pipe-icon { border-color: var(--success); background: rgba(34,197,94,0.08); }
.pipe-label { font-size: 0.72rem; color: var(--text-dim); text-align: center; line-height: 1.3; transition: color 0.3s; }
.pipe-active .pipe-label { color: var(--accent); font-weight: 600; }
.pipe-done .pipe-label { color: var(--success); }
.pipe-line { flex: 1; height: 2px; background: var(--border); margin-top: 26px; max-width: 55px; transition: background 0.5s; }
.pipe-line-done { background: linear-gradient(90deg, var(--success), var(--success)); }
.pipe-line-active { background: linear-gradient(90deg, var(--success), var(--accent)); }
@keyframes pulseIcon { 0%,100% { transform: scale(1); } 50% { transform: scale(1.08); } }

/* ── SCORE RING CARD ── */
.score-card {
    background: linear-gradient(160deg, var(--bg-card) 0%, var(--bg-card-2) 100%);
    border: 1px solid var(--border); border-radius: 20px; padding: 1.5rem 1.5rem 1rem; text-align: center;
    opacity: 0; animation: fadeUp 0.6s ease 0.1s both;
}
.score-out-of { text-align:center; color: var(--text-dim); font-size: 0.75rem; margin-top: -0.5rem; letter-spacing: 0.05em; text-transform: uppercase; }

/* ── FEEDBACK CARD ── */
.feedback-card {
    background: linear-gradient(160deg, var(--bg-card) 0%, var(--bg-card-2) 100%);
    border: 1px solid var(--border); border-left: 3px solid var(--accent);
    border-radius: 0 16px 16px 0; padding: 1.7rem 1.9rem; font-size: 0.97rem; line-height: 1.8;
    color: #D1D5DB; opacity: 0; animation: fadeUp 0.6s ease both; position: relative;
}
.feedback-card::before {
    content: '"'; position: absolute; top: 0.4rem; left: 1rem; font-family: 'Sora', serif;
    font-size: 3rem; color: rgba(245,166,35,0.18); line-height: 1;
}

/* ── FILLER HIGHLIGHT ── */
.filler-hl {
    background: rgba(239,68,68,0.16); color: #FCA5A5;
    padding: 0.05em 0.35em; border-radius: 5px; font-weight: 600;
    box-decoration-break: clone; -webkit-box-decoration-break: clone;
}

/* ── ERROR CARD ── */
.error-card {
    background: rgba(239,68,68,0.06); border: 1px solid rgba(239,68,68,0.28);
    border-left: 3px solid var(--danger); border-radius: 0 14px 14px 0;
    padding: 1.15rem 1.4rem; margin: 1rem 0 0.5rem;
    opacity: 0; animation: fadeUp 0.4s ease both;
}
.error-title { font-family: 'Sora', sans-serif; font-weight: 700; color: #FCA5A5; font-size: 0.95rem; margin-bottom: 0.35rem; }
.error-msg { color: var(--text-mid); font-size: 0.88rem; line-height: 1.6; }
.error-hint { color: var(--text-dim); font-size: 0.78rem; margin-top: 0.55rem; }

/* ── CHAT ── */
.chat-header-wrap {
    background: linear-gradient(160deg, var(--bg-card) 0%, var(--bg-card-2) 100%);
    border: 1px solid var(--border); border-radius: 16px 16px 0 0; padding: 1rem 1.5rem;
    display: flex; align-items: center; gap: 0.85rem;
}
.chat-avatar {
    position: relative; flex-shrink: 0;
    width: 40px; height: 40px; border-radius: 50%;
    background: linear-gradient(135deg, rgba(245,166,35,0.22), rgba(245,166,35,0.08));
    border: 1px solid rgba(245,166,35,0.35);
    display: flex; align-items: center; justify-content: center; font-size: 1.15rem;
}
.chat-dot {
    position: absolute; bottom: -1px; right: -1px;
    width: 10px; height: 10px; border-radius: 50%; background: var(--success);
    border: 2px solid var(--bg-card); box-shadow: 0 0 6px var(--success); animation: blink 2s ease infinite;
}
.chat-title-text { font-family: 'Sora', sans-serif !important; font-size: 0.98rem; font-weight: 700; color: var(--text-hi); }
.chat-sub-text { font-size: 0.78rem; color: var(--text-lo); margin-left: auto; text-align: right; }

[data-testid="stChatMessage"] {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: 12px !important; margin-bottom: 0.5rem !important;
    transition: border-color 0.3s, transform 0.2s;
    opacity: 0; animation: fadeUp 0.35s ease both;
}
[data-testid="stChatMessage"]:hover { border-color: rgba(245,166,35,0.25) !important; }

/* Streamlit's default markdown text color assumes a light page background, so
   chat replies were rendering as faint, washed-out gray on our dark theme —
   force full-contrast text explicitly inside every chat bubble. */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    color: var(--text-hi) !important;
    font-size: 0.95rem !important;
    line-height: 1.65 !important;
}

/* Chat avatars — recolor Streamlit's default user/assistant icon badges to fit the
   dark + amber theme instead of the default red/orange. */
[data-testid="stChatMessageAvatarUser"],
[data-testid*="ChatMessageAvatar"][data-testid*="User"] {
    background: linear-gradient(135deg, var(--accent-2), #0EA5E9) !important;
}
[data-testid="stChatMessageAvatarAssistant"],
[data-testid*="ChatMessageAvatar"][data-testid*="Assistant"] {
    background: linear-gradient(135deg, var(--accent), #E8941A) !important;
}
[data-testid*="ChatMessageAvatar"] { color: #000 !important; flex-shrink: 0 !important; }
[data-testid*="ChatMessageAvatar"] [data-testid="stIconMaterial"] { font-size: 18px !important; }

/* Expander */
[data-testid="stExpander"] {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: 12px !important;
    transition: border-color 0.25s !important; overflow: hidden !important;
}
[data-testid="stExpander"]:hover { border-color: rgba(245,166,35,0.3) !important; }
[data-testid="stExpander"] summary {
    color: var(--text-lo) !important; font-size: 0.85rem !important; font-weight: 500 !important;
    transition: color 0.2s !important;
}
[data-testid="stExpander"] summary:hover { color: var(--text-hi) !important; }
[data-testid="stExpander"] summary [data-testid="stIconMaterial"] {
    color: var(--accent) !important; transition: transform 0.3s ease !important;
}
[data-testid="stExpander"] details[open] summary [data-testid="stIconMaterial"] { transform: rotate(90deg) !important; }

/* File uploader */
[data-testid="stFileUploader"] { background: transparent !important; border: none !important; }
[data-testid="stFileUploader"] > div { background: transparent !important; border: none !important; }

/* The widget's a11y label ("Upload") sits behind the Browse button and was bleeding
   through — force it fully out of the layout instead of relying on label_visibility alone.
   Multiple redundant properties here on purpose: different Streamlit versions hide this
   element in different ways, so one override alone isn't always enough. */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] [data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    width: 0 !important;
    opacity: 0 !important;
    overflow: hidden !important;
    position: absolute !important;
    pointer-events: none !important;
}

/* Streamlit's own "Drag and drop file here / Limit 200MB..." copy duplicated our
   custom heading + pills above, and was the other half of the overlap — hide it
   and keep just a clean, centered Browse button. */
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzoneInstructions"] * {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
}

section[data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(160deg, rgba(255,255,255,0.025) 0%, rgba(255,255,255,0.01) 100%) !important;
    border: 1.5px dashed rgba(245,166,35,0.35) !important; border-radius: 16px !important;
    transition: all 0.3s !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    min-height: 78px !important; padding: 1rem !important;
}
section[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(245,166,35,0.75) !important; background: rgba(245,166,35,0.035) !important;
}
section[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, rgba(245,166,35,0.24), rgba(245,166,35,0.13)) !important;
    color: #FFD9A0 !important;
    border: 1px solid rgba(245,166,35,0.5) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 0.65rem 1.6rem !important;
    transition: all 0.25s !important;
}
section[data-testid="stFileUploaderDropzone"] button p {
    color: #FFD9A0 !important; margin: 0 !important; font-weight: 600 !important;
}
section[data-testid="stFileUploaderDropzone"] button:hover {
    background: linear-gradient(135deg, rgba(245,166,35,0.34), rgba(245,166,35,0.19)) !important;
    border-color: rgba(245,166,35,0.75) !important;
    transform: translateY(-1px) !important;
}

/* Success */
[data-testid="stAlert"] {
    background: rgba(34, 197, 94, 0.08) !important; border: 1px solid rgba(34, 197, 94, 0.25) !important;
    border-radius: 10px !important; color: #86EFAC !important;
    opacity: 0; animation: fadeUp 0.5s ease both;
}

/* Audio player */
audio { width: 100%; border-radius: 10px; margin: 0.5rem 0; }

/* Spinner */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* hr */
hr { border-color: var(--border) !important; margin: 2rem 0 !important; }

/* Chat input — floating footer container Streamlit renders outside the normal
   themed page flow, so it stays stark white unless re-themed explicitly. */
[data-testid="stBottom"] {
    background: var(--bg) !important;
    border-top: 1px solid var(--border) !important;
    width: 100% !important;
    left: 0 !important;
    right: 0 !important;
}
[data-testid="stBottom"] > div { background: var(--bg) !important; }
[data-testid="stBottomBlockContainer"] {
    background: var(--bg) !important;
    padding: 1rem 2rem 1.25rem !important; max-width: 900px !important; margin: auto !important;
}

/* The widget wraps the textarea in several nested divs that carry Streamlit's own
   light-theme background — that's what made typed text (colored near-white for our
   dark theme) invisible against a white box underneath it. Force every layer
   transparent so only our own pill background + border show through. */
[data-testid="stChatInput"] * {
    background-color: transparent !important;
    box-shadow: none !important;
    outline: none !important;
}
[data-testid="stChatInput"] {
    background: var(--bg-card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 18px !important;
    padding: 0.2rem 0.2rem 0.2rem 0.6rem !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(245,166,35,0.55) !important;
    box-shadow: 0 0 0 4px rgba(245,166,35,0.12) !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--text-hi) !important;
    caret-color: var(--accent) !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--text-dim) !important; }
[data-testid="stChatInput"] button {
    background: var(--accent) !important;
    border-radius: 50% !important;
    width: 34px !important; height: 34px !important; min-width: 34px !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    transition: transform 0.2s, background 0.2s !important;
}
[data-testid="stChatInput"] button:hover { background: #E8941A !important; transform: scale(1.08) !important; }
[data-testid="stChatInput"] button:disabled { background: var(--border) !important; }
[data-testid="stChatInput"] button svg { fill: #000 !important; }

/* Typing indicator */
.typing-dots { display: inline-flex; align-items: center; gap: 4px; padding: 0.2rem 0; }
.typing-dots span {
    width: 7px; height: 7px; border-radius: 50%; background: var(--accent);
    animation: typingBounce 1.2s ease-in-out infinite;
}
.typing-dots span:nth-child(2) { animation-delay: 0.15s; }
.typing-dots span:nth-child(3) { animation-delay: 0.3s; }
@keyframes typingBounce { 0%, 60%, 100% { transform: translateY(0); opacity: 0.5; } 30% { transform: translateY(-5px); opacity: 1; } }

/* ── TEXT INPUT (name gate) ── */
[data-testid="stTextInput"] * { background-color: transparent !important; box-shadow: none !important; }
[data-testid="stTextInput"] > div {
    background: var(--bg-card) !important; border: 1.5px solid var(--border) !important;
    border-radius: 12px !important; transition: border-color 0.25s !important;
}
[data-testid="stTextInput"] > div:focus-within { border-color: rgba(245,166,35,0.55) !important; box-shadow: 0 0 0 3px rgba(245,166,35,0.12) !important; }
[data-testid="stTextInput"] input {
    color: var(--text-hi) !important; caret-color: var(--accent) !important;
    font-size: 0.95rem !important; padding: 0.65rem 0.9rem !important;
}
[data-testid="stTextInput"] input::placeholder { color: var(--text-dim) !important; }

/* ── RESPONSIVE ── */
@media (max-width: 768px) {
    .block-container { padding: 0 1.1rem 3rem !important; }
    .hero-wrap { padding: 2.75rem 0.5rem 1.25rem; }
    .hero-title { font-size: 2.5rem !important; }
    .hero-desc { font-size: 0.95rem !important; padding: 0 0.25rem; }
    .stats-bar { flex-wrap: wrap; gap: 1.25rem 1.75rem; padding: 1.1rem 1.25rem; }
    .metrics-grid { grid-template-columns: repeat(2, 1fr) !important; }
    .pipeline { flex-wrap: wrap; row-gap: 1.25rem; }
    .pipe-step { width: 30%; }
    .pipe-line { display: none; }
    .chat-header-wrap { flex-wrap: wrap; row-gap: 0.5rem; }
    .chat-sub-text { margin-left: 0; text-align: left; width: 100%; }
    .wave { transform: scale(0.85); }
    [data-testid="stBottomBlockContainer"] { padding: 0.85rem 1.1rem 1rem !important; }
}
@media (max-width: 480px) {
    .hero-title { font-size: 2.05rem !important; }
    .hero-badge { font-size: 0.64rem; padding: 0.32rem 0.8rem; }
    .stats-bar { gap: 1rem 1.25rem; }
    .stat-val { font-size: 1.15rem; }
    .metrics-grid { grid-template-columns: repeat(2, 1fr) !important; gap: 0.6rem; }
    .m-val { font-size: 1.6rem; }
    .pipe-step { width: 45%; }
    .feedback-card { padding: 1.3rem 1.4rem; }
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──
for key in ["transcript", "metrics", "feedback", "messages", "analysed"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "messages" else (False if key == "analysed" else None)
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "view" not in st.session_state:
    st.session_state.view = "analyze"


def score_ring_component(score: int, color: str) -> str:
    """Animated SVG confidence ring with a count-up number, rendered via an embedded script."""
    r = 80
    circumference = 2 * 3.14159265 * r
    offset = circumference * (1 - score / 100)
    return f"""
    <div style="display:flex;justify-content:center;align-items:center;">
      <svg width="210" height="210" viewBox="0 0 200 200">
        <circle cx="100" cy="100" r="{r}" fill="none" stroke="#1F2937" stroke-width="14"/>
        <circle id="ring" cx="100" cy="100" r="{r}" fill="none" stroke="{color}" stroke-width="14"
          stroke-linecap="round" stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{circumference:.2f}"
          transform="rotate(-90 100 100)"
          style="transition: stroke-dashoffset 1.5s cubic-bezier(.16,1,.3,1); filter: drop-shadow(0 0 6px {color}66);"/>
        <text id="scoreText" x="100" y="112" text-anchor="middle" font-family="Sora, sans-serif"
          font-weight="700" font-size="46" fill="{color}">0</text>
      </svg>
    </div>
    <script>
      const ring = document.getElementById('ring');
      const text = document.getElementById('scoreText');
      const target = {score};
      setTimeout(() => {{ ring.style.strokeDashoffset = '{offset:.2f}'; }}, 120);
      let cur = 0;
      const duration = 1400, start = performance.now();
      function tick(now) {{
        const p = Math.min(1, (now - start) / duration);
        const eased = 1 - Math.pow(1 - p, 3);
        cur = Math.round(eased * target);
        text.textContent = cur;
        if (p < 1) requestAnimationFrame(tick);
      }}
      requestAnimationFrame(tick);
    </script>
    """


def render_pipeline(states) -> str:
    steps = [("🎧", "Transcribing audio"), ("📊", "Analyzing patterns"), ("🧠", "Generating feedback")]
    out = '<div class="pipeline">'
    for i, ((icon, label), state) in enumerate(zip(steps, states)):
        icon_display = "✓" if state == "done" else icon
        out += f'<div class="pipe-step pipe-{state}"><div class="pipe-icon">{icon_display}</div><div class="pipe-label">{label}</div></div>'
        if i < len(steps) - 1:
            line_state = "done" if states[i] == "done" else ("active" if state in ("active", "done") else "")
            out += f'<div class="pipe-line pipe-line-{line_state}"></div>'
    out += '</div>'
    return out


def copy_button_html(text: str) -> str:
    """Small self-contained copy-to-clipboard button. Needs components.html (not
    st.markdown) since the click handler is real JS and markdown scripts aren't
    reliably executed."""
    payload = json.dumps(text)
    return f"""
    <button id="copyBtn" style="
        width:100%; background:rgba(245,166,35,0.14); color:#FFD9A0;
        border:1px solid rgba(245,166,35,0.4); border-radius:10px;
        padding:0.6rem 1rem; font-family:'Inter',sans-serif; font-size:0.85rem;
        font-weight:600; cursor:pointer; transition:transform .2s, background .2s;">
        📋 Copy feedback
    </button>
    <script>
      const btn = document.getElementById('copyBtn');
      const text = {payload};
      btn.addEventListener('click', async () => {{
        try {{
          await navigator.clipboard.writeText(text);
          btn.textContent = '✓ Copied!';
          btn.style.background = 'rgba(34,197,94,0.16)';
          btn.style.color = '#86EFAC';
          setTimeout(() => {{
            btn.textContent = '📋 Copy feedback';
            btn.style.background = 'rgba(245,166,35,0.14)';
            btn.style.color = '#FFD9A0';
          }}, 1800);
        }} catch (e) {{
          btn.textContent = 'Copy failed — select manually';
        }}
      }});
      btn.addEventListener('mouseenter', () => {{ btn.style.transform = 'translateY(-1px)'; }});
      btn.addEventListener('mouseleave', () => {{ btn.style.transform = 'translateY(0)'; }});
    </script>
    """


def highlight_fillers(text: str, filler_words) -> str:
    """HTML-escape the transcript, then wrap each filler word/phrase in a highlight
    span so users can see exactly where fillers land instead of only a total count."""
    escaped = html.escape(text or "")
    for w in sorted({w for w in filler_words if w}, key=len, reverse=True):
        pattern = re.compile(rf'(?<!\w)({re.escape(html.escape(w))})(?!\w)', re.IGNORECASE)
        escaped = pattern.sub(r'<mark class="filler-hl">\1</mark>', escaped)
    return escaped


def render_history_page():
    st.markdown('<div class="section-label" style="margin-top:1rem;">Your Session History</div>', unsafe_allow_html=True)
    rows = get_user_history(st.session_state.user_name)

    if not rows:
        st.markdown(
            '<div class="upload-sub" style="margin-top:0.5rem;">No past sessions yet — analyse a recording and it\'ll show up here.</div>',
            unsafe_allow_html=True,
        )
        return

    if len(rows) > 1:
        chron = list(reversed(rows))
        fig = px.line(
            x=[r["created_at"].replace("T", " ")[5:16] for r in chron],
            y=[r["confidence_score"] for r in chron],
            markers=True, labels={"x": "", "y": "Score"},
        )
        fig.update_traces(line_color="#F5A623", marker=dict(color="#F5A623", size=8))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8A93A3", family="Inter", size=12),
            margin=dict(l=0, r=0, t=10, b=0), height=220,
            xaxis=dict(gridcolor="#1F2937"), yaxis=dict(gridcolor="#1F2937", range=[0, 100]),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    for r in rows:
        dt = r["created_at"].replace("T", " ")[:16]
        score = r["confidence_score"] or 0
        color = "#22C55E" if score >= 75 else "#F5A623" if score >= 50 else "#EF4444"
        wpm_val = r["wpm"] or 0
        st.markdown(f"""
        <div class="m-card" style="text-align:left; display:flex; align-items:center; justify-content:space-between; padding:1rem 1.3rem; opacity:1; animation:none;">
            <div>
                <div style="color:var(--text-hi); font-weight:600; font-size:0.9rem;">{html.escape(r['filename'] or 'Recording')}</div>
                <div style="color:var(--text-dim); font-size:0.76rem; margin-top:2px;">{dt} · {wpm_val:.0f} WPM · {r['total_fillers']} fillers</div>
            </div>
            <div style="font-family:'Sora',sans-serif; font-weight:700; font-size:1.3rem; color:{color};">{score}</div>
        </div>
        <div style="height:0.55rem;"></div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️  Clear my history", type="secondary"):
        delete_user_history(st.session_state.user_name)
        st.rerun()


# ══════════════════
# NAME GATE
# ══════════════════
if not st.session_state.user_name:
    st.markdown("""
    <div class="welcome-wrap">
        <div class="welcome-orb o1"></div>
        <div class="welcome-orb o2"></div>
        <div class="welcome-orb o3"></div>
        <div class="welcome-orb o4"></div>
        <div class="welcome-sparkle" style="top:18%; left:15%; animation-duration:2.6s; animation-delay:0s;"></div>
        <div class="welcome-sparkle" style="top:28%; left:78%; animation-duration:3.2s; animation-delay:-1s;"></div>
        <div class="welcome-sparkle" style="top:60%; left:10%; animation-duration:2.9s; animation-delay:-2s;"></div>
        <div class="welcome-sparkle" style="top:12%; left:60%; animation-duration:3.4s; animation-delay:-0.6s;"></div>
        <div class="welcome-sparkle" style="top:70%; left:85%; animation-duration:2.4s; animation-delay:-1.6s;"></div>
        <div class="welcome-sparkle" style="top:45%; left:92%; animation-duration:3s; animation-delay:-2.4s;"></div>
        <div class="robot-wrap">
            <svg width="150" height="150" viewBox="0 0 150 150" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <radialGradient id="robotGlow" cx="50%" cy="50%" r="50%">
                        <stop offset="0%" stop-color="rgba(245,166,35,0.4)"/>
                        <stop offset="100%" stop-color="rgba(245,166,35,0)"/>
                    </radialGradient>
                </defs>
                <circle cx="75" cy="78" r="58" fill="url(#robotGlow)" class="robot-glow"/>
                <g class="robot-float">
                    <line x1="75" y1="24" x2="75" y2="40" stroke="#F5A623" stroke-width="3" stroke-linecap="round"/>
                    <circle cx="75" cy="19" r="6" fill="#F5A623" class="robot-antenna-dot"/>
                    <rect x="33" y="40" width="84" height="66" rx="24" fill="#1F2937" stroke="#F5A623" stroke-width="2"/>
                    <circle cx="58" cy="71" r="7" fill="#F5A623" class="robot-eye"/>
                    <circle cx="93" cy="71" r="7" fill="#F5A623" class="robot-eye right"/>
                    <path d="M58 89 Q75.5 100 93 89" stroke="#F5A623" stroke-width="3.5" fill="none" stroke-linecap="round"/>
                    <line x1="35" y1="65" x2="14" y2="43" stroke="#374151" stroke-width="9" stroke-linecap="round" class="robot-arm-wave"/>
                    <circle cx="14" cy="43" r="6.5" fill="#F5A623" class="robot-arm-wave"/>
                    <line x1="115" y1="70" x2="133" y2="92" stroke="#374151" stroke-width="9" stroke-linecap="round"/>
                    <circle cx="133" cy="92" r="6.5" fill="#374151"/>
                </g>
            </svg>
        </div>
        <div class="welcome-title"><span class="typewriter">Hi! I'm Spark, your speech coach.</span></div>
        <div class="welcome-sub">Tell me your name and let's get started — I'll keep track of your progress every time you practice.</div>
    </div>
    """, unsafe_allow_html=True)

    gate_col1, gate_col2, gate_col3 = st.columns([1, 1.4, 1])
    with gate_col2:
        name_input = st.text_input("Your name", placeholder="e.g. Priya", label_visibility="collapsed")
        if st.button("Continue  →", use_container_width=True):
            if name_input.strip():
                st.session_state.user_name = name_input.strip()
                st.rerun()
            else:
                st.markdown('<div style="text-align:center;color:#FCA5A5;font-size:0.85rem;margin-top:0.6rem;">Please enter a name to continue.</div>', unsafe_allow_html=True)
    st.stop()

# ══════════════════
# NAV
# ══════════════════
st.markdown('<div class="nav-bar">', unsafe_allow_html=True)
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([3, 1.3, 1.3, 1.2], vertical_alignment="center")
with nav_col1:
    st.markdown(f'<div class="nav-bar-greeting">👋 Hi, <b style="color:var(--text-hi)">{html.escape(st.session_state.user_name)}</b></div>', unsafe_allow_html=True)
with nav_col2:
    if st.button("🎤 New Analysis", type=("primary" if st.session_state.view == "analyze" else "secondary"), use_container_width=True):
        st.session_state.view = "analyze"
        st.rerun()
with nav_col3:
    if st.button("📜 History", type=("primary" if st.session_state.view == "history" else "secondary"), use_container_width=True):
        st.session_state.view = "history"
        st.rerun()
with nav_col4:
    if st.button("Switch user", type="secondary", use_container_width=True):
        st.session_state.user_name = None
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.view == "history":
    render_history_page()
    st.stop()

# ══════════════════
# HERO
# ══════════════════
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge"><span class="hero-badge-dot"></span> AI-Powered Speech Analysis</div>
    <div class="hero-title">Speak better,<br><span class="grad">every time.</span></div>
    <div class="hero-desc">Upload any recording and get instant, personalised coaching — filler word analysis, pacing feedback, and an AI coach ready to chat.</div>
    <div class="wave">
        <span></span><span></span><span></span><span></span><span></span><span></span>
        <span></span><span></span><span></span><span></span><span></span>
    </div>
</div>
""", unsafe_allow_html=True)

# Stats bar
st.markdown("""
<div class="stats-bar">
    <div class="stat-item"><div class="stat-val">3</div><div class="stat-lbl">AI Stages</div></div>
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
        "Upload", type=["mp3", "wav", "mp4", "m4a", "mov"],
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}"
    )

    st.markdown("""
    <div class="format-pills">
        <span class="pill">🎵 MP3</span><span class="pill">🎵 WAV</span>
        <span class="pill">🎬 MP4</span><span class="pill">🎵 M4A</span><span class="pill">🎬 MOV</span>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file:
        size_mb = uploaded_file.size / (1024 * 1024)
        too_large = size_mb > 200

        if too_large:
            st.markdown(f"""
            <div class="error-card">
                <div class="error-title">⚠️  File too large</div>
                <div class="error-msg">This file is {size_mb:.0f}MB — the limit is 200MB.</div>
                <div class="error-hint">Trim or compress the recording, then upload it again.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<br>", unsafe_allow_html=True)
            st.audio(uploaded_file)
            st.markdown("<br>", unsafe_allow_html=True)

        if not too_large and st.button("✨  Analyse my speech"):
            pipeline_slot = st.empty()
            error_slot = st.empty()
            tmp_path = None

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                pipeline_slot.markdown(render_pipeline(["active", "pending", "pending"]), unsafe_allow_html=True)
                try:
                    transcript = transcribe_audio(tmp_path)
                except Exception as e:
                    raise RuntimeError(f"Couldn't transcribe that recording — {e}") from e

                pipeline_slot.markdown(render_pipeline(["done", "active", "pending"]), unsafe_allow_html=True)
                try:
                    metrics = analyze_metrics(transcript)
                except Exception as e:
                    raise RuntimeError(f"Couldn't analyse the speech patterns — {e}") from e

                pipeline_slot.markdown(render_pipeline(["done", "done", "active"]), unsafe_allow_html=True)
                try:
                    feedback = get_coaching_feedback(transcript, metrics)
                except Exception as e:
                    raise RuntimeError(f"Couldn't generate coaching feedback — {e}") from e

                pipeline_slot.markdown(render_pipeline(["done", "done", "done"]), unsafe_allow_html=True)
                time.sleep(0.5)

                try:
                    save_session_to_history(st.session_state.user_name, uploaded_file.name, metrics)
                except Exception:
                    pass  # history is a nice-to-have — never let it block showing results

                st.session_state.transcript = transcript
                st.session_state.metrics = metrics
                st.session_state.feedback = feedback
                st.session_state.analysed = True
                st.rerun()

            except Exception as e:
                pipeline_slot.empty()
                error_slot.markdown(f"""
                <div class="error-card">
                    <div class="error-title">⚠️  Analysis didn't complete</div>
                    <div class="error-msg">{html.escape(str(e))}</div>
                    <div class="error-hint">Check the file format and your connection, then try "Analyse my speech" again — nothing was saved, so it's safe to retry.</div>
                </div>
                """, unsafe_allow_html=True)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

# ══════════════════
# RESULTS
# ══════════════════
if st.session_state.analysed:
    transcript = st.session_state.transcript
    metrics = st.session_state.metrics
    feedback = st.session_state.feedback
    score = metrics["confidence_score"]

    st.success("✅  Analysis complete — your coaching report is ready.")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Score + Metrics ──
    col_score, col_right = st.columns([1, 2], gap="large")

    with col_score:
        st.markdown('<div class="section-label">Confidence Score</div>', unsafe_allow_html=True)
        ring_color = "#22C55E" if score >= 75 else "#F5A623" if score >= 50 else "#EF4444"
        st.markdown('<div class="score-card">', unsafe_allow_html=True)
        components.html(score_ring_component(score, ring_color), height=220)
        st.markdown('<div class="score-out-of">out of 100</div></div>', unsafe_allow_html=True)

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
            labels={"x": "", "y": "Count"},
            color=list(metrics["filler_breakdown"].values()),
            color_continuous_scale=[[0, "#1F2937"], [0.5, "#D97706"], [1, "#F5A623"]],
            text=list(metrics["filler_breakdown"].values()),
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#6B7280", family="Inter", size=12),
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0), height=240,
            xaxis=dict(gridcolor="#1F2937", tickfont=dict(color="#9CA3AF", size=13)),
            yaxis=dict(gridcolor="#1F2937", tickfont=dict(color="#6B7280")),
            transition=dict(duration=500, easing="cubic-in-out"),
        )
        fig_bar.update_traces(
            marker_line_width=0,
            textposition="outside",
            textfont=dict(color="#F5A623", size=13),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    # ── Transcript ──
    with st.expander("📄  View full transcript"):
        filler_words = list(metrics.get("filler_breakdown", {}).keys())
        highlighted_text = highlight_fillers(transcript["text"], filler_words)
        st.markdown(f'<div style="color:#B4BCC8;font-size:0.92rem;line-height:1.9;padding:0.5rem 0">{highlighted_text}</div>', unsafe_allow_html=True)
        if filler_words:
            st.markdown('<div style="font-size:0.74rem;color:var(--text-dim);margin-top:0.4rem;">🔴 highlighted words are the fillers counted above</div>', unsafe_allow_html=True)

    # ── Coaching Feedback ──
    st.markdown('<div class="section-label">Coaching Feedback</div>', unsafe_allow_html=True)

    with st.expander("🔍  Show draft (before self-review)"):
        st.markdown(f'<div style="color:#8A93A3;font-size:0.88rem;line-height:1.75">{feedback["draft"]}</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="feedback-card">{feedback["final"]}</div>', unsafe_allow_html=True)

    fb_col1, fb_col2 = st.columns([1, 1])
    with fb_col1:
        if st.button("🔄  Regenerate feedback", type="secondary"):
            try:
                with st.spinner("Getting a fresh take..."):
                    st.session_state.feedback = get_coaching_feedback(transcript, metrics)
                st.rerun()
            except Exception as e:
                st.markdown(f"""
                <div class="error-card" style="margin-top:0.75rem;">
                    <div class="error-title">⚠️  Couldn't regenerate feedback</div>
                    <div class="error-msg">{html.escape(str(e))}</div>
                </div>
                """, unsafe_allow_html=True)
    with fb_col2:
        components.html(copy_button_html(feedback["final"]), height=48)

    # ── Chat ──
    st.markdown("---")
    st.markdown("""
    <div class="chat-header-wrap">
        <div class="chat-avatar">🎙️<div class="chat-dot"></div></div>
        <div>
            <div class="chat-title-text">Your AI speech coach</div>
            <div style="font-size:0.74rem;color:var(--text-dim);margin-top:1px;">Online · powered by Llama 3.3</div>
        </div>
        <div class="chat-sub-text">Remembers your<br>full session</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div style="color:#8A93A3;font-size:0.82rem;margin:0.75rem 0 1rem">Your coach has full context of your speech and feedback. Ask anything — it remembers the whole conversation.</div>', unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if user_question := st.chat_input("How can I reduce my filler words?  •  What was my strongest moment?"):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

        system = f"""You are a warm, expert communication coach in an ongoing coaching session.

Full speech context:
- Transcript: {transcript['text']}
- WPM: {metrics['wpm']} (ideal: 120-150), Fillers: {metrics['filler_breakdown']}, Score: {metrics['confidence_score']}/100
- Coaching feedback already given: {feedback['final']}

Be conversational, specific, and encouraging. Reference the actual transcript when relevant.
Keep responses concise — 2-4 sentences unless more detail is genuinely needed."""

        messages_for_api = [{"role": "system", "content": system}]
        for msg in st.session_state.messages:
            messages_for_api.append({"role": msg["role"], "content": msg["content"]})

        with st.chat_message("assistant"):
            typing_slot = st.empty()
            typing_slot.markdown('<div class="typing-dots"><span></span><span></span><span></span></div>', unsafe_allow_html=True)

            try:
                groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages_for_api,
                    max_tokens=500,
                    temperature=0.7
                )
                reply = response.choices[0].message.content.strip()
                typing_slot.write(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                typing_slot.markdown(f"""
                <div class="error-card" style="margin:0;">
                    <div class="error-title">⚠️  Couldn't reach the coach</div>
                    <div class="error-msg">{html.escape(str(e))}</div>
                    <div class="error-hint">Check your connection and try sending that again.</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("↩  Analyse a new recording", type="secondary"):
        for key in ["transcript", "metrics", "feedback", "messages", "analysed"]:
            del st.session_state[key]
        st.session_state.uploader_key += 1
        st.rerun()
