# VoiceCoach AI — AI Communication Coach

An end-to-end agentic AI pipeline that transcribes speech, analyses filler words and pacing, detects naturalness and authenticity, and delivers personalised coaching feedback — with a live chat coach and session history tracking.

Built as an MSc Big Data Analytics course project, satisfying the agentic AI systems requirement while producing a portfolio-ready, deployable application.

---

## Overview

Upload any audio or video recording of yourself speaking. The application transcribes it, analyses your delivery across multiple dimensions, and returns a structured coaching report within approximately two minutes.

**What you get:**

- Confidence score, words per minute, filler word breakdown with inline transcript highlighting
- Naturalness score — five signals indicating whether speech sounds spontaneous or scripted
- Authenticity indicators — five signals distinguishing human speech from AI-generated audio
- Personalised coaching feedback generated and self-reviewed by an agentic LLM loop
- Live chat coach with full session memory
- Session history with progress trend charts across multiple recordings
- Exploratory data analysis notebook with eight sections and ten charts

---

## Architecture

The application is built as a four-stage pipeline. Each stage has a single responsibility and a clean input/output contract.

```
Audio / Video file
        |
        v
Stage 1 — src/transcribe.py
    Whisper extracts audio via ffmpeg
    Converts speech to structured JSON
    Output: full text + timestamps per segment + total duration
        |
        v
Stage 2 — src/metrics.py
    Original Python logic — no AI involved
    Counts filler words, calculates WPM
    Scores naturalness across five linguistic signals
    Scores authenticity across five human speech indicators
        |
        v
Stage 3 — src/coach.py  [Agentic Loop]
    Call 1: Coach persona drafts feedback
    Call 2: Critic persona reviews the draft
            Is it specific? Encouraging? Actionable?
            Rewrites if not — returns unchanged if all three pass
    Final reviewed feedback passed downstream
        |
        v
Stage 4 — app.py
    Streamlit web application
    Named user sessions with welcome screen
    Navigation: New Analysis and History views
    Results: confidence score ring, metrics cards, filler chart,
             highlighted transcript, coaching feedback
    Live chat coach with full conversation memory
    SQLite session history with progress trend charts
```

---

## Agentic Design

Most AI tools call a model once and display the result. This application uses a two-call self-review loop in Stage 3:

- **Call 1 — Coach persona:** Generates two to three specific, encouraging, actionable feedback points grounded in the actual transcript and metrics.
- **Call 2 — Critic persona:** Evaluates the draft against three criteria — specificity, tone, and actionability. Rewrites the feedback if any criterion is not met. Returns it unchanged if all three pass.

The system evaluates and acts on its own output before showing it to the user. That decision loop is the agentic element. Both the draft and the final reviewed version are visible in the application for transparency.

---

## Naturalness Detection

Analyses whether speech sounds spontaneous or scripted using five signals extracted entirely from the transcript. No additional API call is required — all signals are computed from the Whisper JSON output.

| Signal | What it measures | Natural benchmark |
|---|---|---|
| Pace variance | Standard deviation of WPM across segments | Greater than 15 |
| Filler rate | Filler words as a percentage of total words | 3 to 8 percent |
| Natural markers | Phrases such as I mean, you know, I think | At least one found |
| Sentence length variance | Standard deviation of sentence lengths | Higher indicates natural |
| Word repetition rate | Percentage of words that are repeated | Greater than 30 percent |

Output: Spontaneous / Partially Scripted / Likely Scripted with a score from 0 to 100.

---

## Authenticity Detection

Shows five indicators that distinguish human speech from AI-generated or synthetic audio. This does not produce a binary verdict — it surfaces the signals and lets the user interpret them.

| Indicator | Human signal |
|---|---|
| Pace variation across segments | Standard deviation greater than 15 WPM |
| Filler words present | At least one filler word detected |
| Self-corrections and restarts | Mid-speech corrections or immediate word repetitions |
| Sentence completeness | Presence of short or incomplete sentences |
| Natural vocabulary pattern | Greater than 25 percent common everyday words |

No text-based detection system can claim 100 percent accuracy. Even widely used tools such as GPTZero and Turnitin report false positive rates of 20 to 30 percent. This feature is presented as honest indicators, not a definitive verdict.

---

## Exploratory Data Analysis

The notebook at `notebooks/VoiceCoach_EDA.ipynb` performs analysis on both the transcript data and session history. It was run on a real student presentation recording and includes eight sections with ten charts.

| Section | Content |
|---|---|
| 1. Setup and Data Loading | Loads transcript JSON and SQLite session history |
| 2. Transcript Overview | Word count, vocabulary richness, unique words, top 20 content words bar chart |
| 3. Filler Word Analysis | Distribution pie and bar charts, timeline scatter showing filler positions across the recording |
| 4. Segment-Level Pacing | WPM over time with ideal zone, filler count per segment, WPM histogram and box plot |
| 5. Session History | Progress line charts for confidence score, WPM, and filler count across sessions |
| 6. Naturalness Signals | Radar chart of all five naturalness signals with benchmarks |
| 7. Summary | Full findings printout with pass/fail indicators per metric |
| 8. Conclusions | Written analysis connecting all findings, with top recommendation |

**Sample findings from the EDA on a real presentation recording:**

- Speaking pace: 125 WPM — within the ideal 120 to 150 range
- Top filler word: like, used 10 times across the recording
- Pace variance (std dev): 29.4 WPM — consistent with natural, spontaneous speech
- Naturalness score: 84 out of 100 — classified as Spontaneous
- Authenticity: Strong Human Indicators — 4 out of 5 signals matched human speech patterns

To run the notebook:

```bash
# Make sure the venv is active
jupyter notebook notebooks/VoiceCoach_EDA.ipynb
```

Or open it in VS Code and select the project virtual environment as the kernel.

---

## Tech Stack

| Stage | File | Tool | Type | Internet required | Cost |
|---|---|---|---|---|---|
| Transcription | src/transcribe.py | OpenAI Whisper base model | Local library | First download only | Free |
| Audio decoding | automatic | ffmpeg | System tool | No | Free |
| Metrics and detection | src/metrics.py | Pure Python | Original code | No | Free |
| Coaching feedback | src/coach.py | Groq API — Llama 3.3 70B | API | Yes | Free tier |
| Web interface | app.py | Streamlit | Local library | No | Free |
| Charts | app.py | Plotly | Local library | No | Free |
| Session history | app.py | SQLite | Built-in Python | No | Free |
| EDA | notebooks/ | Pandas, Matplotlib, Seaborn | Local libraries | No | Free |

Only the Groq API requires an internet connection. Every other stage runs entirely on the local machine.

---

## Project Structure

```
ai-communication-coach/
|
|-- app.py                      # Stage 4 — Streamlit UI, navigation, chat, history
|
|-- src/
|   |-- __init__.py
|   |-- transcribe.py           # Stage 1 — Whisper transcription to structured JSON
|   |-- metrics.py              # Stage 2 — filler words, WPM, naturalness, authenticity
|   |-- coach.py                # Stage 3 — agentic two-call coaching loop
|
|-- notebooks/
|   |-- VoiceCoach_EDA.ipynb    # EDA notebook — 8 sections, 10 charts
|   |-- eda_top_words.png
|   |-- eda_fillers.png
|   |-- eda_filler_timeline.png
|   |-- eda_pacing.png
|   |-- eda_wpm_dist.png
|   |-- eda_progress.png
|   |-- eda_naturalness_radar.png
|   |-- eda_correlation.png
|
|-- data/
|   |-- recordings/             # Raw audio and video files (gitignored)
|   |-- transcripts/            # Saved Whisper JSON output
|
|-- voicecoach_history.db       # SQLite session history (auto-created, gitignored)
|-- .env                        # GROQ_API_KEY (gitignored, never committed)
|-- .gitignore
|-- requirements.txt
|-- LICENSE
|-- README.md
```

---

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/vi25-BDA04/voicecoach-ai.git
cd voicecoach-ai
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS and Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install ffmpeg

Whisper requires ffmpeg to decode audio from video files.

**Windows:**
1. Download `ffmpeg-release-essentials.zip` from [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/)
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\ffmpeg-8.x.x-essentials_build\bin` to your system PATH
4. Restart your terminal and verify: `ffmpeg -version`

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

### 5. Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up with Google — no credit card required
3. Create an API key under API Keys and copy it

### 6. Configure environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

This file is already excluded via `.gitignore`. Never commit it.

### 7. Run the application

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## Requirements

```
openai-whisper
ipykernel
groq
python-dotenv
streamlit
plotly
pandas
numpy
matplotlib
seaborn
```

Install all:

```bash
pip install -r requirements.txt
```

---

## How to Use

1. Run `streamlit run app.py` and open `http://localhost:8501`
2. Enter your name on the welcome screen
3. Upload a recording — audio or video file
4. Click Analyse my speech
5. Review your results:
   - Confidence score ring, speech metrics, filler word chart
   - Transcript with filler words highlighted in red
   - Coaching feedback — draft and final reviewed version
   - Chat with the AI coach for follow-up questions
6. Click History in the navigation bar to see your progress across sessions

---

## Engineering Decisions

**Why is filler word detection written in Python rather than delegated to an LLM?**
Counting word occurrences is a deterministic operation. String matching is faster, free, completely reproducible, and testable. Using a language model for this task would introduce cost, latency, and non-determinism into a stage that requires none of those tradeoffs. AI is used only where it genuinely cannot be replaced by simpler code.

**Why Groq instead of the OpenAI API?**
Groq provides a free tier with generous rate limits and requires no credit card, making it suitable for a student project. The Llama 3.3 70B model produces coaching feedback of comparable quality to GPT-4o-mini for this use case. The client interface is nearly identical, so switching providers requires changing two lines of code.

**Why the Whisper base model?**
The base model runs locally, requires no API key, and is accurate enough for clear speech. It downloads once and works offline thereafter. Switching to the small or medium model for better accuracy on noisier recordings is a one-word change in transcribe.py.

**Why show authenticity indicators rather than a binary verdict?**
No text-based system can reliably distinguish human speech from high-quality synthetic audio. Displaying the individual signals is more honest, more educational, and more defensible than a single label the user might over-trust.

**Why SQLite for session history?**
SQLite is part of the Python standard library — no database server, no additional dependencies, no configuration. It works offline and is appropriate for a single-user local application.

**Why keep naturalness and authenticity detection in Python rather than an LLM?**
Both features analyse linguistic patterns that are fully computable from the transcript — pace variance, filler rates, sentence length distributions, word repetition ratios. These are statistical calculations, not language understanding tasks. Keeping them in Python means they run instantly, work offline, cost nothing, and produce fully explainable outputs. An LLM would add none of those benefits here.

---

## Roadmap

- Streamlit Cloud deployment for a publicly accessible live URL
- Progress tracking charts surfaced directly inside the application
- Downloadable PDF coaching report per session
- Multi-language support — Whisper supports 99 languages natively
- Real-time waveform display synchronised to audio playback
- Visual confidence signals from video input such as eye contact and posture

---

## Author

Vijaya
MSc Big Data Analytics

Built as a course project with a personal motivation: improving public speaking confidence through objective, data-driven feedback.

GitHub: [github.com/vi25-BDA04/voicecoach-ai](https://github.com/vi25-BDA04/voicecoach-ai)

---

## License

MIT License. Free to use, modify, and distribute with attribution.
