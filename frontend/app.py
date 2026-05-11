from __future__ import annotations

import html
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import OPTION_LABELS, RAW_DATA_DIR
from backend.inference import build_quiz_from_race_row
from backend.model_a import load_model, predict_best_answer
from backend.preprocessing import load_race_csv


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RACE Tutor AI",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS injection ──────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    /* ── Palette variables ── */
    :root {
        --ink:        #0f1117;
        --surface:    #161b27;
        --card:       #1e2535;
        --border:     #2d3650;
        --muted:      #5a6480;
        --body:       #a8b2cc;
        --bright:     #e8edf8;
        --amber:      #f0a500;
        --amber-dim:  #c47d00;
        --green:      #2ecc8a;
        --red:        #e55e5e;
        --blue:       #5b9cf6;
        --radius:     12px;
    }

    /* ── Global resets ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--ink) !important;
        font-family: 'DM Sans', sans-serif !important;
        color: var(--body) !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    section[data-testid="stSidebar"] { display: none; }
    .block-container {
        padding: 2rem 3rem 4rem !important;
        max-width: 1200px !important;
    }

    /* ── Hero banner ── */
    .hero {
        background: linear-gradient(135deg, #1a2140 0%, #0f1117 60%, #1a1505 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 240px; height: 240px;
        background: radial-gradient(circle, rgba(240,165,0,0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.6rem;
        font-weight: 700;
        color: var(--bright);
        margin: 0 0 0.4rem;
        letter-spacing: -0.5px;
    }
    .hero-title span { color: var(--amber); }
    .hero-sub {
        font-size: 0.95rem;
        color: var(--muted);
        font-weight: 300;
        letter-spacing: 0.5px;
    }

    /* ── Stat pills (top bar) ── */
    .stat-bar {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }
    .stat-pill {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 50px;
        padding: 0.45rem 1.1rem;
        display: flex;
        align-items: center;
        gap: 0.55rem;
        font-size: 0.82rem;
        color: var(--body);
    }
    .stat-pill .label { color: var(--muted); font-size: 0.75rem; }
    .stat-pill .value { color: var(--bright); font-weight: 600; }
    .stat-pill .dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: var(--amber);
        box-shadow: 0 0 6px var(--amber);
    }

    /* ── Section headings ── */
    .section-heading {
        font-family: 'Playfair Display', serif;
        font-size: 1.35rem;
        font-weight: 600;
        color: var(--bright);
        margin: 0 0 1.25rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid var(--border);
        display: flex;
        align-items: center;
        gap: 0.55rem;
    }
    .section-heading .icon {
        font-size: 1rem;
        opacity: 0.7;
    }

    /* ── Cards ── */
    .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.25rem;
    }
    .card-sm {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }

    /* ── Article display ── */
    .article-box {
        background: var(--surface);
        border-left: 3px solid var(--amber);
        border-radius: 0 var(--radius) var(--radius) 0;
        padding: 1.5rem 1.75rem;
        font-size: 0.9rem;
        line-height: 1.85;
        color: var(--body);
        margin-bottom: 1.5rem;
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Question display ── */
    .question-text {
        font-family: 'Playfair Display', serif;
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--bright);
        margin-bottom: 1.25rem;
        line-height: 1.5;
    }

    /* ── Answer option cards ── */
    .option-label {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px; height: 28px;
        border-radius: 50%;
        background: var(--border);
        color: var(--amber);
        font-weight: 600;
        font-size: 0.8rem;
        margin-right: 0.6rem;
        flex-shrink: 0;
        font-family: 'DM Mono', monospace;
    }

    /* ── Hint cards ── */
    .hint-card {
        background: linear-gradient(135deg, #1a2140, #161b27);
        border: 1px solid var(--blue);
        border-radius: var(--radius);
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        display: flex;
        gap: 0.75rem;
        align-items: flex-start;
    }
    .hint-badge {
        background: var(--blue);
        color: #fff;
        border-radius: 6px;
        padding: 0.15rem 0.55rem;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        white-space: nowrap;
        font-family: 'DM Mono', monospace;
    }
    .hint-text { color: var(--body); font-size: 0.88rem; line-height: 1.6; }

    /* ── Result banners ── */
    .result-correct {
        background: linear-gradient(135deg, rgba(46,204,138,0.12), rgba(46,204,138,0.04));
        border: 1px solid var(--green);
        border-radius: var(--radius);
        padding: 1rem 1.25rem;
        color: var(--green);
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-top: 1rem;
    }
    .result-wrong {
        background: linear-gradient(135deg, rgba(229,94,94,0.12), rgba(229,94,94,0.04));
        border: 1px solid var(--red);
        border-radius: var(--radius);
        padding: 1rem 1.25rem;
        color: var(--red);
        font-weight: 500;
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
        margin-top: 1rem;
    }
    .result-model {
        background: rgba(91,156,246,0.08);
        border: 1px solid rgba(91,156,246,0.3);
        border-radius: var(--radius);
        padding: 0.85rem 1.25rem;
        color: var(--blue);
        font-size: 0.85rem;
        margin-top: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Analytics metric cards ── */
    .metric-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.25rem 1.5rem;
        text-align: center;
    }
    .metric-value {
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--amber);
        line-height: 1;
        margin-bottom: 0.4rem;
    }
    .metric-label {
        font-size: 0.75rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
    }

    /* ── Upload / input overrides ── */
    [data-testid="stFileUploader"] {
        background: var(--surface) !important;
        border: 2px dashed var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 0.5rem !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--amber) !important;
    }

    /* Streamlit input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--bright) !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--amber) !important;
        box-shadow: 0 0 0 2px rgba(240,165,0,0.15) !important;
    }

    /* Labels */
    label, .stTextInput label, .stTextArea label,
    .stSelectbox label, .stRadio label {
        color: var(--muted) !important;
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        font-weight: 500 !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* Radio buttons */
    .stRadio > div { gap: 0.5rem !important; }
    .stRadio > div > label {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        cursor: pointer !important;
        transition: border-color 0.2s, background 0.2s !important;
        color: var(--body) !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
        font-size: 0.9rem !important;
        width: 100% !important;
    }
    .stRadio > div > label:hover {
        border-color: var(--amber) !important;
        background: rgba(240,165,0,0.05) !important;
    }
    .stRadio [data-checked="true"] > label,
    .stRadio > div > label[data-baseweb="radio"]:has(input:checked) {
        border-color: var(--amber) !important;
        background: rgba(240,165,0,0.08) !important;
    }

    /* Buttons */
    .stButton > button {
        background: var(--amber) !important;
        color: #0f1117 !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.3px !important;
        padding: 0.6rem 1.4rem !important;
        transition: opacity 0.2s, transform 0.1s !important;
    }
    .stButton > button:hover {
        opacity: 0.9 !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button[kind="secondary"] {
        background: var(--surface) !important;
        color: var(--body) !important;
        border: 1px solid var(--border) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--surface) !important;
        border-radius: var(--radius) !important;
        padding: 4px !important;
        border: 1px solid var(--border) !important;
        gap: 4px !important;
        margin-bottom: 1.75rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px !important;
        color: var(--muted) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        padding: 0.5rem 1.25rem !important;
        transition: all 0.2s !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--card) !important;
        color: var(--amber) !important;
        border: 1px solid var(--border) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }
    .stTabs [data-baseweb="tab-border"] { display: none !important; }

    /* Divider */
    hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        overflow: hidden !important;
    }

    /* Scrollbars */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--ink); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

    /* Hide streamlit chrome */
    #MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    return load_race_csv(path)


@st.cache_resource
def load_answer_model(path: str):
    return load_model(path)


def init_state() -> None:
    st.session_state.setdefault("quiz", None)
    st.session_state.setdefault("hint_count", 0)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("answer_revealed", False)


def _html(content: str) -> None:
    st.markdown(content, unsafe_allow_html=True)


def _safe(value: object) -> str:
    return html.escape(str(value), quote=False)


# ── Hero ──────────────────────────────────────────────────────────────────────
def render_hero() -> None:
    history = st.session_state.history
    attempts = len(history)
    accuracy = (
        f"{sum(h['is_correct'] for h in history) / attempts:.0%}" if attempts else "—"
    )
    avg_lat = (
        f"{sum(h['latency_ms'] for h in history) / attempts:.0f} ms"
        if attempts
        else "—"
    )
    quiz_loaded = "Active" if st.session_state.quiz else "None"

    _html(f"""
    <div class="hero">
        <div class="hero-title">RACE <span>Tutor</span> AI</div>
        <div class="hero-sub">Reading comprehension · Quiz generation · Answer verification</div>
    </div>
    <div class="stat-bar">
        <div class="stat-pill"><span class="dot"></span>
            <span class="label">Quiz</span>
            <span class="value">{quiz_loaded}</span>
        </div>
        <div class="stat-pill">
            <span class="label">Attempts</span>
            <span class="value">{attempts}</span>
        </div>
        <div class="stat-pill">
            <span class="label">Accuracy</span>
            <span class="value">{accuracy}</span>
        </div>
        <div class="stat-pill">
            <span class="label">Avg Latency</span>
            <span class="value">{avg_lat}</span>
        </div>
    </div>
    """)


# ── Article Input tab ─────────────────────────────────────────────────────────
def show_article_input() -> None:
    _html('<div class="section-heading"><span class="icon">📂</span> Load a Dataset</div>')
    default_path = RAW_DATA_DIR / "test.csv"

    col1, col2 = st.columns([3, 1], gap="medium")
    with col1:
        uploaded = st.file_uploader(
            "Upload a RACE CSV file",
            type=["csv"],
            help="CSV must have: article, question, A, B, C, D, answer columns",
        )
    with col2:
        st.write("")
        st.write("")
        use_default = st.button(
            "⚡  Random RACE Sample",
            use_container_width=True,
            help="Pull a random row from data/raw/test.csv",
        )

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        row = df.sample(1).iloc[0]
        st.session_state.quiz = build_quiz_from_race_row(row)
        st.session_state.hint_count = 0
        st.session_state.answer_revealed = False
        _html('<div class="result-correct">✓ Random sample loaded from uploaded CSV.</div>')

    if use_default:
        if default_path.exists():
            df = load_dataset(str(default_path))
            row = df.sample(1).iloc[0]
            st.session_state.quiz = build_quiz_from_race_row(row)
            st.session_state.hint_count = 0
            st.session_state.answer_revealed = False
            _html('<div class="result-correct">✓ Random sample loaded from data/raw/test.csv.</div>')
        else:
            st.warning("⚠️  Add data/raw/test.csv first, then try again.")

    st.markdown("---")
    _html('<div class="section-heading"><span class="icon">✏️</span> Build a Manual Quiz</div>')

    _html('<div class="card">')
    article = st.text_area(
        "Reading passage",
        height=200,
        placeholder="Paste the article or reading comprehension text here…",
    )
    question = st.text_input(
        "Question",
        placeholder="e.g. What is the main idea of the passage?",
    )

    _html('<div style="margin-top:0.75rem"></div>')
    opt_cols = st.columns(4, gap="small")
    options: dict[str, str] = {}
    for label, col in zip(OPTION_LABELS, opt_cols):
        with col:
            options[label] = st.text_input(
                f"Option {label}",
                placeholder=f"Answer {label}",
            )

    correct_label = st.selectbox(
        "Correct answer",
        OPTION_LABELS,
        format_func=lambda l: f"Option {l}",
    )
    _html("</div>")

    if st.button("Create Quiz →", type="primary"):
        if not article.strip() or not question.strip() or not all(options.values()):
            st.error("Please fill in the passage, question, and all four options.")
        else:
            row = pd.Series(
                {
                    "article": article,
                    "question": question,
                    **options,
                    "answer": correct_label,
                }
            )
            st.session_state.quiz = build_quiz_from_race_row(row)
            st.session_state.hint_count = 0
            st.session_state.answer_revealed = False
            _html('<div class="result-correct" style="margin-top:1rem">✓ Quiz created — switch to the Quiz View tab.</div>')


# ── Quiz View tab ─────────────────────────────────────────────────────────────
def show_quiz_view() -> None:
    quiz = st.session_state.quiz
    if not quiz:
        _html("""
        <div class="card" style="text-align:center;padding:3rem 2rem;color:var(--muted)">
            <div style="font-size:2.5rem;margin-bottom:1rem">📖</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.1rem;color:var(--body);margin-bottom:0.5rem">
                No quiz loaded yet
            </div>
            <div style="font-size:0.85rem">Go to <strong style="color:var(--amber)">Article Input</strong> to load or create a quiz.</div>
        </div>
        """)
        return

    # Article
    _html('<div class="section-heading"><span class="icon">📄</span> Reading Passage</div>')
    _html(f'<div class="article-box">{_safe(quiz["article"])}</div>')

    # Question
    _html(f'<div class="question-text">❓ {_safe(quiz["question"])}</div>')

    # Answer options
    selected = st.radio(
        "Select your answer",
        OPTION_LABELS,
        format_func=lambda label: f"{label}.  {quiz['options'][label]}",
        label_visibility="collapsed",
    )

    st.write("")
    model_path = PROJECT_ROOT / "models" / "model_a" / "logreg_candidate_rich.pkl"
    use_model = model_path.exists()

    if st.button("Check Answer →", type="primary"):
        started = time.perf_counter()
        model_result = None
        if use_model:
            model = load_answer_model(str(model_path))
            model_result = predict_best_answer(
                model, quiz["article"], quiz["question"], quiz["options"]
            )
        latency_ms = (time.perf_counter() - started) * 1000

        is_correct = selected == quiz["correct_label"]

        if is_correct:
            _html('<div class="result-correct">✓ Correct!  Well done.</div>')
        else:
            _html(
                f'<div class="result-wrong">'
                f'<span>✗</span>'
                f'<span>Incorrect.  The correct answer is '
                f'<strong>{_safe(quiz["correct_label"])}</strong>: {_safe(quiz["correct_answer"])}</span>'
                f'</div>'
            )

        if model_result:
            conf_pct = f"{model_result['confidence']:.0%}"
            _html(
                f'<div class="result-model">'
                f'🤖 Model A predicts <strong>{_safe(model_result["answer"])}</strong> '
                f'with {conf_pct} confidence'
                f'</div>'
            )

        st.session_state.history.append(
            {
                "selected": selected,
                "correct": quiz["correct_label"],
                "is_correct": is_correct,
                "latency_ms": latency_ms,
            }
        )


# ── Hints tab ─────────────────────────────────────────────────────────────────
def show_hints() -> None:
    quiz = st.session_state.quiz
    if not quiz:
        _html("""
        <div class="card" style="text-align:center;padding:3rem 2rem;color:var(--muted)">
            <div style="font-size:2.5rem;margin-bottom:1rem">💡</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.1rem;color:var(--body);margin-bottom:0.5rem">
                No quiz loaded
            </div>
            <div style="font-size:0.85rem">Load a quiz first to access hints.</div>
        </div>
        """)
        return

    total_hints = len(quiz["hints"])
    shown = st.session_state.hint_count

    # Progress header
    _html(f"""
    <div class="section-heading">
        <span class="icon">💡</span> Hints
        <span style="margin-left:auto;font-family:'DM Mono',monospace;font-size:0.75rem;
                     color:var(--muted);font-weight:400">{shown} / {total_hints} revealed</span>
    </div>
    """)

    # Reveal hints
    for i, hint in enumerate(quiz["hints"][:shown], start=1):
        _html(
            f'<div class="hint-card">'
            f'<span class="hint-badge">HINT {i}</span>'
            f'<span class="hint-text">{_safe(hint)}</span>'
            f'</div>'
        )

    if shown == 0:
        _html('<div class="card-sm" style="color:var(--muted);font-size:0.875rem">Click the button below to reveal hints one at a time.</div>')

    col1, col2 = st.columns([1, 3])
    with col1:
        if shown < total_hints:
            if st.button("Show Next Hint", use_container_width=True):
                st.session_state.hint_count += 1
                st.rerun()
        else:
            st.button("All Hints Shown", disabled=True, use_container_width=True)

    if shown >= total_hints:
        st.write("")
        _html('<div style="margin-bottom:0.5rem;color:var(--muted);font-size:0.8rem;text-transform:uppercase;letter-spacing:1px">Still stuck?</div>')
        if not st.session_state.answer_revealed:
            if st.button("🔓 Reveal Answer", type="primary"):
                st.session_state.answer_revealed = True
                st.rerun()
        if st.session_state.answer_revealed:
            _html(
                f'<div class="result-correct" style="margin-top:0.5rem">'
                f'Answer: <strong>{_safe(quiz["correct_label"])}</strong> — {_safe(quiz["correct_answer"])}'
                f'</div>'
            )

    st.write("")
    _html('<div class="section-heading" style="font-size:1rem"><span class="icon">🎯</span> Generated Distractors</div>')
    for label, distractor in zip(["1", "2", "3"], quiz.get("generated_distractors", [])):
        _html(
            f'<div class="card-sm">'
            f'<span class="option-label">{label}</span>{_safe(distractor)}'
            f'</div>'
        )


# ── Analytics tab ─────────────────────────────────────────────────────────────
def show_dashboard() -> None:
    _html('<div class="section-heading"><span class="icon">📊</span> Session Analytics</div>')
    history = st.session_state.history

    if not history:
        _html("""
        <div class="card" style="text-align:center;padding:3rem 2rem;color:var(--muted)">
            <div style="font-size:2.5rem;margin-bottom:1rem">📈</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.1rem;color:var(--body);margin-bottom:0.5rem">
                No data yet
            </div>
            <div style="font-size:0.85rem">Complete a quiz attempt to see your analytics.</div>
        </div>
        """)
        return

    df = pd.DataFrame(history)
    accuracy = df["is_correct"].mean()
    avg_latency = df["latency_ms"].mean()
    streak = 0
    for h in reversed(history):
        if h["is_correct"]:
            streak += 1
        else:
            break

    # Metric cards
    c1, c2, c3, c4 = st.columns(4, gap="small")
    for col, label, value in [
        (c1, "Total Attempts", str(len(df))),
        (c2, "Accuracy", f"{accuracy:.0%}"),
        (c3, "Avg Latency", f"{avg_latency:.0f} ms"),
        (c4, "Current Streak", f"{streak} ✓"),
    ]:
        with col:
            _html(
                f'<div class="metric-card">'
                f'<div class="metric-value">{value}</div>'
                f'<div class="metric-label">{label}</div>'
                f'</div>'
            )

    st.write("")
    _html('<div class="section-heading" style="font-size:1rem"><span class="icon">📋</span> Attempt Log</div>')

    display_df = df.copy()
    display_df["result"] = display_df["is_correct"].map({True: "✓ Correct", False: "✗ Wrong"})
    display_df["latency_ms"] = display_df["latency_ms"].round(1)
    display_df = display_df[["result", "selected", "correct", "latency_ms"]].rename(
        columns={
            "result": "Result",
            "selected": "Your Answer",
            "correct": "Correct Answer",
            "latency_ms": "Latency (ms)",
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.write("")
    col_dl, _ = st.columns([1, 3])
    with col_dl:
        st.download_button(
            "⬇  Export Session Log",
            df.to_csv(index=False).encode("utf-8"),
            "race_tutor_session_log.csv",
            "text/csv",
            use_container_width=True,
        )


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    init_state()
    render_hero()

    tabs = st.tabs(["📂  Article Input", "📖  Quiz View", "💡  Hints", "📊  Analytics"])
    with tabs[0]:
        show_article_input()
    with tabs[1]:
        show_quiz_view()
    with tabs[2]:
        show_hints()
    with tabs[3]:
        show_dashboard()


if __name__ == "__main__":
    main()
