"""
app.py
------
Resume Screening AI — Streamlit web application entry point.

Architecture:
  • resume_parser.py     → PDF text extraction
  • text_preprocessing.py → NLP pipeline (lowercase, stop-words, etc.)
  • app.py               → UI layout, TF-IDF scoring, results dashboard
"""

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from resume_parser import extract_text_from_pdf, get_pdf_metadata
from text_preprocessing import preprocess_text, get_token_set

# ──────────────────────────────────────────────────────────────
# Page configuration (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Screening AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Custom CSS — dark editorial aesthetic
# ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* ── Global Reset ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background: #0d0f14;
        color: #e8e6e0;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #13161e !important;
        border-right: 1px solid #1f2330;
    }
    section[data-testid="stSidebar"] * { color: #b0adb8 !important; }

    /* ── Hero title ── */
    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 3rem;
        font-weight: 400;
        letter-spacing: -0.02em;
        line-height: 1.1;
        color: #f0ede8;
        margin-bottom: 0.25rem;
    }
    .hero-sub {
        font-family: 'DM Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.18em;
        color: #6b7a99;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }

    /* ── Card containers ── */
    .card {
        background: #13161e;
        border: 1px solid #1f2330;
        border-radius: 12px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.25rem;
    }
    .card-accent {
        border-left: 3px solid #4f8ef7;
    }

    /* ── Score ring display ── */
    .score-block {
        text-align: center;
        padding: 2rem 1rem;
    }
    .score-number {
        font-family: 'DM Serif Display', serif;
        font-size: 5rem;
        line-height: 1;
        font-weight: 400;
    }
    .score-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-top: 0.4rem;
    }

    /* ── Recommendation badge ── */
    .badge {
        display: inline-block;
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        padding: 0.35rem 0.9rem;
        border-radius: 100px;
        font-weight: 500;
    }
    .badge-strong  { background: #0d2e1a; color: #3ddc84; border: 1px solid #1a5c35; }
    .badge-moderate{ background: #2a2000; color: #fbbf24; border: 1px solid #5c4000; }
    .badge-weak    { background: #2a0f0f; color: #f87171; border: 1px solid #5c1f1f; }

    /* ── Keyword chips ── */
    .chip-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.6rem; }
    .chip {
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        padding: 0.2rem 0.7rem;
        border-radius: 6px;
        letter-spacing: 0.04em;
    }
    .chip-match   { background: #0d2e1a; color: #3ddc84; border: 1px solid #1a5c35; }
    .chip-missing { background: #2a0f0f; color: #f87171; border: 1px solid #5c1f1f; }

    /* ── Section labels ── */
    .section-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #6b7a99;
        margin-bottom: 0.6rem;
    }

    /* ── Divider ── */
    hr.styled { border: none; border-top: 1px solid #1f2330; margin: 1.5rem 0; }

    /* ── Streamlit overrides ── */
    .stTextArea textarea {
        background: #13161e !important;
        border: 1px solid #1f2330 !important;
        color: #e8e6e0 !important;
        font-family: 'DM Sans', sans-serif !important;
        border-radius: 8px !important;
    }
    .stFileUploader {
        background: #13161e !important;
        border: 1px dashed #2e3450 !important;
        border-radius: 10px !important;
        padding: 0.5rem !important;
    }
    div[data-testid="stFileUploadDropzone"] {
        background: #13161e !important;
        border: 1.5px dashed #2e3450 !important;
        border-radius: 10px !important;
    }
    div[data-testid="stFileUploadDropzone"] * { color: #6b7a99 !important; }

    /* Primary button */
    .stButton > button {
        background: #4f8ef7 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        padding: 0.65rem 2rem !important;
        width: 100% !important;
        transition: opacity 0.2s ease !important;
    }
    .stButton > button:hover { opacity: 0.85 !important; }

    /* Progress bar */
    .stProgress > div > div { background: #4f8ef7 !important; border-radius: 99px !important; }
    .stProgress > div { background: #1f2330 !important; border-radius: 99px !important; }

    /* Expander */
    .streamlit-expanderHeader {
        background: #13161e !important;
        color: #b0adb8 !important;
        border: 1px solid #1f2330 !important;
        border-radius: 8px !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.1em !important;
    }
    .streamlit-expanderContent {
        background: #13161e !important;
        border: 1px solid #1f2330 !important;
        border-top: none !important;
        color: #9099b0 !important;
        font-size: 0.85rem !important;
        line-height: 1.7 !important;
    }

    /* Hide default Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    /* Metric cards */
    .metric-mini {
        background: #0d0f14;
        border: 1px solid #1f2330;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        text-align: center;
    }
    .metric-mini-val {
        font-family: 'DM Serif Display', serif;
        font-size: 1.8rem;
        color: #f0ede8;
    }
    .metric-mini-lbl {
        font-family: 'DM Mono', monospace;
        font-size: 0.6rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #6b7a99;
        margin-top: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────
# Scoring engine
# ──────────────────────────────────────────────────────────────

def compute_similarity(resume_text: str, job_desc_text: str) -> float:
    """
    Compute cosine similarity between resume and job description using TF-IDF.

    Returns:
        Float in [0, 1] representing the match score.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc_text])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return float(similarity[0][0])


def get_recommendation(score_pct: float) -> tuple[str, str, str]:
    """
    Map a percentage score to a hiring recommendation.

    Returns:
        (recommendation_text, badge_class, colour_hex)
    """
    if score_pct >= 80:
        return "Strong Match", "badge-strong", "#3ddc84"
    elif score_pct >= 60:
        return "Moderate Match", "badge-moderate", "#fbbf24"
    else:
        return "Weak Match", "badge-weak", "#f87171"


# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style='margin-bottom:1.5rem;'>
            <div style='font-family:"DM Serif Display",serif;font-size:1.3rem;color:#f0ede8;'>
                Resume<br>Screening AI
            </div>
            <div style='font-family:"DM Mono",monospace;font-size:0.6rem;letter-spacing:0.15em;
                        color:#4f8ef7;text-transform:uppercase;margin-top:0.3rem;'>
                v1.0 · NLP Engine
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**How it works**")
    steps = [
        ("01", "Upload your resume PDF"),
        ("02", "Paste the job description"),
        ("03", "Click Analyze"),
        ("04", "Review your match score"),
    ]
    for num, step in steps:
        st.markdown(
            f"""<div style='display:flex;gap:0.75rem;align-items:flex-start;
                            margin-bottom:0.6rem;'>
                    <span style='font-family:"DM Mono",monospace;font-size:0.65rem;
                                 color:#4f8ef7;padding-top:0.05rem;'>{num}</span>
                    <span style='font-size:0.8rem;color:#9099b0;'>{step}</span>
                </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='styled'>", unsafe_allow_html=True)

    st.markdown(
        """<div style='font-family:"DM Mono",monospace;font-size:0.65rem;
                        letter-spacing:0.1em;color:#3b4260;text-transform:uppercase;
                        margin-bottom:0.5rem;'>Scoring Thresholds</div>""",
        unsafe_allow_html=True,
    )
    thresholds = [
        ("≥ 80%", "Strong Match", "#3ddc84"),
        ("60–79%", "Moderate Match", "#fbbf24"),
        ("< 60%", "Weak Match", "#f87171"),
    ]
    for pct, label, color in thresholds:
        st.markdown(
            f"""<div style='display:flex;justify-content:space-between;
                            margin-bottom:0.35rem;font-size:0.78rem;'>
                    <span style='font-family:"DM Mono",monospace;color:#6b7a99;'>{pct}</span>
                    <span style='color:{color};font-weight:500;'>{label}</span>
                </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='styled'>", unsafe_allow_html=True)
    st.markdown(
        """<div style='font-size:0.72rem;color:#3b4260;line-height:1.6;'>
            Built with Streamlit · scikit-learn · PyPDF2
           </div>""",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────
# Main layout
# ──────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class='hero-title'>Resume Screening <em>AI</em></div>
    <div class='hero-sub'>Intelligent candidate matching · TF-IDF cosine similarity engine</div>
    """,
    unsafe_allow_html=True,
)

# ── Input columns ──────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("<div class='section-label'>Upload Resume (PDF)</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="",
        type=["pdf"],
        help="Supported format: PDF only",
        label_visibility="collapsed",
    )
    if uploaded_file:
        st.markdown(
            f"""<div style='font-family:"DM Mono",monospace;font-size:0.72rem;
                            color:#3ddc84;margin-top:0.4rem;'>
                    ✓ {uploaded_file.name} ({uploaded_file.size // 1024} KB)
                </div>""",
            unsafe_allow_html=True,
        )

with col_right:
    st.markdown("<div class='section-label'>Job Description</div>", unsafe_allow_html=True)
    job_description = st.text_area(
        label="",
        placeholder="Paste the full job description here…",
        height=160,
        label_visibility="collapsed",
    )

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ── Analyze button ─────────────────────────────────────────────
_, btn_col, _ = st.columns([2, 1, 2])
with btn_col:
    analyze_clicked = st.button("⚡ Analyze Resume", use_container_width=True)


# ──────────────────────────────────────────────────────────────
# Analysis & Results
# ──────────────────────────────────────────────────────────────

if analyze_clicked:

    # ── Validation ────────────────────────────────────────────
    if not uploaded_file:
        st.error("Please upload a PDF resume before analyzing.")
        st.stop()
    if not job_description.strip():
        st.error("Please paste a job description before analyzing.")
        st.stop()

    # ── Extraction ────────────────────────────────────────────
    with st.spinner("Extracting text from PDF…"):
        try:
            resume_raw = extract_text_from_pdf(uploaded_file)
            uploaded_file.seek(0)  # Reset for metadata read
            meta = get_pdf_metadata(uploaded_file)
        except ValueError as e:
            st.error(str(e))
            st.stop()

    if not resume_raw.strip():
        st.error("No readable text found in the PDF. It may be image-based or encrypted.")
        st.stop()

    # ── NLP Preprocessing ─────────────────────────────────────
    with st.spinner("Running NLP preprocessing…"):
        resume_clean = preprocess_text(resume_raw)
        jd_clean = preprocess_text(job_description)

    # ── TF-IDF + Cosine Similarity ────────────────────────────
    with st.spinner("Computing TF-IDF similarity…"):
        score = compute_similarity(resume_clean, jd_clean)
        score_pct = round(score * 100, 1)

    # ── Keyword Analysis ──────────────────────────────────────
    resume_tokens = get_token_set(resume_raw)
    jd_tokens = get_token_set(job_description)

    matching_keywords = sorted(resume_tokens & jd_tokens)
    missing_keywords = sorted(jd_tokens - resume_tokens)

    # Filter very short tokens for display cleanliness
    matching_keywords = [k for k in matching_keywords if len(k) > 2]
    missing_keywords = [k for k in missing_keywords if len(k) > 2]

    recommendation, badge_class, rec_color = get_recommendation(score_pct)

    # ─────────────────────────────────────────────────────────
    # Results Dashboard
    # ─────────────────────────────────────────────────────────
    st.markdown("<hr class='styled'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-label' style='margin-bottom:1rem;'>Analysis Results</div>",
        unsafe_allow_html=True,
    )

    # ── Row 1: Score + Recommendation + Mini Metrics ──────────
    r1_a, r1_b, r1_c = st.columns([1, 1.4, 1.4], gap="medium")

    with r1_a:
        st.markdown(
            f"""
            <div class='card score-block'>
                <div class='section-label'>Match Score</div>
                <div class='score-number' style='color:{rec_color};'>{score_pct}<span style='font-size:2rem;'>%</span></div>
                <div class='score-label' style='color:{rec_color};margin-top:0.6rem;'>Cosine Similarity</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with r1_b:
        st.markdown(
            f"""
            <div class='card' style='height:100%;'>
                <div class='section-label'>Hiring Recommendation</div>
                <div style='margin:1rem 0 1.2rem;'>
                    <span class='badge {badge_class}'>{recommendation}</span>
                </div>
                <div style='font-size:0.82rem;color:#9099b0;line-height:1.6;'>
                    {"This resume demonstrates strong alignment with the role requirements. Recommend proceeding to the interview stage." if score_pct >= 80
                     else "Partial alignment detected. Consider assessing transferable skills or requesting a cover letter for further evaluation." if score_pct >= 60
                     else "Limited keyword and contextual overlap with the job description. This candidate may not meet the core requirements."}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with r1_c:
        st.markdown(
            f"""
            <div class='card' style='height:100%;'>
                <div class='section-label'>Quick Stats</div>
                <div style='display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;margin-top:0.6rem;'>
                    <div class='metric-mini'>
                        <div class='metric-mini-val'>{len(matching_keywords)}</div>
                        <div class='metric-mini-lbl'>Matching</div>
                    </div>
                    <div class='metric-mini'>
                        <div class='metric-mini-val'>{len(missing_keywords)}</div>
                        <div class='metric-mini-lbl'>Missing</div>
                    </div>
                    <div class='metric-mini'>
                        <div class='metric-mini-val'>{meta["num_pages"]}</div>
                        <div class='metric-mini-lbl'>Pages</div>
                    </div>
                    <div class='metric-mini'>
                        <div class='metric-mini-val'>{len(resume_raw.split())}</div>
                        <div class='metric-mini-lbl'>Words</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Progress bar ──────────────────────────────────────────
    st.markdown(
        "<div class='section-label'>Match Progress</div>", unsafe_allow_html=True
    )
    st.progress(int(score_pct))

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Row 2: Matching & Missing keywords ────────────────────
    kw_left, kw_right = st.columns(2, gap="medium")

    with kw_left:
        matching_html = "".join(
            f"<span class='chip chip-match'>{k}</span>"
            for k in matching_keywords[:40]
        ) or "<span style='color:#3b4260;font-size:0.8rem;'>No matching keywords found.</span>"
        st.markdown(
            f"""
            <div class='card card-accent'>
                <div class='section-label'>✓ Matching Keywords ({len(matching_keywords)})</div>
                <div class='chip-row'>{matching_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kw_right:
        missing_html = "".join(
            f"<span class='chip chip-missing'>{k}</span>"
            for k in missing_keywords[:40]
        ) or "<span style='color:#3b4260;font-size:0.8rem;'>No missing keywords — great coverage!</span>"
        st.markdown(
            f"""
            <div class='card' style='border-left:3px solid #f87171;'>
                <div class='section-label'>✗ Missing Keywords ({len(missing_keywords)})</div>
                <div class='chip-row'>{missing_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Resume text preview ───────────────────────────────────
    with st.expander("📄 Resume Text Preview", expanded=False):
        preview = resume_raw[:2000] + ("…" if len(resume_raw) > 2000 else "")
        st.markdown(
            f"<div style='font-size:0.82rem;line-height:1.75;color:#9099b0;'>{preview}</div>",
            unsafe_allow_html=True,
        )

    # ── Keyword frequency table (bonus insight) ───────────────
    with st.expander("📊 Top Keyword Frequency Analysis", expanded=False):
        from collections import Counter

        all_tokens = resume_clean.split()
        freq = Counter(all_tokens)
        top_kw = [(w, c) for w, c in freq.most_common(20) if len(w) > 2]
        if top_kw:
            df = pd.DataFrame(top_kw, columns=["Keyword", "Frequency"])
            df.index = df.index + 1  # 1-based index
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=False,
            )
        else:
            st.info("Not enough tokens for frequency analysis.")
