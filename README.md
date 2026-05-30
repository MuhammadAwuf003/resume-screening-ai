# Resume Screening AI 🎯

> An intelligent, end-to-end resume screening web application powered by NLP, TF-IDF vectorisation, and cosine similarity — built with Python and Streamlit.

---

## Overview

**Resume Screening AI** automates the first stage of recruitment by comparing a candidate's resume against a job description and producing an objective match score, keyword analysis, and a hiring recommendation — all inside a polished, interactive web interface.

Recruiters spend an average of **6–7 seconds** per resume at the initial screening stage. This tool assists them by surfacing the most relevant signals instantly, so human judgment can be applied where it matters most.

---

## Features

| Feature | Details |
|---|---|
| 📄 **PDF Resume Upload** | Drag-and-drop or click to upload; text extracted via PyPDF2 |
| 📝 **Job Description Input** | Free-text area for any role description |
| 🤖 **NLP Preprocessing** | Lowercase → punctuation removal → stopword removal → whitespace normalisation |
| 📊 **TF-IDF + Cosine Similarity** | Industry-standard vectorisation and similarity scoring |
| 🎯 **Match Score (%)** | Clear percentage displayed with a progress bar |
| ✅ **Matching Keywords** | Keywords present in both resume and JD |
| ❌ **Missing Keywords** | JD keywords absent from the resume |
| 💡 **Hiring Recommendation** | Strong / Moderate / Weak Match with contextual advice |
| 📈 **Keyword Frequency Table** | Top 20 tokens from the resume by frequency |
| 🎨 **Professional Dark UI** | DM Serif Display + DM Mono typography, editorial dark theme |

---

## Architecture

```
resume_screening_app/
│
├── app.py                  # Streamlit entry point — UI, orchestration, scoring
├── resume_parser.py        # PDF text & metadata extraction (PyPDF2)
├── text_preprocessing.py   # NLP pipeline (lowercase, stopwords, tokenisation)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

### Data Flow

```
User uploads PDF
      │
      ▼
resume_parser.py ──► Raw text string
      │
      ▼
text_preprocessing.py ──► Cleaned token stream
      │
      ▼
TF-IDF Vectorizer (scikit-learn) ──► Document vectors
      │
      ▼
Cosine Similarity ──► Score [0–1]
      │
      ▼
Streamlit UI ──► Score %, Keywords, Recommendation
```

---

## Scoring Logic

| Score | Recommendation | Action |
|---|---|---|
| ≥ 80% | 🟢 **Strong Match** | Proceed to interview |
| 60–79% | 🟡 **Moderate Match** | Review transferable skills |
| < 60% | 🔴 **Weak Match** | Unlikely to meet requirements |

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip

### 1. Clone the repository

```bash
git clone https://github.com/your-username/resume-screening-ai.git
cd resume-screening-ai
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## Usage

1. **Upload** your resume as a PDF using the file uploader on the left.
2. **Paste** the full job description into the text area on the right.
3. **Click** the `⚡ Analyze Resume` button.
4. **Review** the results dashboard:
   - Match score ring and progress bar
   - Hiring recommendation with contextual guidance
   - Quick stats (matching keywords, missing keywords, page count, word count)
   - Colour-coded keyword chips (green = present, red = missing)
   - Resume text preview
   - Keyword frequency table

---

## Screenshots

> _Add your screenshots here after running the app._

| Input Screen | Results Dashboard |
|---|---|
| `screenshots/input.png` | `screenshots/results.png` |

---

## Technical Notes

### Why TF-IDF + Cosine Similarity?

- **TF-IDF** (Term Frequency–Inverse Document Frequency) weights terms by how important they are to a document relative to a corpus, reducing noise from common words.
- **Cosine Similarity** measures the angle between two document vectors, making it length-invariant — a two-page resume and a five-page resume are compared fairly.
- Together they provide a robust, interpretable baseline that outperforms simple keyword counting.

### Stopword Handling

The app ships a **built-in stopword list** covering ~90 common English function words, so it works without a network connection. If `nltk` is installed and the `stopwords` corpus is downloadable, it upgrades to NLTK's full 179-word English list automatically.

---

## Future Improvements

- [ ] **Section-aware parsing** — weight experience, skills, and education sections differently
- [ ] **Semantic similarity** — integrate sentence-transformers for embedding-based matching beyond keyword overlap
- [ ] **Multi-resume batch mode** — rank a folder of resumes against a single JD
- [ ] **Named Entity Recognition** — extract and highlight skills, companies, and qualifications
- [ ] **Export report** — download a PDF/CSV summary of the analysis
- [ ] **ATS simulation** — flag common formatting issues that confuse Applicant Tracking Systems
- [ ] **Dark/light mode toggle** — user preference for theme
- [ ] **Resume improvement tips** — actionable suggestions to close the keyword gap

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥ 1.32 | Web UI framework |
| `PyPDF2` | ≥ 3.0 | PDF text extraction |
| `scikit-learn` | ≥ 1.4 | TF-IDF vectorisation & cosine similarity |
| `pandas` | ≥ 2.1 | Keyword frequency table display |
| `nltk` | ≥ 3.8 | Optional enhanced stopword corpus |

---

## License

MIT License — see `LICENSE` for details.

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

_Built with ♥ using Python, Streamlit, and scikit-learn_
