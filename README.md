# 🔍 AI Candidate Search Engine

A production-grade semantic search system that lets recruiters find the best candidates using natural language queries — no keyword memorisation required.

---

## 🏗️ Architecture

```
User Query (natural language)
        │
        ▼
┌───────────────────┐
│   Streamlit UI    │  ← app.py
└────────┬──────────┘
         │
         ▼
┌───────────────────┐        ┌──────────────────────┐
│  Query Embedding  │        │  Candidate Embeddings │
│  (embedder.py)    │        │  (cached .npy file)   │
└────────┬──────────┘        └──────────┬───────────┘
         │                              │
         └──────────┬───────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │   Hybrid Ranker      │
         │  search_engine.py    │
         │                      │
         │  score = 0.7×cosine  │
         │        + 0.3×keyword │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  Explanation Engine  │
         │  explanation.py      │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │   Ranked Result Cards│
         └──────────────────────┘
```

---

## 📂 Project Structure

```
search-system/
├── app.py                  # Streamlit UI + integration layer
├── requirements.txt
├── README.md
├── data/
│   ├── candidates.csv      # Parsed candidate profiles (1 700+ rows)
│   └── embeddings.npy      # Auto-generated on first run; cached thereafter
└── src/
    ├── __init__.py
    ├── data_loader.py      # CSV loading, cleaning, profile_text construction
    ├── embedder.py         # sentence-transformers model + embedding cache
    ├── search_engine.py    # Cosine similarity + keyword overlap hybrid scorer
    └── explanation.py      # Keyword extraction + explanation generation
```

---

## ⚙️ Setup

### 1. Clone / download the project

```bash
git clone https://github.com/<your-username>/search-system.git
cd search-system
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> First install downloads ~90 MB of model weights (all-MiniLM-L6-v2). This happens once.

---

## 🚀 Run Locally

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

**First run:** embeddings are computed for all ~1 700 candidates and saved to `data/embeddings.npy`. This takes ~30 s on CPU. Every subsequent run loads the cached file instantly.

---

## ☁️ Deploy on Streamlit Cloud

1. Push the project to a **public GitHub repo**:

```bash
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/<your-username>/search-system.git
git push -u origin main
```

2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, entry point `app.py`
4. Click **Deploy**

> Streamlit Cloud's free tier has ~1 GB RAM — sufficient for this app.

---

## 💡 Example Queries

| Query | What it finds |
|---|---|
| `senior backend engineer Python fintech` | Experienced Python devs with fintech exposure |
| `startup engineer who can do everything` | Versatile full-stack engineers |
| `ML engineer with production deployment` | MLOps-oriented data scientists |
| `React frontend developer TypeScript` | Frontend engineers with modern JS stack |
| `Java Spring Boot microservices architect` | Enterprise Java backend engineers |
| `mobile developer Flutter iOS Android` | Cross-platform mobile specialists |
| `data scientist NLP deep learning` | NLP/ML researchers and practitioners |

---

## 🧠 How It Works

### Semantic Similarity
Each candidate's resume text is converted into a 384-dimensional vector using `sentence-transformers/all-MiniLM-L6-v2`. The query is embedded the same way. Cosine similarity is computed via a fast dot product (vectors are L2-normalised at encoding time).

### Keyword Overlap
A simple token intersection score catches exact skill/technology matches that the neural model might miss.

### Hybrid Score
```
score = 0.7 × semantic_similarity + 0.3 × keyword_overlap
```

### Explanation Engine
Keyword banks for skills, experience level, domain, and location are matched against both the query and the candidate's profile to produce a structured explanation.

---

## ⚠️ Limitations

- Resume text parsing is heuristic — highly unusual resume formats may produce noisy `name` / `title` fields.
- The keyword overlap is case-insensitive but not lemmatised (e.g. "engineer" ≠ "engineering").
- All-MiniLM-L6-v2 has a 256-token limit; very long resumes are truncated.

---

## 🔮 Future Improvements

- Re-ranking with a cross-encoder model (e.g. `ms-marco-MiniLM-L-6-v2`)
- Structured skill extraction with a NER model
- Filters by location, years of experience, or education level
- FAISS / Annoy approximate nearest-neighbour index for 100k+ scale
- Saved searches and candidate shortlisting
- Job description ↔ candidate matching (dual-encoder)
