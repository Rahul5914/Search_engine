"""
app.py - AI Candidate Search Engine
Streamlit UI: semantic + keyword hybrid search over candidate resumes.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import load_candidates
from embedder import get_embeddings, embed_query
from search_engine import search
from explanation import generate_explanation, highlight_keywords

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Candidate Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
/* ---- global ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ---- header ---- */
.hero { text-align:center; padding:2.5rem 0 1.5rem; }
.hero h1 { font-size:2.6rem; font-weight:700; letter-spacing:-0.5px;
           background:linear-gradient(135deg,#6366f1,#8b5cf6,#ec4899);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero p  { font-size:1.1rem; color:#6b7280; margin-top:.4rem; }

/* ---- search box ---- */
.search-wrap { max-width:760px; margin:0 auto 2rem; }

/* ---- card ---- */
.card {
    background:#ffffff;
    border:1px solid #e5e7eb;
    border-radius:14px;
    padding:1.4rem 1.6rem;
    margin-bottom:1rem;
    box-shadow:0 1px 4px rgba(0,0,0,.06);
    transition:box-shadow .2s;
}
.card:hover { box-shadow:0 4px 16px rgba(99,102,241,.12); }
.card-rank { display:inline-block; background:#6366f1; color:#fff;
             font-size:.72rem; font-weight:700; padding:2px 8px;
             border-radius:20px; margin-bottom:.6rem; }
.card-name { font-size:1.15rem; font-weight:700; color:#111827; margin:0 0 .1rem; }
.card-title { font-size:.9rem; color:#6366f1; font-weight:500; margin:0 0 .3rem; }
.card-meta  { font-size:.82rem; color:#9ca3af; margin-bottom:.8rem; }
.score-label { font-size:.75rem; color:#6b7280; margin-bottom:4px; font-weight:500; }
.explanation { background:#f9fafb; border-left:3px solid #6366f1;
               border-radius:0 8px 8px 0; padding:.75rem 1rem;
               font-size:.84rem; color:#374151; margin-top:.9rem; line-height:1.65; }

/* ---- sidebar ---- */
.stSidebar [data-testid="stMarkdownContainer"] p { font-size:.88rem; }

/* ---- stats badge ---- */
.stat-chip { display:inline-block; background:#f3f4f6; border:1px solid #e5e7eb;
             border-radius:8px; padding:4px 12px; font-size:.82rem;
             color:#374151; margin:0 6px 6px 0; font-weight:500; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Data loading ──────────────────────────────────────────────────────────────

DATA_PATH = "data/candidates.csv"


@st.cache_data(show_spinner=False)
def load_data():
    return load_candidates(DATA_PATH)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Search Settings")
    top_k = st.slider("Results to show", min_value=5, max_value=50, value=20, step=5)
    min_score = st.slider(
        "Minimum relevance score", min_value=0.0, max_value=1.0, value=0.1, step=0.05
    )
    show_resume = st.toggle("Show resume snippet", value=False)

    st.markdown("---")
    st.markdown("### 💡 Example queries")
    example_queries = [
        "senior backend engineer Python fintech",
        "startup engineer who can do everything",
        "ML engineer with production deployment",
        "React frontend developer with TypeScript",
        "data scientist with NLP and deep learning",
        "Java Spring Boot microservices architect",
        "mobile developer Flutter iOS Android",
        "full stack engineer Node React AWS",
    ]
    selected_example = st.selectbox(
        "Try a sample query:", [""] + example_queries, label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:#9ca3af'>Built with sentence-transformers · all-MiniLM-L6-v2</small>",
        unsafe_allow_html=True,
    )

# ── Hero ──────────────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="hero">
  <h1>🔍 AI Candidate Search Engine</h1>
  <p>Semantic search over 1 700+ engineer profiles — find the right hire in seconds</p>
</div>
""",
    unsafe_allow_html=True,
)

# ── Load data + embeddings ────────────────────────────────────────────────────

with st.spinner("🔄 Loading candidate profiles…"):
    df = load_data()

with st.spinner("🧠 Preparing search index…"):
    profile_texts_tuple = tuple(df["profile_text"].tolist())
    embeddings = get_embeddings(profile_texts_tuple)

# ── Search input ──────────────────────────────────────────────────────────────

default_query = selected_example if selected_example else ""

col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input(
        "Search",
        value=default_query,
        placeholder="e.g. senior backend engineer Python fintech",
        label_visibility="collapsed",
    )
with col_btn:
    search_clicked = st.button("Search 🔍", use_container_width=True, type="primary")

# ── Run search ────────────────────────────────────────────────────────────────

if query and (search_clicked or query):
    with st.spinner("⚡ Searching…"):
        q_emb = embed_query(query)
        results = search(query, q_emb, embeddings, df, top_k=top_k)
        results = results[results["score"] >= min_score].reset_index(drop=True)

    # Summary stats
    if len(results):
        avg_score = results["score"].mean()
        top_score = results["score"].iloc[0]
        st.markdown(
            f"""
<div style="margin-bottom:1.5rem">
  <span class="stat-chip">📊 {len(results)} results</span>
  <span class="stat-chip">🏆 Top score: {top_score:.2%}</span>
  <span class="stat-chip">📈 Avg score: {avg_score:.2%}</span>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.info("No candidates matched with the current score threshold. Try lowering it in the sidebar.")

    # ── Result cards ─────────────────────────────────────────────────────────

    for _, row in results.iterrows():
        explanation = generate_explanation(query, row)
        score_pct = float(row["score"])
        sem_pct   = float(row["semantic_score"])
        kw_pct    = float(row["keyword_score"])

        name  = str(row.get("name", "Unknown")).strip() or "Unknown"
        title = str(row.get("title", "")).strip()

        # Score bar colour
        if score_pct >= 0.65:
            bar_color = "#10b981"   # green
        elif score_pct >= 0.40:
            bar_color = "#f59e0b"   # amber
        else:
            bar_color = "#6366f1"   # indigo

        with st.container():
            st.markdown(
                f"""
<div class="card">
  <span class="card-rank">#{int(row['rank'])}</span>
  <p class="card-name">{name}</p>
  <p class="card-title">{title or "Candidate"}</p>
""",
                unsafe_allow_html=True,
            )

            # Score + breakdown
            c1, c2, c3 = st.columns([3, 1, 1])
            with c1:
                st.markdown(
                    f'<p class="score-label">Relevance score: <strong>{score_pct:.1%}</strong></p>',
                    unsafe_allow_html=True,
                )
                st.progress(min(score_pct, 1.0))
            with c2:
                st.metric("Semantic", f"{sem_pct:.1%}", delta=None)
            with c3:
                st.metric("Keyword", f"{kw_pct:.1%}", delta=None)

            # Explanation
            st.markdown(
                f'<div class="explanation">{explanation}</div>',
                unsafe_allow_html=True,
            )

            # Optional resume snippet
            if show_resume:
                resume_snippet = str(row.get("resume_text", ""))[:800]
                highlighted = highlight_keywords(resume_snippet, query)
                with st.expander("📄 Resume snippet"):
                    st.markdown(highlighted, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

elif not query:
    # Landing placeholder
    st.markdown(
        """
<div style="text-align:center;padding:4rem 0;color:#9ca3af">
  <div style="font-size:3rem;margin-bottom:1rem">🎯</div>
  <p style="font-size:1.1rem;font-weight:500;color:#6b7280">Enter a search query above to find matching candidates</p>
  <p style="font-size:.9rem">Try: <em>"senior Python engineer"</em> · <em>"ML engineer with deployment experience"</em> · <em>"React frontend startup"</em></p>
</div>
""",
        unsafe_allow_html=True,
    )
