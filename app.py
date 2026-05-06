import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# ==========================================
# PAGE CONFIG (must be first Streamlit call)
# ==========================================
st.set_page_config(
    page_title="ScholarRAG — Interactive RAG Explorer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ---------- Global ---------- */
html, body {
    font-family: 'Inter', sans-serif;
}
/* Apply font to Streamlit main app without breaking icon fonts */
.stApp {
    font-family: 'Inter', sans-serif;
}
.block-container { padding-top: 2rem; }

/* ---------- Level Header Badges ---------- */
.level-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 8px;
}
.level-badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: #fff;
}
.badge-l1 { background: linear-gradient(135deg, #4FC3F7, #0288D1); }
.badge-l2 { background: linear-gradient(135deg, #CE93D8, #7B1FA2); }
.badge-l3 { background: linear-gradient(135deg, #FFAB91, #E64A19); }
.badge-l4 { background: linear-gradient(135deg, #A5D6A7, #2E7D32); }

/* ---------- Under-the-Hood Cards ---------- */
.debug-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 16px 20px;
    margin: 6px 0;
}
.debug-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #888;
    margin-bottom: 4px;
}

/* ---------- Metric Cards ---------- */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 8px 0 4px 0;
}
.metric-label {
    font-size: 0.8rem;
    color: #aaa;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e1117 0%, #161b22 100%);
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08);
}

/* ---------- Gradient Divider ---------- */
.gradient-divider {
    height: 3px;
    border-radius: 2px;
    margin: 16px 0 24px 0;
}
.div-l1 { background: linear-gradient(90deg, #4FC3F7, transparent); }
.div-l2 { background: linear-gradient(90deg, #CE93D8, transparent); }
.div-l3 { background: linear-gradient(90deg, #FFAB91, transparent); }
.div-l4 { background: linear-gradient(90deg, #A5D6A7, transparent); }

/* ---------- Chunk viewer ---------- */
.chunk-box {
    background: rgba(255,255,255,0.03);
    border-left: 3px solid #4FC3F7;
    padding: 10px 14px;
    margin: 6px 0;
    border-radius: 0 6px 6px 0;
    font-size: 0.82rem;
    line-height: 1.55;
    max-height: 200px;
    overflow-y: auto;
}

/* ---------- Score bar ---------- */
.score-bar-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 6px;
    height: 10px;
    width: 100%;
    margin: 4px 0 8px 0;
}
.score-bar-fill {
    height: 10px;
    border-radius: 6px;
}

/* Space out radio options */
.st-emotion-cache-1c7n2ri {
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# CACHED MODEL LOADERS
# ==========================================
@st.cache_resource(show_spinner="Loading Bi-Encoder model...")
def load_bi_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource(show_spinner="Loading Cross-Encoder re-ranker...")
def load_cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


@st.cache_resource(show_spinner="Connecting to Gemini LLM...")
def load_llm(api_key):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash')


# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
def init_state():
    defaults = {
        "current_pdf_name": None,
        "pdf_path": None,
        # Per-level: collection refs, chat histories, ingestion flags
        "l1_collection": None, "l1_chat": [], "l1_ingested_pdf": None,
        "l2_collection": None, "l2_chat": [], "l2_ingested_pdf": None, "l2_pages": 0,
        "l3_collection": None, "l3_chat": [], "l3_ingested_pdf": None,
        "l4_results": None, "l4_per_question": None,
        "l5_collection": None, "l5_chat": [], "l5_ingested_pdf": None,
        "user_api_key": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ==========================================
# HELPER: Save uploaded PDF to disk
# ==========================================
def handle_pdf_upload(uploaded_file):
    """Save uploaded PDF and reset collections if the file changed."""
    if uploaded_file is None:
        return

    new_name = uploaded_file.name
    if new_name != st.session_state.current_pdf_name:
        # New file — save to disk
        upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        save_path = os.path.join(upload_dir, new_name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.current_pdf_name = new_name
        st.session_state.pdf_path = save_path
        # Reset all level data so they re-ingest on next use
        for prefix in ["l1", "l2", "l3", "l5"]:
            st.session_state[f"{prefix}_collection"] = None
            st.session_state[f"{prefix}_chat"] = []
            st.session_state[f"{prefix}_ingested_pdf"] = None
        st.session_state.l4_results = None
        st.session_state.l4_per_question = None


# ==========================================
# HELPER: Render debug chunks
# ==========================================
def render_chunks(chunks, accent_color="#4FC3F7"):
    for i, chunk in enumerate(chunks):
        st.markdown(
            f'<div class="chunk-box" style="border-left-color:{accent_color};">'
            f'<strong>Chunk {i+1}</strong><br>{chunk[:500]}{"..." if len(chunk)>500 else ""}</div>',
            unsafe_allow_html=True,
        )


def render_score_bar(label, score, color):
    pct = max(0, min(100, score * 100))
    st.markdown(f"""
    <div style="margin-bottom:2px"><strong>{label}</strong>: {pct:.1f}%</div>
    <div class="score-bar-bg">
        <div class="score-bar-fill" style="width:{pct}%;background:{color};"></div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# LEVEL RENDERERS
# ==========================================

# ---------- LEVEL 1 ----------
def render_level1(bi_encoder, llm):
    from core import level1_naive

    st.markdown("""
    <div class="level-header">
        <h1 style="margin:0;">Naive RAG</h1>
        <span class="level-badge badge-l1">LEVEL 1</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="gradient-divider div-l1"></div>', unsafe_allow_html=True)

    with st.expander("📐 How this pipeline works", expanded=False):
        st.markdown("""
**Naive RAG** is the simplest possible retrieval pipeline — the "hello world" of RAG.

| Stage | Technique |
|---|---|
| **Chunking** | Fixed 1000-character slices with 100-char overlap (often breaks mid-sentence) |
| **Embedding** | `all-MiniLM-L6-v2` Bi-Encoder |
| **Retrieval** | Top-3 cosine similarity search |
| **Synthesis** | Gemini 2.5 Flash with a strict anti-hallucination prompt |

**Limitations:** The fixed chunking strategy is blind to paragraph boundaries, which means a retrieved chunk may contain only half a thought. There is no query optimization — what the user types is exactly what gets searched.
        """)

    # Ensure PDF is ingested
    if st.session_state.pdf_path is None:
        st.info("⬅️ Upload a PDF in the sidebar to get started.")
        return

    if st.session_state.l1_ingested_pdf != st.session_state.current_pdf_name:
        with st.spinner("⏳ Ingesting PDF with Naive chunking..."):
            _, collection, n = level1_naive.ingest(st.session_state.pdf_path, bi_encoder)
            st.session_state.l1_collection = collection
            st.session_state.l1_ingested_pdf = st.session_state.current_pdf_name
        st.success(f"✅ Ingested **{st.session_state.current_pdf_name}** into {n} fixed chunks.")

    # Chat history
    for msg in st.session_state.l1_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "debug" in msg:
                with st.expander("🔍 Under the Hood — Retrieved Chunks"):
                    render_chunks(msg["debug"]["retrieved_chunks"], "#4FC3F7")

    # Chat input
    if prompt := st.chat_input("Ask a question (Level 1)..."):
        st.session_state.l1_chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching & synthesizing..."):
                result = level1_naive.query(prompt, st.session_state.l1_collection, bi_encoder, llm)
            st.markdown(result["answer"])
            with st.expander("🔍 Under the Hood — Retrieved Chunks"):
                render_chunks(result["retrieved_chunks"], "#4FC3F7")

        st.session_state.l1_chat.append({
            "role": "assistant",
            "content": result["answer"],
            "debug": result,
        })


# ---------- LEVEL 2 ----------
def render_level2(bi_encoder, llm):
    from core import level2_intermediate

    st.markdown("""
    <div class="level-header">
        <h1 style="margin:0;">Intermediate RAG</h1>
        <span class="level-badge badge-l2">LEVEL 2</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="gradient-divider div-l2"></div>', unsafe_allow_html=True)

    with st.expander("📐 How this pipeline works", expanded=False):
        st.markdown("""
**Intermediate RAG** fixes the blind spots of Level 1 with three upgrades:

| Upgrade | What Changed |
|---|---|
| **Smart Chunking** | `RecursiveCharacterTextSplitter` — respects paragraph and sentence boundaries |
| **Metadata Tagging** | Each chunk is tagged with its source page number |
| **Query Expansion** | The LLM generates 3 alternative search terms to cast a wider retrieval net |

**New Capability:** You can filter retrieval by a specific page number, letting you drill into a section of the paper.
        """)

    if st.session_state.pdf_path is None:
        st.info("⬅️ Upload a PDF in the sidebar to get started.")
        return

    if st.session_state.l2_ingested_pdf != st.session_state.current_pdf_name:
        with st.spinner("⏳ Ingesting PDF with smart semantic chunking..."):
            _, collection, n, pages = level2_intermediate.ingest(st.session_state.pdf_path, bi_encoder)
            st.session_state.l2_collection = collection
            st.session_state.l2_ingested_pdf = st.session_state.current_pdf_name
            st.session_state.l2_pages = pages
        st.success(f"✅ Ingested **{st.session_state.current_pdf_name}** ({pages} pages) into {n} semantic chunks with metadata.")

    # Page filter in sidebar
    page_filter = st.sidebar.number_input(
        "🎯 Filter by page (0 = all pages)",
        min_value=0, max_value=st.session_state.l2_pages, value=0, step=1,
        key="l2_page_filter",
    )
    effective_filter = page_filter if page_filter > 0 else None

    # Chat history
    for msg in st.session_state.l2_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "debug" in msg:
                with st.expander("🔍 Under the Hood"):
                    d = msg["debug"]
                    st.markdown("**Expanded Queries:**")
                    for q in d.get("expanded_queries", []):
                        st.markdown(f"- `{q}`")
                    if d.get("page_filter_applied"):
                        st.markdown(f"**Page Filter:** Page {d['page_filter_applied']}")
                    st.markdown(f"**Retrieved {len(d['retrieved_chunks'])} unique chunks:**")
                    render_chunks(d["retrieved_chunks"], "#CE93D8")

    if prompt := st.chat_input("Ask a question (Level 2)..."):
        st.session_state.l2_chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Expanding query & searching..."):
                result = level2_intermediate.query(
                    prompt, st.session_state.l2_collection, bi_encoder, llm, effective_filter
                )
            st.markdown(result["answer"])
            with st.expander("🔍 Under the Hood"):
                st.markdown("**Expanded Queries:**")
                for q in result["expanded_queries"]:
                    st.markdown(f"- `{q}`")
                if effective_filter:
                    st.markdown(f"**Page Filter:** Page {effective_filter}")
                st.markdown(f"**Retrieved {len(result['retrieved_chunks'])} unique chunks:**")
                render_chunks(result["retrieved_chunks"], "#CE93D8")

        st.session_state.l2_chat.append({
            "role": "assistant",
            "content": result["answer"],
            "debug": result,
        })


# ---------- LEVEL 3 ----------
def render_level3(bi_encoder, cross_encoder, llm):
    from core import level3_agentic

    st.markdown("""
    <div class="level-header">
        <h1 style="margin:0;">Agentic RAG</h1>
        <span class="level-badge badge-l3">LEVEL 3</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="gradient-divider div-l3"></div>', unsafe_allow_html=True)

    with st.expander("📐 How this pipeline works", expanded=False):
        st.markdown("""
**Agentic RAG** introduces autonomous decision-making and precision retrieval:

| Stage | Technique |
|---|---|
| **Intent Router** | An LLM classifies the user's intent as `GREETING` or `RESEARCH` before engaging the pipeline |
| **Query Expansion** | 3 alternative search terms (same as Level 2) |
| **Wide Retrieval** | Bi-Encoder fetches top 5 chunks per expanded query |
| **Precision Re-rank** | A **Cross-Encoder** (`ms-marco-MiniLM-L-6-v2`) deeply scores each chunk against the original question |
| **Final Selection** | Only the top 4 highest-scored chunks survive to the synthesis prompt |

**Why this matters:** The Bi-Encoder is fast but shallow. The Cross-Encoder is slow but deeply understands relevance. Combining both gives the best of speed and accuracy.
        """)

    if st.session_state.pdf_path is None:
        st.info("⬅️ Upload a PDF in the sidebar to get started.")
        return

    if st.session_state.l3_ingested_pdf != st.session_state.current_pdf_name:
        with st.spinner("⏳ Ingesting PDF for Agentic RAG..."):
            _, collection, n = level3_agentic.ingest(st.session_state.pdf_path, bi_encoder)
            st.session_state.l3_collection = collection
            st.session_state.l3_ingested_pdf = st.session_state.current_pdf_name
        st.success(f"✅ Ingested **{st.session_state.current_pdf_name}** into {n} semantic chunks.")

    # Chat history
    for msg in st.session_state.l3_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "debug" in msg:
                with st.expander("🔍 Under the Hood"):
                    _render_l3_debug(msg["debug"])

    if prompt := st.chat_input("Ask a question (Level 3)..."):
        st.session_state.l3_chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🧠 Agent is analyzing intent..."):
                result = level3_agentic.query(
                    prompt, st.session_state.l3_collection, bi_encoder, cross_encoder, llm
                )
            st.markdown(result["answer"])
            with st.expander("🔍 Under the Hood"):
                _render_l3_debug(result)

        st.session_state.l3_chat.append({
            "role": "assistant",
            "content": result["answer"],
            "debug": result,
        })


def _render_l3_debug(d):
    route = d.get("route", "UNKNOWN")
    color = "#66BB6A" if route == "GREETING" else "#FFAB91"
    st.markdown(f'**🔀 Router Decision:** <span style="color:{color};font-weight:700">{route}</span>', unsafe_allow_html=True)

    if route == "GREETING":
        st.caption("The agent detected casual conversation — the vector database was NOT queried.")
        return

    st.markdown("**Expanded Queries:**")
    for q in d.get("expanded_queries", []):
        st.markdown(f"- `{q}`")

    st.markdown(f"**Bi-Encoder retrieved {len(d.get('bi_encoder_chunks', []))} chunks (wide net):**")
    render_chunks(d.get("bi_encoder_chunks", []), "#FFAB91")

    scores = d.get("cross_encoder_scores", [])
    if scores:
        st.markdown("**Cross-Encoder Re-ranking Scores:**")
        max_score = max(s for s, _ in scores) if scores else 1
        for score, chunk in scores:
            norm = max(0, min(1, score / max(abs(max_score), 1)))
            bar_color = "#66BB6A" if norm > 0.6 else "#FFA726" if norm > 0.3 else "#EF5350"
            st.markdown(f"""
            <div class="debug-card">
                <div style="display:flex;justify-content:space-between;">
                    <span style="font-size:0.8rem;">{chunk[:120]}...</span>
                    <strong style="color:{bar_color}">{score:.3f}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"**✅ Final {len(d.get('final_chunks', []))} chunks sent to LLM:**")
    render_chunks(d.get("final_chunks", []), "#66BB6A")


# ---------- LEVEL 4 ----------
def render_level4(bi_encoder, cross_encoder, llm):
    st.markdown("""
    <div class="level-header">
        <h1 style="margin:0;">RAG Evaluation</h1>
        <span class="level-badge badge-l4">LEVEL 4</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="gradient-divider div-l4"></div>', unsafe_allow_html=True)

    st.markdown("""
This level **does not chat** — it **grades** the Level 3 pipeline using the **Ragas** framework.

A separate "judge" LLM (Gemini 1.5 Flash) automatically runs 15 predefined questions against the database
and mathematically scores the system on four dimensions:

| Metric | What it Measures |
|---|---|
| **Context Precision** | Did the retriever find the *right* chunks? (signal-to-noise ratio) |
| **Context Recall** | Did the retriever find *all* the relevant chunks? (completeness) |
| **Faithfulness** | Did the answer use *only* the context? (no hallucination) |
| **Answer Relevancy** | Did the answer directly address the user's question? |

> ⏱️ **This process is intentionally live** — it takes several minutes, just like real-world evaluation.
> This lets you experience what a developer goes through when benchmarking a RAG system.
    """)

    if st.session_state.pdf_path is None:
        st.info("⬅️ Upload a PDF in the sidebar to get started.")
        return

    if st.session_state.l3_ingested_pdf != st.session_state.current_pdf_name:
        st.warning("⚠️ The Level 3 database has not been built for the current PDF yet. Please visit **Level 3** first and ask at least one question to trigger ingestion.")
        return

    if st.session_state.l4_results is not None:
        _render_eval_results(st.session_state.l4_results, st.session_state.l4_per_question)
        if st.button("🔄 Re-run Evaluation"):
            st.session_state.l4_results = None
            st.session_state.l4_per_question = None
            st.rerun()
        return

    if st.button("🚀 Run Live Evaluation", type="primary", use_container_width=True):
        from core import level4_evaluate

        progress_bar = st.progress(0, text="Starting evaluation...")
        status_text = st.empty()

        def progress_cb(current, total, message):
            progress_bar.progress(current / total if total > 0 else 0, text=message)
            status_text.caption(f"Step {current}/{total}")

        try:
            overall, per_q = level4_evaluate.run_evaluation(
                st.session_state.l3_collection, bi_encoder, cross_encoder, llm, st.session_state.user_api_key, progress_cb
            )
            progress_bar.progress(1.0, text="✅ Evaluation complete!")
            st.session_state.l4_results = overall
            st.session_state.l4_per_question = per_q
            st.rerun()
        except Exception as e:
            st.error(f"Evaluation failed: {e}")


# ---------- LEVEL 5 ----------
def render_level5(bi_encoder, cross_encoder, llm):
    from core import level5_self_healing_rag

    st.markdown("""
    <div class="level-header">
        <h1 style="margin:0;">Self-Healing RAG</h1>
        <span class="level-badge badge-l3" style="background: linear-gradient(135deg, #FFD54F, #F57F17);">LEVEL 5</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="gradient-divider" style="height:3px;background:linear-gradient(90deg, #FFD54F, transparent);margin:16px 0 24px 0;"></div>', unsafe_allow_html=True)

    with st.expander("📐 How this pipeline works", expanded=False):
        st.markdown("""
**Self-Healing RAG** extends Level 3 by adding a robust fallback mechanism:

| Stage | Technique |
|---|---|
| **Retrieval & Re-rank** | Same as Level 3 (Bi-Encoder + Cross-Encoder) |
| **The Grader Gate** | An LLM acts as a strict grader, evaluating if the retrieved local context *actually answers* the question. |
| **Self-Healing (Web Search)** | If the grader outputs `NO` (the DB failed), the agent dynamically initiates a live **DuckDuckGo search** to fetch the answer from the internet instead. |

**Why this matters:** A typical RAG system will confidently hallucinate or say "I don't know" if the answer isn't in the PDF. This system *heals itself* by fetching external data when the local knowledge base falls short.
        """)

    if st.session_state.pdf_path is None:
        st.info("⬅️ Upload a PDF in the sidebar to get started.")
        return

    if st.session_state.l5_ingested_pdf != st.session_state.current_pdf_name:
        with st.spinner("⏳ Ingesting PDF for Self-Healing RAG..."):
            _, collection, n = level5_self_healing_rag.ingest(st.session_state.pdf_path, bi_encoder)
            st.session_state.l5_collection = collection
            st.session_state.l5_ingested_pdf = st.session_state.current_pdf_name
        st.success(f"✅ Ingested **{st.session_state.current_pdf_name}** into {n} semantic chunks.")

    # Chat history
    for msg in st.session_state.l5_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "debug" in msg:
                with st.expander("🔍 Under the Hood"):
                    _render_l5_debug(msg["debug"])

    if prompt := st.chat_input("Ask a question (Level 5)..."):
        st.session_state.l5_chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🧠 Searching and Grading Context..."):
                result = level5_self_healing_rag.query(
                    prompt, st.session_state.l5_collection, bi_encoder, cross_encoder, llm
                )
            st.markdown(result["answer"])
            with st.expander("🔍 Under the Hood"):
                _render_l5_debug(result)

        st.session_state.l5_chat.append({
            "role": "assistant",
            "content": result["answer"],
            "debug": result,
        })


def _render_l5_debug(d):
    route = d.get("route", "UNKNOWN")
    if route == "GREETING":
        st.markdown('**🔀 Router Decision:** <span style="color:#66BB6A;font-weight:700">GREETING</span>', unsafe_allow_html=True)
        st.caption("The agent detected casual conversation — the vector database was NOT queried.")
        return

    st.markdown('**🔀 Router Decision:** <span style="color:#FFAB91;font-weight:700">RESEARCH</span>', unsafe_allow_html=True)
    
    st.markdown("**Expanded Queries:**")
    for q in d.get("expanded_queries", []):
        st.markdown(f"- `{q}`")

    st.markdown(f"**Bi-Encoder retrieved {len(d.get('bi_encoder_chunks', []))} chunks.**")
    
    grade = d.get("grade")
    healed = d.get("healed")
    
    st.markdown("---")
    if grade == "YES":
        st.markdown('👨‍🏫 **Grader Evaluation:** <span style="color:#66BB6A;font-weight:700">YES</span> (Local context contains the answer)', unsafe_allow_html=True)
        st.markdown(f"**✅ Final {len(d.get('final_chunks', []))} local chunks sent to LLM:**")
        render_chunks(d.get("final_chunks", []), "#66BB6A")
    elif grade == "NO":
        st.markdown('👨‍🏫 **Grader Evaluation:** <span style="color:#EF5350;font-weight:700">NO</span> (Local context failed to answer)', unsafe_allow_html=True)
        if healed:
            st.markdown("🌐 **Self-Healing Triggered:** Initiated DuckDuckGo Web Search")
            st.markdown("**Retrieved Web Context:**")
            render_chunks(d.get("final_chunks", []), "#FFD54F")


def _render_eval_results(overall, per_question):
    st.markdown("### 📊 Overall Scores")
    metric_colors = {
        "context_precision": "#4FC3F7",
        "context_recall": "#CE93D8",
        "faithfulness": "#FFAB91",
        "answer_relevancy": "#A5D6A7",
    }
    cols = st.columns(4)
    for i, (key, color) in enumerate(metric_colors.items()):
        score = overall.get(key, 0)
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{key.replace('_', ' ').title()}</div>
                <div class="metric-value" style="color:{color}">{score*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(score)

    if per_question:
        st.markdown("### 📝 Per-Question Breakdown")
        for i, item in enumerate(per_question):
            with st.expander(f"Q{i+1}: {item['question'][:80]}..."):
                st.markdown(f"**Ground Truth:** {item['ground_truth']}")
                st.markdown(f"**Model Response:** {item['response']}")
                st.caption(f"Chunks used: {item['num_chunks']}")


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("# 📚 ScholarRAG")
    st.caption("Interactive RAG Pipeline Explorer")
    st.markdown("---")

    uploaded = st.file_uploader("Upload a Research Paper", type=["pdf"], help="Upload any PDF to query with RAG")
    handle_pdf_upload(uploaded)

    if st.session_state.current_pdf_name:
        st.success(f"📄 **{st.session_state.current_pdf_name}**")
    else:
        st.info("No PDF loaded yet.")

    st.markdown("---")
    
    api_key_input = st.text_input("Gemini API Key", type="password", help="Enter your Gemini API key from Google AI Studio")
    if api_key_input:
        st.session_state.user_api_key = api_key_input
        
    st.markdown("---")
    st.markdown("### Navigate Levels")

    level = st.radio(
        "Select a RAG Pipeline Level:",
        [
            "Level 1: Naive RAG",
            "Level 2: Intermediate RAG",
            "Level 3: Agentic RAG",
            "Level 4: Evaluation",
            "Level 5: Self-Healing RAG",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#555;font-size:0.75rem;'>"
        "Built with Streamlit · Powered by Gemini</div>",
        unsafe_allow_html=True,
    )


# ==========================================
# MAIN CONTENT ROUTER
# ==========================================
if not st.session_state.user_api_key:
    st.warning("⚠️ Please enter your Gemini API Key in the sidebar to continue.")
    st.stop()

bi_encoder = load_bi_encoder()
llm = load_llm(st.session_state.user_api_key)

if "Level 1" in level:
    render_level1(bi_encoder, llm)
elif "Level 2" in level:
    render_level2(bi_encoder, llm)
elif "Level 3" in level:
    cross_encoder = load_cross_encoder()
    render_level3(bi_encoder, cross_encoder, llm)
elif "Level 4" in level:
    cross_encoder = load_cross_encoder()
    render_level4(bi_encoder, cross_encoder, llm)
elif "Level 5" in level:
    cross_encoder = load_cross_encoder()
    render_level5(bi_encoder, cross_encoder, llm)
