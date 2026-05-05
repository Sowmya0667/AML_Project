"""
app.py — Streamlit UI for Legal GraphRAG
Run: streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="⚖️ Vidhi: The Justice Engine",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styles ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root { --accent: #4f46e5; }
.title  { 
    text-align:center; 
    font-size:3rem; 
    font-weight:800; 
    background: -webkit-linear-gradient(45deg, #4f46e5, #9333ea);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.sub    { text-align:center; color:#6b7280; font-size:1.1rem; margin-bottom:2rem; }
.badge-graph  { color:#16a34a; font-weight:700; background:#dcfce7; padding:0.2rem 0.5rem; border-radius:4px; font-size:0.8rem;}
.badge-vector { color:#2563eb; font-weight:700; background:#dbeafe; padding:0.2rem 0.5rem; border-radius:4px; font-size:0.8rem;}
.badge-hybrid { color:#9333ea; font-weight:700; background:#f3e8ff; padding:0.2rem 0.5rem; border-radius:4px; font-size:0.8rem;}
.card {
    background: rgba(240, 247, 255, 0.85);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.5);
    border-left: 4px solid #4f46e5;
    padding: 1rem 1.25rem; 
    margin: 0.8rem 0; 
    border-radius: 8px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    font-size: 0.95rem; 
    line-height: 1.6;
    transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
}
.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
}
.stChatMessage { border-radius: 8px; }
.stChatInput { border-radius: 12px !important; }

/* Premium App Background */
.stApp {
    background: linear-gradient(
        135deg,
        #ffffff 0%,
        #fcfdff 35%,
        #f8fbff 70%,
        #f5faff 100%
    );
}

.main .block-container {
    padding-top: 2rem;
    max-width: 900px;
}

/* Float the audio recorder to the bottom right, to sit inside the right edge of the chat input */
iframe[src*="audio_recorder_streamlit"] {
    position: fixed;
    bottom: 30px;
    right: 55px;
    z-index: 999999;
    width: 45px !important;
    height: 45px !important;
    border: none;
    background: transparent;
}
/* Optional: Make the chat input have a little more padding on the right to fit the mic */
div[data-testid="stChatInput"] textarea {
    padding-right: 55px !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Sidebar Navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚖️ Navigation")
    page = st.radio("Go to:", ["🏠 Home", "ℹ️ About Vidhi", "📖 How to Use", "💬 The Chatbot"])
    
    st.divider()
    st.header("⚙️ Configuration")

    st.subheader("Retrieval")
    g_k = st.slider("Graph Top-K",  1, 10, 5)
    v_k = st.slider("Vector Top-K", 1, 10, 5)
    f_k = st.slider("Final Top-K",  1, 10, 6)
    stream_on = st.toggle("Stream response", value=True)

    st.divider()

    st.subheader("📊 Graph Stats")
    if st.button("Refresh"):
        try:
            from src.graph.graph_queries  import GraphQueryEngine
            from src.vector.vector_store  import LegalVectorStore
            s = GraphQueryEngine().stats()
            st.metric("Cases",    s.get("cases",    0))
            st.metric("Judges",   s.get("judges",   0))
            st.metric("Statutes", s.get("statutes", 0))
            st.metric("Concepts", s.get("concepts", 0))
            st.metric("Chunks",   LegalVectorStore().count())
        except Exception as e:
            st.error(str(e))

# ─── Routing ──────────────────────────────────────────────────────────────────

if page == "🏠 Home":
    st.markdown(
        "<div style='display: flex; flex-direction: column; align-items: center; justify-content: center; height: 70vh;'>"
        "<h1 style='font-size: 5rem; font-weight: 900; text-align: center; background: -webkit-linear-gradient(45deg, #4f46e5, #9333ea); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0px;'>⚖️ Vidhi</h1>"
        "<h2 style='font-size: 2.5rem; font-weight: 700; text-align: center; color: #4B5563; margin-top: 0px; margin-bottom: 10px;'>The Justice Engine</h2>"
        "<p style='text-align: center; color: #6B7280; font-size: 1.2rem; margin-bottom: 3rem;'>Hybrid Knowledge Graph + Semantic Search · Indian Supreme Court Judgments</p>"
        "<div style='text-align: center; max-width: 800px; padding: 2rem; background: rgba(255,255,255,0.4); border-radius: 16px; box-shadow: 0 8px 32px 0 rgba(31,38,135,0.07); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.18);'>"
        "<h3 style='font-weight: 400; line-height: 1.6; color: #4B5563; font-style: italic; margin-bottom: 1rem;'>\"Constitution is not a mere lawyers document, it is a vehicle of Life, and its spirit is always the spirit of Age.\"</h3>"
        "<p style='font-weight: 700; color: #6B7280; font-size: 1.1rem; margin: 0;'>— Dr. B.R. Ambedkar</p>"
        "</div></div>", 
        unsafe_allow_html=True
    )

elif page == "ℹ️ About Vidhi":
    st.markdown('<p class="title">ℹ️ About Vidhi</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🔗 Knowledge Graph")
    st.write("Neo4j graph stores Cases, Judges, Statutes and Legal Concepts as nodes connected by CITES, APPLIES, INVOLVES and DECIDED_BY edges. This allows the AI to understand complex legal relationships.")
    
    st.markdown("### 🧠 Semantic Search")
    st.write("ChromaDB stores sentence-transformer embeddings of every judgment chunk. Queries are matched by cosine similarity, enabling meaning-based search over thousands of documents.")
    
    st.markdown("### ⚖️ Hybrid Fusion")
    st.write("Reciprocal Rank Fusion (RRF) perfectly merges the structural graph results and the semantic vector results into a single highly relevant ranked list fed to the LLM.")
    
    st.markdown("### 👁️ Multimodal Vision")
    st.write("Integrated with Groq's Vision API, you can seamlessly upload legal notices and documents to be analyzed in real time alongside standard text queries.")

elif page == "📖 How to Use":
    st.markdown('<p class="title">📖 How to Use</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.info(
        "**Getting started with Vidhi:**  \n\n"
        "1. **Data Ingestion**: Place your `.txt` judgment files in the `data/judgements/` folder and run the backend ingestion script (`uv run python -m src.ingestion.data_loader`).  \n"
        "2. **Open the Chatbot**: Navigate to **💬 The Chatbot** page from the sidebar menu.  \n"
        "3. **Ask Questions**: Type a legal research query in the chat box at the bottom.  \n"
        "4. **Visual Input**: Click the paperclip icon in the chat box to upload an image of a document for analysis.  \n"
        "5. **Voice Input**: Click the microphone floating near the text box to record your query via speech-to-text.  \n"
    )

elif page == "💬 The Chatbot":
    st.markdown('<p class="title">💬 Legal Chatbot</p>', unsafe_allow_html=True)
    
    # ─── Chat Interface ─────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("View Retrieved Sources"):
                    for c in msg["sources"]:
                        st.markdown(f"**{c.get('case_name', 'Unknown')}** ({c.get('case_number', 'N/A')}) - Score: {c.get('score', 0):.3f}")
                        text_snippet = c.get('text', '')[:400] + ("..." if len(c.get('text', '')) > 400 else "")
                        st.text(text_snippet)

    # ─── Chat Input ──────────────────────────────────────────────────────────────
    # We render the audio recorder and use CSS to float it over the bottom-right
    from audio_recorder_streamlit import audio_recorder
    audio_bytes = audio_recorder(text="", icon_size="1.5x")
    if audio_bytes:
        if "last_audio" not in st.session_state or st.session_state.last_audio != audio_bytes:
            with st.spinner("Transcribing..."):
                from src.generation.stt import AudioTranscriber
                transcriber = AudioTranscriber()
                stt_text = transcriber.transcribe(audio_bytes)
                if stt_text:
                    st.session_state.voice_query = stt_text
                    st.session_state.last_audio = audio_bytes
                    st.rerun()

    user_input = st.chat_input("e.g. What is the law on right to speedy trial under Article 21?", accept_file=True)

    query = None
    if user_input:
        if hasattr(user_input, "text"):
            query = user_input.text
            if hasattr(user_input, "files") and user_input.files:
                import base64
                st.session_state.current_image = base64.b64encode(user_input.files[0].getvalue()).decode('utf-8')
            else:
                st.session_state.current_image = None
        else:
            query = user_input
            st.session_state.current_image = None

    if "voice_query" in st.session_state and st.session_state.voice_query:
        query = st.session_state.voice_query
        st.session_state.voice_query = None
        st.session_state.current_image = None

    # ─── Process Query ───────────────────────────────────────────────────────────
    if query:
        # Build the user message content
        if st.session_state.get("current_image"):
            st.session_state.messages.append({
                "role": "user", 
                "content": query,
                "image": st.session_state.current_image
            })
        else:
            st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)
            if st.session_state.get("current_image"):
                st.markdown("*[Image Attached]*")

        from config import config as cfg
        cfg.GRAPH_TOP_K  = g_k
        cfg.VECTOR_TOP_K = v_k
        cfg.FINAL_TOP_K  = f_k

        with st.chat_message("assistant"):
            try:
                import string
                clean_q = query.strip().lower().translate(str.maketrans('', '', string.punctuation))
                if clean_q in {"hi", "hello", "hey", "good morning", "good evening", "good afternoon", "greetings"}:
                    full_text = "Hi. I am your legal Assistant . tell me how may i help you"
                    if stream_on:
                        placeholder = st.empty()
                        import time
                        displayed = ""
                        for word in full_text.split():
                            displayed += word + " "
                            placeholder.markdown(displayed + "▌")
                            time.sleep(0.05)
                        placeholder.markdown(full_text)
                    else:
                        st.markdown(full_text)
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_text, "sources": []})
                    st.stop()

                from src.generation.llm_chain import LegalRAGChain
                
                @st.cache_resource
                def get_chain():
                    return LegalRAGChain()
                    
                chain = get_chain()
                
                history = st.session_state.messages[:-1]

                if stream_on:
                    resp_gen = chain.stream_query(query, history=history, image=st.session_state.get("current_image"))
                    full_text = st.write_stream(resp_gen)
                else:
                    with st.spinner("Retrieving and generating…"):
                        resp = chain.query(query, history=history, image=st.session_state.get("current_image"))
                    full_text = resp.answer
                    st.markdown(full_text)
                    st.caption(f"Graph hits: {resp.graph_hits} | Vector hits: {resp.vector_hits} | Model: {resp.model_used.split('-')[0]}")
                
                # Recover the last result and sources from the chain (fixes missing score and text)
                if hasattr(chain, 'last_result') and chain.last_result:
                    sources = [
                        {
                            "case_name": c.case_name,
                            "case_number": c.case_number,
                            "text": c.text,
                            "score": c.score,
                        }
                        for c in chain.last_result.chunks
                    ]
                else:
                    sources = []
                
                if sources:
                    with st.expander("View Retrieved Sources"):
                        for c in sources:
                            st.markdown(f"**{c.get('case_name', 'Unknown')}** ({c.get('case_number', 'N/A')}) - Score: {c.get('score', 0):.3f}")
                            text_snippet = c.get('text', '')[:400] + ("..." if len(c.get('text', '')) > 400 else "")
                            st.text(text_snippet)

                st.session_state.messages.append({"role": "assistant", "content": full_text, "sources": sources})

            except Exception as e:
                st.error(f"Generation error: {e}")
                st.exception(e)
