import streamlit as st
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os
import tempfile
import gc  # <--- NEW: Garbage Collector to force memory cleanup

# --- RAG LIBRARIES ---
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ------------------------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(page_title="AI Multi-Mode", page_icon="🤖", layout="wide")

REPO_ID = "abertekth/model"
FILENAME = "mio-modello-q4_k_m.gguf"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

if "page" not in st.session_state:
    st.session_state.page = "home"


# ------------------------------------------------------------------------------
# 2. MEMORY MANAGEMENT FUNCTIONS
# ------------------------------------------------------------------------------

def clear_rag_resources():
    """Aggressively removes RAG components from memory."""
    if "vector_store" in st.session_state:
        del st.session_state.vector_store
    if "embedding_model" in st.session_state:
        del st.session_state.embedding_model

    # Force Python to release memory immediately
    gc.collect()
    print("🧹 RAG Memory Cleared")


@st.cache_resource
def load_llm():
    """Loads the Llama model once. This stays in memory."""
    try:
        with st.spinner(f'Downloading LLM...'):
            model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=2,
            verbose=False
        )
        return llm
    except Exception as e:
        st.error(f"Failed to load LLM: {e}")
        return None


# Load LLM immediately (needed for both)
llm = load_llm()


# ------------------------------------------------------------------------------
# 3. PAGE: HOME
# ------------------------------------------------------------------------------
def render_home():
    st.title("🤖 Choose Your AI Version")

    # Ensure RAG memory is cleared when sitting on Home
    clear_rag_resources()

    col1, col2 = st.columns(2)

    with col1:
        st.info("⚡ **FAST VERSION**")
        st.write("Optimized for speed. No documents, limited context window.")
        if st.button("Launch Fast Version", use_container_width=True):
            st.session_state.page = "fast"
            st.rerun()

    with col2:
        st.warning("📚 **RAG VERSION**")
        st.write("Upload documents. Heavy memory usage, slower response.")
        if st.button("Launch RAG Version", use_container_width=True):
            st.session_state.page = "rag"
            st.rerun()


# ------------------------------------------------------------------------------
# 4. PAGE: FAST VERSION (Optimized)
# ------------------------------------------------------------------------------
def render_fast_version():
    st.title("⚡ Fast Chat")

    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    if "messages_fast" not in st.session_state:
        st.session_state.messages_fast = [
            {"role": "system", "content": "You are a helpful AI assistant. Be concise."}
        ]

    for message in st.session_state.messages_fast:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Fast chat..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages_fast.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # OPTIMIZATION: Only send the last 6 messages to the LLM
            # This keeps the "prompt processing" time low.
            recent_history = [st.session_state.messages_fast[0]] + st.session_state.messages_fast[-6:]

            stream = llm.create_chat_completion(
                messages=recent_history,
                stream=True,
                max_tokens=512,
                temperature=0.7
            )

            for chunk in stream:
                if "content" in chunk["choices"][0]["delta"]:
                    full_response += chunk["choices"][0]["delta"]["content"]
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        st.session_state.messages_fast.append({"role": "assistant", "content": full_response})


# ------------------------------------------------------------------------------
# 5. PAGE: RAG VERSION (Standard)
# ------------------------------------------------------------------------------
def render_rag_version():
    st.title("📚 RAG Assistant")

    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    # Lazy Load Embeddings into Session State (NOT CACHED globally)
    if "embedding_model" not in st.session_state:
        with st.spinner('Loading embedding model into RAM...'):
            st.session_state.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    embeddings = st.session_state.embedding_model

    with st.sidebar:
        st.header("📂 Upload Documents")
        uploaded_file = st.file_uploader("Upload .txt or .pdf", type=["txt", "pdf"])

        if uploaded_file and "vector_store" not in st.session_state:
            with st.spinner("Indexing..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file_path = tmp_file.name

                    if uploaded_file.name.endswith(".pdf"):
                        loader = PyPDFLoader(tmp_file_path)
                    else:
                        loader = TextLoader(tmp_file_path, encoding="utf-8")

                    docs = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    chunks = text_splitter.split_documents(docs)

                    vector_store = FAISS.from_documents(chunks, embeddings)
                    st.session_state.vector_store = vector_store
                    st.success(f"Indexed {len(chunks)} chunks!")
                    os.remove(tmp_file_path)
                except Exception as e:
                    st.error(f"Error: {e}")

    # RAG Chat Logic
    if "messages_rag" not in st.session_state:
        st.session_state.messages_rag = [{"role": "system", "content": "You are a RAG assistant."}]

    for message in st.session_state.messages_rag:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Ask about docs..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages_rag.append({"role": "user", "content": prompt})

        context_text = ""
        if "vector_store" in st.session_state:
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(prompt)
            context_text = "\n\n".join([d.page_content for d in docs])

            augmented_prompt = f"Context:\n{context_text}\n\nQuestion:\n{prompt}"
            # RAG needs strict context, so we send the augmented prompt
            # But we can still limit history to previous 2 turns to save RAM
            history_slice = [st.session_state.messages_rag[0]] + st.session_state.messages_rag[-4:]
            messages_for_llm = history_slice[:-1] + [{"role": "user", "content": augmented_prompt}]
        else:
            messages_for_llm = st.session_state.messages_rag

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            stream = llm.create_chat_completion(messages=messages_for_llm, stream=True, max_tokens=512)
            for chunk in stream:
                if "content" in chunk["choices"][0]["delta"]:
                    full_response += chunk["choices"][0]["delta"]["content"]
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        st.session_state.messages_rag.append({"role": "assistant", "content": full_response})


# ------------------------------------------------------------------------------
# 6. ROUTER
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    if st.session_state.page == "home":
        render_home()
    elif st.session_state.page == "fast":
        render_fast_version()
    elif st.session_state.page == "rag":
        render_rag_version()