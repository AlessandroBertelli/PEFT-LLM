import streamlit as st
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os
import tempfile

# --- RAG LIBRARIES ---
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ------------------------------------------------------------------------------
# 1. CONFIGURATION & PAGE SETUP
# ------------------------------------------------------------------------------
st.set_page_config(page_title="AI Multi-Mode", page_icon="🤖", layout="wide")

REPO_ID = "abertekth/model"
FILENAME = "mio-modello-q4_k_m.gguf"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# System prompts
SYSTEM_PROMPT_FAST = "You are a helpful AI assistant. Answer the user's questions clearly and concisely."
SYSTEM_PROMPT_RAG = """You are a Retrieval-Augmented Generation (RAG) assistant.
Your answers must be based solely and strictly on the information contained in the retrieved documents provided in the context.
Rules:
1. Do not use any outside knowledge not explicitly present in the retrieved context.
2. If the answer is not supported by the retrieved documents, reply with: "The provided documents do not contain enough information."
3. Cite specific document sections where relevant.
"""

# Initialize Session State for Navigation
if "page" not in st.session_state:
    st.session_state.page = "home"


# ------------------------------------------------------------------------------
# 2. SHARED RESOURCE LOADING (Caching)
# ------------------------------------------------------------------------------
@st.cache_resource
def load_llm():
    """Loads the Llama model once for the whole app."""
    try:
        with st.spinner(f'Downloading LLM ({FILENAME})...'):
            model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

        # Load into RAM (CPU optimized based on your previous file)
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


@st.cache_resource
def load_embedding_model():
    """Loads the embedding model only when needed (RAG mode)."""
    with st.spinner('Loading embedding model...'):
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


# Load LLM immediately as it is needed for both
llm = load_llm()


# ------------------------------------------------------------------------------
# 3. PAGE: HOME (Selection Screen)
# ------------------------------------------------------------------------------
def render_home():
    st.title("🤖 Choose Your AI Version")
    st.markdown("Please select the modality you wish to use:")

    col1, col2 = st.columns(2)

    with col1:
        st.info("⚡ **FAST VERSION**")
        st.write("A simple, high-speed chat with the model. No document upload, just direct conversation.")
        if st.button("Launch Fast Version", use_container_width=True):
            st.session_state.page = "fast"
            st.rerun()

    with col2:
        st.warning("📚 **RAG VERSION**")
        st.write("Upload documents (PDF/TXT) and ask questions based on them. Slower, but accurate to your data.")
        if st.button("Launch RAG Version", use_container_width=True):
            st.session_state.page = "rag"
            st.rerun()


# ------------------------------------------------------------------------------
# 4. PAGE: FAST VERSION (Simple Chat)
# ------------------------------------------------------------------------------
def render_fast_version():
    st.title("⚡ Fast Chat")

    # Back Button
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    # Chat History Setup
    if "messages_fast" not in st.session_state:
        st.session_state.messages_fast = [{"role": "system", "content": SYSTEM_PROMPT_FAST}]

    # Render History
    for message in st.session_state.messages_fast:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask me anything..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages_fast.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Streaming Generation (Standard LLM)
            stream = llm.create_chat_completion(
                messages=st.session_state.messages_fast,
                stream=True,
                max_tokens=512,
                temperature=0.7
            )

            for chunk in stream:
                if "content" in chunk["choices"][0]["delta"]:
                    token = chunk["choices"][0]["delta"]["content"]
                    full_response += token
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        st.session_state.messages_fast.append({"role": "assistant", "content": full_response})


# ------------------------------------------------------------------------------
# 5. PAGE: RAG VERSION (Document Upload + Chat)
# ------------------------------------------------------------------------------
def render_rag_version():
    st.title("📚 RAG Assistant")

    # Back Button
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    # Load Embeddings (Only here, to save RAM for Fast users)
    embeddings = load_embedding_model()

    # Sidebar: Document Management
    with st.sidebar:
        st.header("📂 Upload Documents")
        uploaded_file = st.file_uploader("Upload .txt or .pdf", type=["txt", "pdf"])

        if uploaded_file and "vector_store" not in st.session_state:
            with st.spinner("Indexing documents..."):
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

                    # Create Vector Store
                    vector_store = FAISS.from_documents(chunks, embeddings)
                    st.session_state.vector_store = vector_store
                    st.success(f"Indexed {len(chunks)} chunks!")

                    os.remove(tmp_file_path)
                except Exception as e:
                    st.error(f"Error reading file: {e}")

        if st.button("Clear Chat & Memory"):
            st.session_state.messages_rag = [{"role": "system", "content": SYSTEM_PROMPT_RAG}]
            if "vector_store" in st.session_state:
                del st.session_state.vector_store
            st.rerun()

    # Chat Logic
    if "messages_rag" not in st.session_state:
        st.session_state.messages_rag = [{"role": "system", "content": SYSTEM_PROMPT_RAG}]

    for message in st.session_state.messages_rag:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Ask about your documents..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages_rag.append({"role": "user", "content": prompt})

        # Retrieval Logic
        context_text = ""
        if "vector_store" in st.session_state:
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(prompt)
            context_text = "\n\n".join([d.page_content for d in docs])

            augmented_prompt = f"""Use ONLY the following context to answer the question.
            CONTEXT:
            {context_text}
            QUESTION:
            {prompt}
            """
            messages_for_llm = st.session_state.messages_rag[:-1] + [{"role": "user", "content": augmented_prompt}]
        else:
            messages_for_llm = st.session_state.messages_rag

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            stream = llm.create_chat_completion(
                messages=messages_for_llm,
                stream=True,
                max_tokens=512,
                temperature=0.7
            )

            for chunk in stream:
                if "content" in chunk["choices"][0]["delta"]:
                    full_response += chunk["choices"][0]["delta"]["content"]
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        st.session_state.messages_rag.append({"role": "assistant", "content": full_response})


# ------------------------------------------------------------------------------
# 6. MAIN ROUTER
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    if st.session_state.page == "home":
        render_home()
    elif st.session_state.page == "fast":
        render_fast_version()
    elif st.session_state.page == "rag":
        render_rag_version()