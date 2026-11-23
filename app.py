
import streamlit as st
from pathlib import Path

from modules.document_loader import load_document
from modules.text_chunker import chunk_documents
from modules.embedding_engine import embed_chunks
from modules.chroma_store import ChromaStore
from modules.answer_engine import AnswerEngine


# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Universal RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

PROJECT_DIR = Path(__file__).resolve().parent
DOCUMENT_DIR = PROJECT_DIR / "documents"
DOCUMENT_DIR.mkdir(exist_ok=True)

SUPPORTED_FILES = [
    "pdf", "docx", "xlsx", "pptx",
    "txt", "md", "csv", "json", "html", "htm"
]


# -----------------------------------
# SESSION STATE
# -----------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------------
# SOURCE CITATION
# -----------------------------------

def format_source(source_info):

    metadata = source_info.get("metadata", {})

    details = [
        source_info.get("source", "Unknown source")
    ]

    for key, label in [
        ("page", "Page"),
        ("sheet", "Sheet"),
        ("row", "Row"),
        ("slide", "Slide"),
        ("paragraph", "Paragraph"),
        ("table", "Table")
    ]:

        if metadata.get(key) is not None:
            details.append(f"{label}: {metadata[key]}")

    return " | ".join(details)


# -----------------------------------
# MAIN HEADER
# -----------------------------------

st.title("🤖 Universal RAG Assistant")

st.write(
    "Upload your documents and ask questions. "
    "Answers are generated using your document evidence."
)

st.caption(
    "Powered by Python, Ollama, Qwen 2.5 and ChromaDB"
)


# -----------------------------------
# SIDEBAR - DOCUMENT UPLOAD
# -----------------------------------

with st.sidebar:

    st.header("📁 Document Upload")

    uploaded_files = st.file_uploader(
        "Choose your documents",
        type=SUPPORTED_FILES,
        accept_multiple_files=True
    )

    if st.button("Process Documents", use_container_width=True):

        if not uploaded_files:

            st.warning("Please upload at least one document.")

        else:

            total_stored = 0
            errors = []

            with st.spinner("Processing your documents..."):

                database = ChromaStore()

                for uploaded_file in uploaded_files:

                    try:

                        file_path = DOCUMENT_DIR / uploaded_file.name

                        file_path.write_bytes(
                            uploaded_file.getvalue()
                        )

                        # Extract document text
                        documents = load_document(str(file_path))

                        # Split extracted text into chunks
                        chunks = chunk_documents(documents)

                        if not chunks:
                            errors.append(
                                f"{uploaded_file.name}: No readable text found."
                            )
                            continue

                        # Generate document embeddings
                        embedded_chunks = embed_chunks(chunks)

                        # Store chunks in ChromaDB
                        stored = database.store_chunks(
                            embedded_chunks
                        )

                        total_stored += stored

                    except Exception as error:

                        errors.append(
                            f"{uploaded_file.name}: {error}"
                        )

            if total_stored > 0:

                st.success(
                    f"Successfully processed {total_stored} chunks!"
                )

            for error in errors:
                st.error(error)

    st.divider()

    database = ChromaStore()

    st.metric(
        "Stored document chunks",
        database.count()
    )

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# -----------------------------------
# DISPLAY CHAT HISTORY
# -----------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("📚 View sources"):

                for source in message["sources"]:
                    st.write("• " + format_source(source))


# -----------------------------------
# CHAT INPUT AND AI RESPONSE
# -----------------------------------

question = st.chat_input(
    "Ask a question about your documents..."
)

if question:

    # Display user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate AI answer
    with st.chat_message("assistant"):

        with st.spinner("Searching documents and generating answer..."):

            try:

                engine = AnswerEngine(top_k=5)

                result = engine.generate_answer(question)

                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                if sources:

                    with st.expander("📚 View sources"):

                        for source in sources:
                            st.write("• " + format_source(source))

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as error:

                error_message = (
                    "Something went wrong while generating the answer. "
                    "Please check the terminal for details."
                )

                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "sources": []
                })

                print("APPLICATION ERROR:", error)
