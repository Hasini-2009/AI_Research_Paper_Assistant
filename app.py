
import streamlit as st
from groq import Groq
from pypdf import PdfReader
import io

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="AI Research Paper Assistant",
    page_icon="📚",
    layout="wide"
)

MODEL_NAME = "openai/gpt-oss-20b"

# ---------------- GROQ CONNECTION ----------------

@st.cache_resource
def get_groq_client():
    api_key = st.secrets["GROQ_API_KEY"]
    return Groq(api_key=api_key, timeout=120.0)

def ask_groq(messages):
    response = get_groq_client().chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_completion_tokens=2048,
        reasoning_effort="low"
    )
    return response.choices[0].message.content

# ---------------- PAGE DESIGN ----------------

st.title("📚 AI Research Paper Assistant")

st.write(
    "Understand research papers easily with the power of AI."
)

st.divider()

# ---------------- PDF UPLOAD ----------------

st.header("📄 Upload Your Research Paper")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file is None:
    st.info("Please upload a research paper PDF to get started.")
    st.stop()

pdf_bytes = uploaded_file.getvalue()

if len(pdf_bytes) > 20 * 1024 * 1024:
    st.error("Please upload a PDF smaller than 20 MB.")
    st.stop()

# Extract text from PDF

try:
    reader = PdfReader(io.BytesIO(pdf_bytes))

    paper_text = "\n\n".join(
        page.extract_text() or ""
        for page in reader.pages
    )

except Exception as e:
    st.error(f"Could not read the PDF: {e}")
    st.stop()

if not paper_text.strip():
    st.error(
        "No readable text was found. "
        "This PDF may contain scanned images instead of selectable text."
    )
    st.stop()

# Keep document within a manageable prompt size

MAX_DOCUMENT_CHARS = 50000
document_text = paper_text[:MAX_DOCUMENT_CHARS]

st.success(
    f"PDF uploaded successfully: {uploaded_file.name}"
)

st.caption(
    f"Pages: {len(reader.pages)} | "
    f"Extracted characters: {len(paper_text):,}"
)

if len(paper_text) > MAX_DOCUMENT_CHARS:
    st.warning(
        "This is a long PDF. Only its first 50,000 characters "
        "will be used in this version."
    )

# ---------------- SUMMARY SECTION ----------------

st.divider()
st.header("📝 Research Paper Summary")

if st.button("Generate Summary", use_container_width=True):

    with st.spinner("Groq is reading your research paper..."):

        try:
            answer = ask_groq([
                {
                    "role": "system",
                    "content": (
                        "You are a research paper assistant. "
                        "Use only the supplied research paper. "
                        "Do not invent information."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Summarize this research paper clearly. "
                        "Include the topic, objectives, methodology, "
                        "major findings, conclusion and limitations.\n\n"
                        + document_text
                    )
                }
            ])

            st.subheader("Paper Summary")
            st.markdown(answer)

        except Exception as e:
            st.error(f"Summary generation failed: {e}")

# ---------------- KEY POINTS SECTION ----------------

st.divider()
st.header("🔑 Important Key Points")

if st.button("Extract Key Points", use_container_width=True):

    with st.spinner("Extracting important points..."):

        try:
            answer = ask_groq([
                {
                    "role": "system",
                    "content": (
                        "You are a research assistant. "
                        "Extract accurate information from the supplied paper. "
                        "Do not invent facts."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Extract the 10 most important key points "
                        "from this research paper. Explain each point "
                        "briefly using clear language.\n\n"
                        + document_text
                    )
                }
            ])

            st.subheader("Key Points to Remember")
            st.markdown(answer)

        except Exception as e:
            st.error(f"Key point extraction failed: {e}")

# ---------------- CHAT SECTION ----------------

st.divider()
st.header("💬 Chat with Your Research Paper")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous conversation

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Ask anything about your research paper..."
)

if question:

    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Groq is thinking..."):

            try:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an AI research paper assistant. "
                            "Answer questions using the supplied paper. "
                            "If the paper does not contain the answer, "
                            "clearly say so. Do not invent facts.\n\n"
                            "RESEARCH PAPER:\n"
                            + document_text
                        )
                    }
                ]

                # Include recent conversation

                messages.extend(
                    st.session_state.chat_history[-10:]
                )

                answer = ask_groq(messages)

                st.markdown(answer)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"Chat response failed: {e}")

st.divider()

st.caption(
    "Powered by Groq AI | Built with Python and Streamlit"
)
