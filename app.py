import os
import streamlit as st
from dotenv import load_dotenv
from summary_gen import configure_genai, get_transcript, generate_summary, answer_question

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemma-3-27b-it")

# App title and description
st.title("YouTube Video Summarizer with Q&A")
st.markdown("Enter a YouTube URL to generate a summary and ask questions about the video.")

# Initialize session state
if "model" not in st.session_state:
    try:
        st.session_state.model = configure_genai(GOOGLE_API_KEY, MODEL_NAME)
    except ValueError as e:
        st.error(str(e))
        st.stop()
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "summary" not in st.session_state:
    st.session_state.summary = None

# Input for YouTube URL
url = st.text_input("YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")

# Button to generate summary
if st.button("Generate Summary"):
    if url:
        with st.spinner("Fetching transcript and generating summary..."):
            try:
                st.session_state.transcript = get_transcript(url)
                st.session_state.summary = generate_summary(st.session_state.transcript, st.session_state.model)
                st.success("Summary generated!")
            except ValueError as e:
                st.error(str(e))
    else:
        st.warning("Please enter a valid YouTube URL.")

# Display summary if available
if st.session_state.summary:
    st.subheader("Video Summary")
    st.markdown(st.session_state.summary)

# Q&A section
if st.session_state.transcript:
    st.subheader("Ask Questions About the Video")
    question = st.text_input("Your Question:", placeholder="e.g., What was the main argument?")
    if st.button("Get Answer"):
        if question:
            with st.spinner("Answering your question..."):
                try:
                    answer = answer_question(question, st.session_state.transcript, st.session_state.model)
                    st.markdown("**Answer:**")
                    st.markdown(answer)
                except ValueError as e:
                    st.error(str(e))
        else:
            st.warning("Please enter a question.")