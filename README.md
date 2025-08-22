# YouTube Video Summarizer 🎥📝

A simple yet powerful web application that generates summaries of YouTube videos using transcripts and AI.  
Built with **Python, Streamlit, and YouTube Transcript API**, this tool allows you to extract video transcripts, summarize them into concise text, and even interact with the content via a Q&A module.

---

## 🚀 Features
- Extracts transcripts directly from YouTube videos.  
- Generates clean and punctuated transcripts using `rpunct`.  
- Summarizes long videos using Google Generative AI.  
- Handles large transcripts by splitting into chunks with overlaps.  
- Interactive **Q&A module** – ask questions directly about the video content.  
- Easy-to-use **Streamlit UI** for quick deployment and interaction.

---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/phantom339/YouTube_Video_Summarizer.git
   cd YouTube_Video_Summarizer
   streamlit run app.py
   ```
if you want to use the app on a personal device for learning purpose must have a .env file which will have api keys for LLM. Format for .env file:

GEMINI_API_KEY = Your api key,

GEMINI_MODEL =gemma-3-27b-it
