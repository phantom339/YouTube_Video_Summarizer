from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from google.generativeai import GenerativeModel, configure
from rpunct import RestorePuncts
from urllib.parse import urlparse, parse_qs

# Safe chunk size in characters (adjust based on model; ~50k chars is conservative for ~12k tokens)
MAX_CHUNK_SIZE = 20000
# Approximate max context in chars (e.g., for Gemini flash ~1M tokens * 4 chars/token = 4M chars, but conservative)
MAX_CONTEXT_SIZE = 50000

def configure_genai(api_key, model_name="gemma-3-27b-it"):
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY / GEMINI_API_KEY")
    configure(api_key=api_key)
    # Optional: Safety settings to avoid blocking (adjust as needed)
    return GenerativeModel(model_name)

def get_video_id(url_link):
    parsed = urlparse(url_link)
    if parsed.hostname == 'youtu.be':
        return parsed.path[1:]
    if parsed.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed.path == '/watch':
            return parse_qs(parsed.query)['v'][0]
        if parsed.path[:7] == '/embed/':
            return parsed.path.split('/')[2]
        if parsed.path[:3] == '/v/':
            return parsed.path.split('/')[2]
    raise ValueError("Invalid YouTube URL")

def get_transcript(url_link):
    video_id = get_video_id(url_link)
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id)
        transcript_joined = " ".join([snippet.text for snippet in transcript_list.snippets])
        #rpunct = RestorePuncts()
        #punctuated_transcript = rpunct.punctuate(transcript_joined)
        #return punctuated_transcript
        return transcript_joined
    
    except (NoTranscriptFound, TranscriptsDisabled) as e:
        raise ValueError(f"Transcript not available: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error fetching transcript: {str(e)}")

def chunk_text(text, max_chunk_size=MAX_CHUNK_SIZE):
    """Split text into chunks <= max_chunk_size characters, preferring sentence boundaries."""
    chunks = []
    current_chunk = ""
    sentences = text.split('.')  # Simple split; improve with NLTK if needed
    for sentence in sentences:
        sentence = sentence.strip() + '.' if sentence.strip() else ''
        if len(current_chunk) + len(sentence) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def summarize_chunk(chunk, model):
    """Summarize a single chunk."""
    prompt = f"""
    Provide a concise summary of the following transcript chunk, focusing on key points, main ideas, and any important details. Keep it under 1000 words.
    
    Chunk: {chunk}
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        raise ValueError(f"Error summarizing chunk: {str(e)}")

def generate_summary(transcript, model):
    if len(transcript) <= MAX_CONTEXT_SIZE:
        prompt = f"""
        Summarize the following YouTube video transcript in a concise manner.
        Structure the summary as:
        - TL;DR: A one-sentence overview.
        - Key Points: Bullet list of 5-10 main ideas.
        - Conclusion: Any final thoughts or takeaways.

        Transcript: {transcript}
        """
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "context" in str(e).lower() or "limit" in str(e).lower():
                pass  # Fall through to chunking
            else:
                raise ValueError(f"Error generating summary: {str(e)}")
    
    # Chunk and map-reduce
    chunks = chunk_text(transcript)
    chunk_summaries = [summarize_chunk(chunk, model) for chunk in chunks]
    combined_summaries = "\n\n".join(chunk_summaries)
    if len(combined_summaries) > MAX_CONTEXT_SIZE:
        return generate_summary(combined_summaries, model)  # Recursive
    else:
        prompt = f"""
        Combine and summarize these chunk summaries into a cohesive overall summary.
        Structure as:
        - TL;DR: A one-sentence overview.
        - Key Points: Bullet list of 5-10 main ideas.
        - Conclusion: Any final thoughts or takeaways.

        Chunk Summaries: {combined_summaries}
        """
        response = model.generate_content(prompt)
        return response.text

def extract_relevant_excerpts(chunk, question, model):
    """Extract relevant parts from a chunk based on the question."""
    prompt = f"""
    Extract any parts from the following transcript chunk that are relevant to answering the question: "{question}".
    If nothing is relevant, respond with exactly "None".
    Otherwise, return the relevant excerpts verbatim, separated by newlines.
    
    Chunk: {chunk}
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        return text if text != "None" else ""
    except Exception as e:
        raise ValueError(f"Error extracting excerpts: {str(e)}")

def answer_question(question, transcript, model):
    if len(transcript) <= MAX_CONTEXT_SIZE:
        prompt = f"""
        Answer the following question based on the YouTube video transcript provided.
        Be concise, accurate, and cite relevant parts if possible.
        If the answer isn't in the transcript, say "The transcript does not contain information on this."

        Question: {question}
        Transcript: {transcript}
        """
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "context" in str(e).lower() or "limit" in str(e).lower():
                pass  # Fall through to chunking
            else:
                raise ValueError(f"Error answering question: {str(e)}")
    
    # Chunk and extract relevant excerpts
    chunks = chunk_text(transcript)
    excerpts = []
    for chunk in chunks:
        excerpt = extract_relevant_excerpts(chunk, question, model)
        if excerpt:
            excerpts.append(excerpt)
    
    combined_excerpts = "\n\n".join(excerpts)
    if not combined_excerpts:
        return "The transcript does not contain information on this."
    
    if len(combined_excerpts) > MAX_CONTEXT_SIZE:
        # If still too long, recurse by treating excerpts as new "transcript"
        return answer_question(question, combined_excerpts, model)
    
    prompt = f"""
    Answer the following question based on the provided relevant excerpts from the YouTube video transcript.
    Be concise, accurate, and cite relevant parts if possible.
    If the answer isn't in the excerpts, say "The transcript does not contain information on this."

    Question: {question}
    Excerpts: {combined_excerpts}
    """
    response = model.generate_content(prompt)
    return response.text