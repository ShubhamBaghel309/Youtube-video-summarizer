import streamlit as st
# Must be the first Streamlit command
st.set_page_config(page_title="chat with video", layout="wide")

import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import google.generativeai as genai
import os
from dotenv import load_dotenv
from speech_to_text import convert_audio_to_text
import time
from db_operations import SummaryDatabase

# Load environment variables
load_dotenv()

# Verify API key
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")
    st.stop()

# Initialize Gemini
try:
    genai.configure(api_key=google_api_key)
    # Configure model settings
    generation_config = {
        "temperature": 0.7,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }
    
    safety_settings = {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }
    
    model = genai.GenerativeModel(
        model_name='gemini-pro',
        generation_config=generation_config,
        safety_settings=safety_settings
    )
except Exception as e:
    st.error(f"Error initializing Gemini: {str(e)}")
    st.stop()

# Initialize database
db = SummaryDatabase()

def extract_video_id(url):
    """Extract YouTube video ID from URL."""
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_transcript(video_id):
    """Get transcript from YouTube video with caching."""
    # Check if transcript exists in session state
    cache_key = f"transcript_{video_id}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
        
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([t["text"] for t in transcript_list])
        # Cache the transcript
        st.session_state[cache_key] = transcript
        return transcript
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        st.info("No YouTube transcript available. Converting audio to text...")
        transcript = convert_audio_to_text(video_id)
        if transcript:
            st.success("Successfully converted audio to text!")
            # Cache the transcript
            st.session_state[cache_key] = transcript
            return transcript
        else:
            st.error("Failed to convert audio to text.")
            return None

def chunk_transcript(transcript, chunk_size=2000):
    """Split transcript into smaller chunks to avoid rate limits"""
    words = transcript.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_size += len(word) + 1
        if current_size > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def generate_chunk_summary(chunk: str, summary_type: str, retry_count=0):
    """Generate summary for a chunk with retry logic"""
    max_retries = 3
    base_delay = 60  # Base delay of 60 seconds
    
    try:
        prompt = f"Summarize this transcript chunk in {summary_type} style. Be concise and focus on key points:\n\n{chunk}"
        response = model.generate_content(prompt)
        return response.text if response.text else None
    except Exception as e:
        if retry_count < max_retries and ("429" in str(e) or "Resource has been exhausted" in str(e)):
            wait_time = base_delay * (2 ** retry_count)  # Exponential backoff
            st.warning(f"Rate limit reached. Waiting {wait_time} seconds before retry {retry_count + 1}/{max_retries}...")
            time.sleep(wait_time)
            return generate_chunk_summary(chunk, summary_type, retry_count + 1)
        raise e

def generate_summary(transcript, summary_type):
    """Generate summary with database caching and improved chunk processing"""
    video_id = extract_video_id(st.session_state.get('current_video_url', ''))
    
    # Check if summary exists in database
    if video_id:
        existing_summary = db.get_summary(video_id, summary_type.lower())
        if existing_summary:
            st.success("Retrieved existing summary from database!")
            return existing_summary
    
    # Process in smaller chunks
    chunks = chunk_transcript(transcript)
    
    if len(chunks) > 1:
        st.info(f"Processing video in {len(chunks)} parts...")
    
    chunk_summaries = []
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    start_time = time.time()
    
    # Process chunks sequentially with delay between chunks
    for i, chunk in enumerate(chunks):
        try:
            summary = generate_chunk_summary(chunk, summary_type)
            if summary:
                chunk_summaries.append(summary)
                progress = ((i + 1) / len(chunks))
                progress_bar.progress(progress)
                progress_text.text(f"Processed {i + 1}/{len(chunks)} chunks...")
                
                # Add delay between chunks to avoid rate limits
                if i < len(chunks) - 1:
                    time.sleep(2)  # 2-second delay between chunks
        except Exception as e:
            st.error(f"Failed to process chunk {i}: {str(e)}")
            return None
    
    end_time = time.time()
    st.info(f"Processing completed in {end_time - start_time:.2f} seconds")
    
    # Combine summaries if multiple chunks
    if len(chunks) > 1:
        try:
            combined_prompt = (
                f"Combine these {summary_type} summaries into one coherent summary. "
                f"Maintain the same style and remove any redundancy:\n\n" + 
                "\n".join(chunk_summaries)
            )
            response = model.generate_content(combined_prompt)
            final_summary = response.text
        except Exception as e:
            st.error(f"Error combining summaries: {str(e)}")
            return None
    else:
        final_summary = chunk_summaries[0]
    
    # Store the summary in database
    if video_id and final_summary:
        if db.store_summary(
            video_id, 
            summary_type.lower(), 
            final_summary, 
            st.session_state.get('current_video_url', '')
        ):
            st.success("Summary stored in database for future use!")
    
    return final_summary

def chat_with_bot(transcript, question):
    """Chat with the bot about the video content."""
    prompt = f"""Given the following video transcript:

{transcript}

Please answer this question: {question}"""

    try:
        response = model.generate_content(prompt)
        return response.text if response.text else "Sorry, I couldn't generate a response."
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None

# Streamlit UI
st.title("🎥 YouTube Video Summarizer")
st.write("Get summaries and chat about any YouTube video!")

# Video URL input
video_url = st.text_input("Enter YouTube Video URL:")

if video_url:
    # Store current video URL in session state
    st.session_state['current_video_url'] = video_url
    
    video_id = extract_video_id(video_url)
    
    if video_id:
        # Display video
        st.video(video_url)
        
        # Get transcript with progress indicator
        with st.spinner("Getting transcript..."):
            transcript = get_transcript(video_id)
        
        if transcript:
            # Create tabs for different functionalities
            tab1, tab2, tab3 = st.tabs(["📝 Summaries", "💬 Chat", "📜 Full Transcript"])
            
            with tab1:
                st.subheader("Video Summaries")
                summary_type = st.radio(
                    "Choose summary type:",
                    ["Short", "Detailed", "Bullet Points"],
                    horizontal=True
                )
                
                # Store summary in session state
                summary_key = f"summary_{video_id}_{summary_type.lower()}"
                
                if st.button("Generate Summary"):
                    if summary_key in st.session_state:
                        st.markdown(st.session_state[summary_key])
                    else:
                        with st.spinner("Generating summary..."):
                            summary = generate_summary(transcript, summary_type.lower())
                            if summary:
                                # Cache the summary
                                st.session_state[summary_key] = summary
                                st.markdown(summary)
                                
                                # Download button for summary
                                try:
                                    st.download_button(
                                        "Download Summary",
                                        summary,
                                        file_name="video_summary.txt",
                                        mime="text/plain"
                                    )
                                except Exception as e:
                                    st.error("Failed to create download button. Please try generating the summary again.")
            
            with tab2:
                st.subheader("Chat about the Video")
                user_question = st.text_input("Ask a question about the video:")
                
                if user_question:
                    with st.spinner("Thinking..."):
                        response = chat_with_bot(transcript, user_question)
                        if response:
                            st.markdown(response)
            
            with tab3:
                st.subheader("Full Transcript")
                st.text_area("Video Transcript", transcript, height=300)
        else:
            st.error("Sorry, couldn't retrieve the transcript for this video. It might be disabled or unavailable.")
    else:
        st.error("Invalid YouTube URL. Please check the URL and try again.")
