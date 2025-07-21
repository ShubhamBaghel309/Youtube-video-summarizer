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
        "temperature": 0.2,
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
    except (TranscriptsDisabled, NoTranscriptFound):
        st.info("No YouTube transcript available. Converting audio to text...")
    except Exception as e:
        # Handle XML parsing errors and other unexpected issues
        if "xml" in str(e).lower() or "parse" in str(e).lower():
            st.warning("YouTube transcript data is corrupted. Converting audio to text...")
        else:
            st.warning(f"Error accessing YouTube transcript: {str(e)}. Converting audio to text...")
    
    # Fallback to audio conversion
    try:
        transcript = convert_audio_to_text(video_id)
        if transcript:
            st.success("Successfully converted audio to text!")
            # Cache the transcript
            st.session_state[cache_key] = transcript
            return transcript
        else:
            st.error("Failed to convert audio to text.")
            return None
    except Exception as e:
        st.error(f"Failed to process audio: {str(e)}")
        return None

def chunk_transcript(transcript, chunk_size=2000, overlap=200):
    """Split transcript into smaller chunks with overlap for better context"""
    words = transcript.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for i, word in enumerate(words):
        current_size += len(word) + 1
        current_chunk.append(word)
        
        # Check if we've reached chunk size
        if current_size > chunk_size:
            chunks.append(" ".join(current_chunk))
            # Keep overlap words for context
            overlap_start = max(0, len(current_chunk) - overlap)
            current_chunk = current_chunk[overlap_start:]
            current_size = sum(len(word) + 1 for word in current_chunk)
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def generate_chunk_summary(chunk: str, summary_type: str, retry_count=0):
    """Generate summary for a chunk with improved topic focus and accuracy"""
    max_retries = 3
    base_delay = 60
    
    try:
        # Enhanced prompt with topic extraction and key concepts focus
        prompt = f"""First, identify the main topics and concepts discussed in this transcript chunk.
Then, create a {summary_type} summary following these strict rules:

1. Focus on the actual concepts being taught/explained
2. Include specific technical terms and their explanations
3. Maintain the exact teaching sequence from the video
4. Preserve specific examples and implementations mentioned
5. Do not add information that isn't explicitly in the transcript
6. If code or mathematical concepts are mentioned, include them precisely

If word embeddings, neural networks, or machine learning concepts are discussed:
- Explain their specific implementation details
- Include any mathematical notations or formulas mentioned
- Preserve technical accuracy of the concepts

Transcript chunk to analyze:

{chunk}

Format the response maintaining technical precision and using markdown for structure.
"""
        
        response = model.generate_content(prompt)
        summary = response.text if response.text else None
        
        # Validate the generated summary
        if summary:
            validation_prompt = f"""Verify this summary's technical accuracy:

Original chunk:
{chunk}

Generated summary:
{summary}

Check for:
1. Missing technical concepts
2. Incorrect explanations
3. Omitted implementation details
4. Lost mathematical notations
5. Skipped examples

List any inaccuracies found."""

            validation = model.generate_content(validation_prompt)
            
            if "inaccurac" in validation.text.lower() or "missing" in validation.text.lower():
                # Regenerate with correction guidance
                correction_prompt = f"""Regenerate the summary addressing these issues:
{validation.text}

Original chunk:
{chunk}

Focus on fixing the identified problems while maintaining the same structure."""

                corrected_response = model.generate_content(correction_prompt)
                return corrected_response.text
            
        return summary
        
    except Exception as e:
        if retry_count < max_retries and ("429" in str(e) or "Resource has been exhausted" in str(e)):
            wait_time = base_delay * (2 ** retry_count)
            st.warning(f"Rate limit reached. Waiting {wait_time} seconds before retry {retry_count + 1}/{max_retries}...")
            time.sleep(wait_time)
            return generate_chunk_summary(chunk, summary_type, retry_count + 1)
        raise e

def generate_summary(transcript, summary_type):
    """Generate summary with intelligent chunking based on transcript length"""
    video_id = extract_video_id(st.session_state.get('current_video_url', ''))
    
    # Check database first
    if video_id:
        existing_summary = db.get_summary(video_id, summary_type.lower())
        if existing_summary:
            st.success("Retrieved existing summary from database!")
            return existing_summary
    
    # Define maximum length for direct processing
    MAX_DIRECT_LENGTH = 4000  # Adjust this based on model's capacity
    
    # Check if transcript is short enough for direct processing
    if len(transcript) <= MAX_DIRECT_LENGTH:
        try:
            st.info("Processing video in single pass...")
            
            prompt = f"""Analyze this complete transcript and provide a {summary_type} summary that:
1. Captures the main topic and core concepts
2. Maintains the original structure and sequence
3. Includes all technical terms and their explanations
4. Preserves specific examples and implementations
5. Ensures technical accuracy throughout

If the content involves technical concepts (e.g., machine learning, programming):
- Include precise technical details and definitions
- Preserve any mathematical notations or formulas
- Maintain accuracy of technical implementations

Transcript to analyze:

{transcript}

Format the response with proper markdown structure."""

            response = model.generate_content(prompt)
            summary = response.text if response.text else None
            
            if summary and validate_summary(summary, transcript):
                # Store in database
                if video_id:
                    if db.store_summary(
                        video_id,
                        summary_type.lower(),
                        summary,
                        st.session_state.get('current_video_url', '')
                    ):
                        st.success("Summary stored in database for future use!")
                return summary
            else:
                st.warning("Direct processing produced low-quality summary. Trying chunk processing...")
                # Fall through to chunk processing
        except Exception as e:
            st.warning(f"Direct processing failed: {str(e)}. Falling back to chunk processing...")
    else:
        st.info(f"Transcript length ({len(transcript)} chars) exceeds direct processing limit. Using chunk processing...")
      # If we reach here, either the transcript is too long or direct processing failed
    chunks = chunk_transcript(transcript, chunk_size=2000, overlap=200)
    
    if len(chunks) > 1:
        st.info(f"Processing video in {len(chunks)} parts...")
    
    # Process chunks with improved context
    chunk_summaries = []
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    for i, chunk in enumerate(chunks):
        try:
            # Include overlap with previous chunk for better context
            context = ""
            if i > 0 and chunk_summaries:
                context = f"Previous context: {chunk_summaries[-1]}\n\n"
            
            summary = generate_chunk_summary(chunk, summary_type)
            if summary:
                chunk_summaries.append(summary)
                progress = ((i + 1) / len(chunks))
                progress_bar.progress(progress)
                progress_text.text(f"Processed {i + 1}/{len(chunks)} chunks...")
                
                if i < len(chunks) - 1:
                    time.sleep(2)
        except Exception as e:
            st.error(f"Failed to process chunk {i}: {str(e)}")
            return None

    # Combine chunks if necessary
    if len(chunks) > 1:
        try:
            combined_prompt = f"""Combine these summaries into one coherent {summary_type} summary.
Follow these rules:
1. Maintain the original video's structure and flow
2. Ensure logical transitions between sections
3. Preserve specific examples and definitions
4. Remove redundant information
5. Use consistent formatting throughout
6. Maintain technical accuracy and precision

Summaries to combine:

{"\n".join(chunk_summaries)}"""

            response = model.generate_content(combined_prompt)
            final_summary = response.text
        except Exception as e:
            st.error(f"Error combining summaries: {str(e)}")
            return None
    else:
        final_summary = chunk_summaries[0]

    # Store in database
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
    """Chat with the bot about the video content with improved context and search."""
    try:
        # First, check if the transcript actually contains relevant content
        search_prompt = f"""Analyze if this transcript contains information relevant to: "{question}"
If it does, specify which parts are relevant. If it doesn't, clearly state that the topic is not covered.

Transcript:
{transcript}

Response format:
1. Contains relevant information: [Yes/No]
2. Relevant sections: [Quote the specific parts]
"""
        search_response = model.generate_content(search_prompt)
        
        if "no" in search_response.text.lower() and "not covered" in search_response.text.lower():
            return "This topic is not covered in the video. The video appears to discuss different subjects."
        
        # If relevant content exists, generate a detailed answer
        answer_prompt = f"""Based on this video transcript, provide a detailed answer to the question.

Transcript:
{transcript}

Question: {question}

Requirements:
1. Answer ONLY based on information present in the transcript
2. If the exact answer isn't in the transcript, say so
3. Include relevant quotes or examples from the transcript
4. Maintain technical accuracy
5. If the transcript discusses related concepts, explain how they connect to the question

Format your response in a clear, structured way."""

        response = model.generate_content(answer_prompt)
        return response.text if response.text else "Sorry, I couldn't generate a response."
        
    except Exception as e:
        st.error(f"Error in chat processing: {str(e)}")
        return "Sorry, there was an error processing your question. Please try again."

def validate_summary(summary, transcript):
    """Validate summary with enhanced technical accuracy checking"""
    try:
        validation_prompt = f"""Perform a thorough technical validation of this summary against the transcript.

Original transcript:
{transcript[:2000]}...

Generated summary:
{summary}

Evaluate based on these criteria:
1. Technical Accuracy Score (0-100)
2. Concept Coverage Score (0-100)
3. Missing Technical Terms or Concepts
4. Implementation Details Accuracy
5. Mathematical/Code Accuracy
6. Examples and Use Cases Coverage

Provide specific improvements needed if any score is below 95.
Format response as:
Technical Score: X
Coverage Score: Y
Issues:
- [List specific issues]
Improvements:
- [List specific improvements]"""
        
        response = model.generate_content(validation_prompt)
        validation_result = response.text
        
        # Extract scores
        technical_score = int(re.search(r'Technical Score: (\d+)', validation_result).group(1))
        coverage_score = int(re.search(r'Coverage Score: (\d+)', validation_result).group(1))
        
        if technical_score < 95 or coverage_score < 95:
            st.warning(f"""Summary needs improvement:
Technical Score: {technical_score}
Coverage Score: {coverage_score}

{validation_result}""")
            return False
            
        return True
            
    except Exception as e:
        st.error(f"Validation error: {str(e)}")
        return True

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
                
                # Add a clear cache button
                if st.button("Clear Cache"):
                    # Clear all summary related cache
                    keys_to_clear = [k for k in st.session_state.keys() if k.startswith("summary_")]
                    for k in keys_to_clear:
                        del st.session_state[k]
                    st.success("Cache cleared!")
                
                # Store summary in session state
                summary_key = f"summary_{video_id}_{summary_type.lower()}"
                
                if st.button("Generate Summary"):
                    # Clear existing summary for this video and type
                    if summary_key in st.session_state:
                        del st.session_state[summary_key]
                    
                    with st.spinner("Generating summary..."):
                        summary = generate_summary(transcript, summary_type.lower())
                        if summary:
                            # Cache the new summary
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

    # Add this near the top of the UI section, after video URL input
    if video_id:
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("Clear All Caches"):
                # Clear session state
                keys_to_clear = [k for k in st.session_state.keys() if k.startswith(("summary_", "transcript_"))]
                for k in keys_to_clear:
                    del st.session_state[k]
                
                # Clear database
                db.clear_summary(video_id)
                
                st.success("All caches cleared! Generate a new summary.")
