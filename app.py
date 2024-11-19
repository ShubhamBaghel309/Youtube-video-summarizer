import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
import re
import textwrap
from datetime import datetime
import pandas as pd

# Load environment variables and configure API
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Streamlit page configuration
st.set_page_config(
    page_title="AI YouTube Summarizer",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .stAlert {margin-top: 20px;}
        .main {padding: 20px;}
        .stButton>button {width: 100%;}
        .reportview-container {background: #f0f2f6}
        .css-1v0mbdj.etr89bj1 {margin-top: 20px;}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

def extract_video_id(url):
    """Extract video ID from various YouTube URL formats"""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_video_info(video_id):
    """Get video thumbnail and other metadata"""
    thumbnail_url = f"http://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
    return thumbnail_url

def get_transcript(video_id, language='en'):
    """Get video transcript with language support"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
        transcript = ' '.join([d['text'] for d in transcript_list])
        return transcript, transcript_list
    except Exception as e:
        st.error(f"❌ Error fetching transcript: {str(e)}")
        return None, None

def generate_summary(transcript, summary_type="detailed"):
    """Generate summary using Google Gemini Pro with different summary types"""
    prompts = {
        "detailed": """Analyze the following video transcript and provide a comprehensive summary including:
            1. Main Topic and Overview
            2. Key Points and Important Details
            3. Technical Concepts Explained
            4. Practical Applications
            5. Conclusion and Takeaways
            
            Make it detailed but clear and well-structured.""",
        
        "quick": """Provide a concise summary of the main points from this video transcript in bullet points.
            Focus on the core message and key takeaways.""",
        
        "academic": """Create an academic-style summary of this video content including:
            • Abstract
            • Methodology (if applicable)
            • Key Findings
            • Discussion
            • References to related concepts"""
    }
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompts[summary_type] + "\n\nTranscript:\n" + transcript)
        return response.text
    except Exception as e:
        st.error(f"❌ Error generating summary: {str(e)}")
        return None

def save_summary(video_id, summary, summary_type):
    """Save summary to session history"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.append({
        "timestamp": timestamp,
        "video_id": video_id,
        "summary": summary,
        "type": summary_type
    })

def main():
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        summary_type = st.selectbox(
            "Summary Type",
            ["detailed", "quick", "academic"],
            help="Choose the style of summary you want"
        )
        
        st.markdown("---")
        st.markdown("### 📖 Previous Summaries")
        if st.session_state.history:
            for item in st.session_state.history[-5:]:  # Show last 5 summaries
                with st.expander(f"Summary from {item['timestamp']}"):
                    st.write(f"Type: {item['type']}")
                    st.write(item['summary'])

    # Main content
    st.title("📺 AI YouTube Video Summarizer")
    st.markdown("Transform any YouTube video into comprehensive notes using AI!")

    # URL input
    url = st.text_input("🔗 Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")

    if url:
        video_id = extract_video_id(url)
        
        if video_id:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.video(f"https://www.youtube.com/watch?v={video_id}")
            
            with col2:
                thumbnail_url = get_video_info(video_id)
                st.image(thumbnail_url, use_column_width=True)

            if st.button("🎯 Generate Summary", use_container_width=True):
                with st.spinner("🎭 Fetching video transcript..."):
                    transcript, transcript_list = get_transcript(video_id)
                    
                if transcript:
                    with st.spinner("🤖 Generating AI summary..."):
                        summary = generate_summary(transcript, summary_type)
                        
                    if summary:
                        st.success("✨ Summary generated successfully!")
                        
                        # Display summary in a nice format
                        st.markdown("### 📝 Summary")
                        st.markdown(summary)
                        
                        # Save to history
                        save_summary(video_id, summary, summary_type)
                        
                        # Additional features
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("📥 Download Summary"):
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"summary_{video_id}_{timestamp}.txt"
                                with open(filename, 'w') as f:
                                    f.write(summary)
                                st.download_button(
                                    label="Download Summary",
                                    data=summary,
                                    file_name=filename,
                                    mime="text/plain"
                                )
                        
                        with col2:
                            if transcript_list:
                                df = pd.DataFrame(transcript_list)
                                st.download_button(
                                    label="📥 Download Full Transcript",
                                    data=df.to_csv(index=False),
                                    file_name=f"transcript_{video_id}.csv",
                                    mime="text/csv"
                                )
                        
                        # Show full transcript in expander
                        with st.expander("👀 View Full Transcript"):
                            st.markdown(transcript)
        else:
            st.error("❌ Please enter a valid YouTube URL")

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            <p>Made by Shubham Baghel with ❤️ using Streamlit and Google Gemini Pro</p>
            <p>© 2024 AI YouTube Summarizer</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()