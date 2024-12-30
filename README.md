# 🎥 YouTube Video Summarizer

An advanced YouTube video processing tool that generates summaries, enables chat interactions, and provides transcription capabilities using AI.

## 🌟 Features

### 1. Video Summaries
- Generate three types of summaries:
  - Short: Concise overview
  - Detailed: Comprehensive summary
  - Bullet Points: Key points in list format
- Automatic caching of summaries for faster retrieval
- Download summaries as text files
- Database storage for previously generated summaries

### 2. Interactive Chat
- Ask questions about video content
- AI-powered responses based on video context
- Natural language processing for accurate answers

### 3. Transcription Capabilities
- Primary: YouTube's built-in transcripts
- Fallback: AI-powered audio transcription using Whisper
- Support for multiple languages
- Full transcript viewing option

### 4. Advanced Audio Processing
- Chunk-based audio processing for long videos
- GPU acceleration support (when available)
- Progress tracking with time estimates
- Memory-efficient streaming capabilities

### 5. Database Integration
- ChromaDB integration for summary storage
- Fast retrieval of previously processed videos
- Efficient embedding-based storage system

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- FFmpeg installed on your system
- GPU (optional, but recommended for faster processing)

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd youtube-video-summarizer
```

2. Install FFmpeg:
- Windows:
```bash
winget install ffmpeg
```
- macOS:
```bash
brew install ffmpeg
```
- Ubuntu/Debian:
```bash
sudo apt update && sudo apt install ffmpeg
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file with:
```
GOOGLE_API_KEY=your_google_api_key_here
```

### Running the Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the displayed URL (typically http://localhost:8501)

## 💡 Usage

1. **Generate Summary**
   - Paste a YouTube URL
   - Select summary type (Short/Detailed/Bullet Points)
   - Click "Generate Summary"
   - Download the summary if needed

2. **Chat with Video**
   - Enter your question about the video
   - Get AI-generated responses based on video content

3. **View Transcript**
   - Access the full transcript in the dedicated tab
   - Automatically uses YouTube's transcript or generates one if unavailable

## ⚙️ Technical Details

### Components
- **Streamlit**: Web interface
- **Google Gemini Pro**: AI model for summaries and chat
- **Whisper**: Speech-to-text conversion
- **ChromaDB**: Vector database for summary storage
- **yt-dlp**: YouTube video processing

### Performance Features
- Chunk-based processing for large videos
- Caching system for faster repeated access
- Progress tracking with time estimates
- Error handling and retry mechanisms
- Rate limit management

## 📝 Notes

- Processing time depends on video length and system capabilities
- GPU acceleration significantly improves transcription speed
- Internet connection required for YouTube access and AI processing
- Some videos may not be accessible due to privacy settings or regional restrictions

## 🔧 Troubleshooting

- **FFmpeg Issues**: Ensure FFmpeg is properly installed and accessible from command line
- **Memory Errors**: Long videos are processed in chunks to manage memory
- **GPU Issues**: Check CUDA installation if using GPU acceleration
- **API Limits**: The system includes retry mechanisms for rate limits

## 📄 License

[Your chosen license]

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 👏 Acknowledgments

- OpenAI Whisper for transcription capabilities
- Google Gemini Pro for AI processing
- Streamlit for the web interface
- ChromaDB for vector storage
