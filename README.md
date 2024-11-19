---

# **AI YouTube Video Summarizer**  
Transform any YouTube video into comprehensive notes using AI!

---

## **Overview**  
This application simplifies note-taking by generating detailed summaries of YouTube videos. With the power of the **YouTube Transcript API** and **Google Generative AI**, it creates summaries in various formats based on your needs.

---

## **Features**  
- **Multiple Summary Types**: Choose from detailed, quick, or academic summaries.  
- **Video Playback**: Watch the video directly within the app.  
- **Transcript Display**: View the full transcript alongside the video.  
- **Download Options**: Download summaries as text files and transcripts as CSV files.  
- **History Tracking**: Access previous summaries for quick reference.

---

## **Installation**  
Follow these steps to get started:  

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/ai-youtube-summarizer.git
   cd ai-youtube-summarizer
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:  
   - Create a `.env` file in the root directory.  
   - Add your **Google API key**:
     ```plaintext
     GOOGLE_API_KEY=your_google_api_key_here
     ```

---

## **Usage**  
1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to:  
   [http://localhost:8501](http://localhost:8501)

3. Enter a **YouTube video URL** in the input field.  
4. Select the desired **summary type**.  
5. Click **"Generate Summary"** to fetch the transcript and create a summary.  
6. View the summary and download options directly on the app.

---

## **Contributing**  
We welcome contributions! 🚀  
- **Submit a pull request** for enhancements or fixes.  
- Open an issue for bugs or feature suggestions.

---
