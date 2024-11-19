AI YouTube Video Summarizer
Transform any YouTube video into comprehensive notes using AI!
Overview
This application allows users to input a YouTube video URL and receive a detailed summary of the video's content. It utilizes the YouTube Transcript API to fetch transcripts and Google Generative AI to generate summaries in various formats.
Features
Multiple Summary Types: Choose from detailed, quick, or academic summaries.
Video Playback: Watch the video directly within the app.
Transcript Display: View the full transcript of the video.
Download Options: Download the generated summary and full transcript as text and CSV files, respectively.
History Tracking: Access previous summaries for quick reference.
Installation
1.Clone the repository:
   git clone <repository-url>
   cd <repository-directory>
2.Install the required packages:
     pip install -r requirements.txt
3.Set up your environment variables:
Create a .env file in the root directory and add your Google API key:
         GOOGLE_API_KEY=your_google_api_key_here
#Usage
1.Run the application:
  streamlit run app.py
2.Open your web browser and navigate to http://localhost:8501.
3.Enter a YouTube video URL in the input field.
4.Select the desired summary type from the settings.
5.Click on "Generate Summary" to fetch the transcript and generate the summary.
6.View the summary and download options.

Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

