from pytube import YouTube
import os
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import streamlit as st
import librosa
import numpy as np
import ffmpeg
import yt_dlp
import time
import requests
from pydub import AudioSegment
import io
import m3u8
import subprocess

def ensure_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        # Try to run ffmpeg -version
        os.system('ffmpeg -version')
        return True
    except Exception:
        st.error("""FFmpeg is not installed. Please install it:
        
1. Windows: Run in PowerShell as administrator:
   ```
   winget install ffmpeg
   ```
   or download from https://www.gyan.dev/ffmpeg/builds/

2. After installing, restart your computer.
""")
        return False

def download_audio(video_id):
    """Download audio from YouTube video using yt-dlp with optimized settings"""
    try:
        url = f'https://www.youtube.com/watch?v={video_id}'
        temp_dir = os.path.join(os.getcwd(), "temp_audio")
        os.makedirs(temp_dir, exist_ok=True)
        output_path = os.path.join(temp_dir, f"{video_id}.%(ext)s")

        # Progress callback
        def progress_hook(d):
            if d['status'] == 'downloading':
                total_bytes = d.get('total_bytes')
                downloaded = d.get('downloaded_bytes', 0)
                if total_bytes:
                    progress = (downloaded / total_bytes)
                    st.progress(progress)
                    st.text(f"Downloading: {progress:.1%}")

        ydl_opts = {
            # Download audio only, lowest quality that maintains speech clarity
            'format': 'worstaudio/worst',
            'outtmpl': output_path,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',  # WAV format for faster processing
                'preferredquality': '32',  # Lower quality for faster download
            }],
            'progress_hooks': [progress_hook],
            'quiet': True,
            'no_warnings': True,
            'extract_audio': True,
            # Additional optimizations
            'socket_timeout': 30,
            'retries': 3,
            'fragment_retries': 3,
            'skip_download': False,
            'overwrites': True,
        }

        download_status = st.empty()
        download_status.info("Starting audio download...")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                duration = info.get('duration', 0)
                
                if duration > 3600:  # If video is longer than 1 hour
                    st.warning(f"Warning: Long video detected ({duration/60:.1f} minutes). Download may take some time.")
                
                download_status.info(f"Downloading audio ({duration/60:.1f} minutes)...")
                ydl.download([url])
                final_path = os.path.join(temp_dir, f"{video_id}.wav")
                
                if os.path.exists(final_path):
                    download_status.success("Audio download successful!")
                    return final_path
                else:
                    download_status.error("Failed to find downloaded audio file")
                    return None
                    
            except Exception as e:
                if "This video is not available" in str(e):
                    download_status.error("Video is not available (private/removed/region-locked)")
                elif "Sign in" in str(e):
                    download_status.error("Video requires authentication (age-restricted)")
                else:
                    download_status.error(f"Download error: {str(e)}")
                return None

    except Exception as e:
        st.error(f"Error in download process: {str(e)}")
        return None

@st.cache_resource
def load_whisper_model():
    """Load Whisper model with caching"""
    try:
        # Check for GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}")
        
        # Use tiny model for faster processing
        model_name = "openai/whisper-tiny"
        st.info(f"Loading {model_name} model...")
        
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        # Move model to GPU if available
        model = model.to(device)
        
        return processor, model
    except Exception as e:
        st.error(f"Error loading Whisper model: {str(e)}")
        return None, None

def process_audio_chunks(audio_path, processor, model, chunk_duration=15):
    """Process audio in chunks to handle memory constraints"""
    try:
        # Check audio file exists
        if not os.path.exists(audio_path):
            st.error("Audio file not found")
            return None
            
        # Load audio file
        st.info("Loading audio file...")
        audio, sr = librosa.load(audio_path, sr=16000)
        duration = len(audio) / sr
        st.info(f"Audio duration: {duration:.2f} seconds")
        
        # Calculate chunk size in samples
        chunk_size = int(chunk_duration * sr)
        
        transcriptions = []
        total_chunks = len(audio) // chunk_size + (1 if len(audio) % chunk_size > 0 else 0)
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_left = st.empty()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        start_time = time.time()
        
        for i in range(0, len(audio), chunk_size):
            chunk_start_time = time.time()
            
            # Get chunk
            chunk = audio[i:i + chunk_size]
            
            # Ensure chunk is not empty and has minimum length
            if len(chunk) < sr:  # Skip chunks shorter than 1 second
                continue
                
            try:
                # Process chunk
                input_features = processor(chunk, sampling_rate=sr, return_tensors="pt").input_features
                input_features = input_features.to(device)
                
                # Generate token ids with fixed parameters
                predicted_ids = model.generate(
                    input_features,
                    max_length=448,  # Fixed max length
                    min_length=1,
                    num_beams=1,
                    do_sample=False
                )
                
                # Decode token ids to text
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                
                if transcription.strip():  # Only add non-empty transcriptions
                    transcriptions.append(transcription)
                
                # Update progress
                chunk_num = (i // chunk_size) + 1
                progress = chunk_num / total_chunks
                progress_bar.progress(progress)
                
                # Estimate time left
                chunk_time = time.time() - chunk_start_time
                time_per_chunk = (time.time() - start_time) / chunk_num
                remaining_chunks = total_chunks - chunk_num
                estimated_time_left = remaining_chunks * time_per_chunk
                
                status_text.text(f"Processing chunk {chunk_num}/{total_chunks}")
                time_left.text(f"Estimated time remaining: {estimated_time_left:.1f} seconds")
                
            except Exception as e:
                st.warning(f"Error processing chunk {(i // chunk_size) + 1}: {str(e)}")
                continue
        
        if not transcriptions:
            st.error("No transcription was generated")
            return None
            
        final_text = " ".join(transcriptions)
        st.success(f"Transcription completed in {time.time() - start_time:.1f} seconds!")
        return final_text
        
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

def get_audio_stream(video_id):
    """Stream audio directly without downloading full video"""
    try:
        # Get the direct audio URL
        cmd = [
            'yt-dlp',
            '-f', 'bestaudio[ext=m4a]',  # Prefer M4A format
            '-g',  # Get direct URL
            f'https://www.youtube.com/watch?v={video_id}'
        ]
        
        url = subprocess.check_output(cmd).decode('utf-8').strip()
        
        # For direct streams
        response = requests.get(url, stream=True)
        chunk_size = 1024 * 1024  # 1MB chunks
        
        def audio_generator():
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    yield chunk
        
        return audio_generator()
            
    except Exception as e:
        st.error(f"Error accessing audio stream: {str(e)}")
        return None

def process_audio_stream(audio_generator, processor, model):
    """Process audio stream in chunks"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        transcriptions = []
        buffer = io.BytesIO()
        
        # Process incoming audio chunks
        for i, chunk in enumerate(audio_generator):
            buffer.write(chunk)
            
            # Process every ~30 seconds of audio
            if buffer.tell() > 16000 * 30 * 2:  # 30 seconds of 16kHz audio in 16-bit
                buffer.seek(0)
                
                try:
                    # Convert audio chunk to proper format using ffmpeg
                    audio = AudioSegment.from_file(buffer, format="mp4")  # or try other formats like "webm"
                    audio = audio.set_frame_rate(16000).set_channels(1)
                    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                    
                    # Normalize samples
                    samples = samples / np.max(np.abs(samples))
                    
                    # Process with Whisper
                    input_features = processor(samples, sampling_rate=16000, return_tensors="pt").input_features
                    input_features = input_features.to(device)
                    
                    predicted_ids = model.generate(
                        input_features,
                        max_length=448,
                        min_length=1,
                        num_beams=1,
                        do_sample=False
                    )
                    
                    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                    if transcription.strip():
                        transcriptions.append(transcription)
                        
                    # Update progress
                    st.text(f"Processed {len(transcriptions)} chunks...")
                    
                except Exception as chunk_error:
                    st.warning(f"Error processing chunk {i}: {str(chunk_error)}")
                    continue
                finally:
                    # Clear buffer for next chunk
                    buffer = io.BytesIO()
        
        # Process any remaining audio
        if buffer.tell() > 0:
            buffer.seek(0)
            try:
                audio = AudioSegment.from_file(buffer, format="mp4")
                audio = audio.set_frame_rate(16000).set_channels(1)
                samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                samples = samples / np.max(np.abs(samples))
                
                input_features = processor(samples, sampling_rate=16000, return_tensors="pt").input_features
                input_features = input_features.to(device)
                
                predicted_ids = model.generate(
                    input_features,
                    max_length=448,
                    min_length=1,
                    num_beams=1,
                    do_sample=False
                )
                
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                if transcription.strip():
                    transcriptions.append(transcription)
            except Exception as final_error:
                st.warning(f"Error processing final chunk: {str(final_error)}")
        
        if not transcriptions:
            st.error("No transcription was generated")
            return None
            
        return " ".join(transcriptions)
        
    except Exception as e:
        st.error(f"Error processing audio stream: {str(e)}")
        return None

def convert_audio_to_text(video_id):
    """Convert video audio to text using Whisper"""
    try:
        # Load Whisper model
        processor, model = load_whisper_model()
        if not processor or not model:
            return None

        # Download audio in a format Whisper can process
        cmd = [
            'yt-dlp',
            '-x',  # Extract audio
            '--audio-format', 'wav',  # Convert to WAV
            '--audio-quality', '0',  # Best quality
            '-o', f'temp_audio_{video_id}.%(ext)s',  # Output filename
            f'https://www.youtube.com/watch?v={video_id}'
        ]

        try:
            st.info("Downloading and converting audio...")
            subprocess.run(cmd, check=True, capture_output=True)
            audio_path = f'temp_audio_{video_id}.wav'

            if not os.path.exists(audio_path):
                st.error("Failed to download audio")
                return None

            # Process audio with Whisper
            st.info("Transcribing audio...")
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Process in smaller chunks
            chunk_size = sr * 30  # 30 seconds
            transcriptions = []
            
            # Create progress tracking elements
            progress_container = st.empty()
            status_container = st.empty()
            time_container = st.empty()
            
            total_chunks = len(audio) // chunk_size + (1 if len(audio) % chunk_size > 0 else 0)
            start_time = time.time()
            
            with progress_container:
                progress_bar = st.progress(0)
            
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                if len(chunk) < sr:  # Skip chunks shorter than 1 second
                    continue
                
                input_features = processor(chunk, sampling_rate=sr, return_tensors="pt").input_features
                input_features = input_features.to("cuda" if torch.cuda.is_available() else "cpu")
                
                predicted_ids = model.generate(
                    input_features,
                    max_length=448,
                    min_length=1,
                    num_beams=1,
                    do_sample=False
                )
                
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                if transcription.strip():
                    transcriptions.append(transcription)
                
                # Update progress
                current_chunk = (i // chunk_size) + 1
                progress = current_chunk / total_chunks
                progress_bar.progress(progress)
                
                # Update status
                status_container.text(f"Processing chunk {current_chunk}/{total_chunks}")
                
                # Update time estimate
                elapsed_time = time.time() - start_time
                time_per_chunk = elapsed_time / current_chunk
                remaining_chunks = total_chunks - current_chunk
                estimated_time_left = remaining_chunks * time_per_chunk
                time_container.text(f"Estimated time remaining: {estimated_time_left:.1f} seconds")

            # Clean up progress displays
            progress_container.empty()
            status_container.empty()
            time_container.empty()
            
            # Clean up audio file
            try:
                os.remove(audio_path)
            except:
                pass

            if not transcriptions:
                st.error("No transcription was generated")
                return None

            final_text = " ".join(transcriptions)
            st.success("Transcription completed!")
            return final_text

        except subprocess.CalledProcessError as e:
            st.error(f"Error downloading video: {e.stderr.decode()}")
            return None
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"Error in conversion process: {str(e)}")
        return None