import os
import yt_dlp
import json
import numpy as np
import librosa
from moviepy.editor import VideoFileClip
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

# Load Whisper model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

def download_youtube_video(video_url, output_folder):
    """Download a YouTube video to the specified output folder."""
    ydl_opts = {'outtmpl': os.path.join(output_folder, '%(title)s.%(ext)s')}
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=True)
        video_title = info_dict.get('title', None)
        video_extension = info_dict.get('ext', None)
    
    video_path = os.path.join(output_folder, f"{video_title}.{video_extension}")
    print(f"Download completed! Video saved to: {video_path}")
    return video_path, video_title

def extract_audio(video_path, output_audio_path):
    """Extract audio from the video file."""
    video_clip = VideoFileClip(video_path)
    video_clip.audio.write_audiofile(output_audio_path)
    video_clip.close()

def split_audio(audio, sr, segment_duration=30):
    """Split audio into chunks of the specified duration (in seconds)."""
    total_duration = len(audio) / sr
    segments = []
    start = 0

    while start < total_duration:
        end = min(start + segment_duration, total_duration)
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segments.append(audio[start_sample:end_sample])
        start = end

    return segments

def transcribe_audio_chunk(audio_chunk, sr):
    """Transcribe a single chunk of audio."""
    inputs = processor(audio_chunk, return_tensors="pt", sampling_rate=sr)

    with torch.no_grad():
        generated_ids = model.generate(inputs.input_features)
    
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription

def transcribe_audio(audio_path, segment_duration=30):
    """Transcribe the entire audio file by splitting it into chunks."""
    audio, sr = librosa.load(audio_path, sr=16000)
    audio_chunks = split_audio(audio, sr, segment_duration)

    transcriptions = []
    for chunk in audio_chunks:
        transcription = transcribe_audio_chunk(chunk, sr)
        transcriptions.append(transcription)

    # Combine all transcriptions
    full_transcription = ' '.join(transcriptions)
    return full_transcription

def transcribe_video(video_path):
    """Complete process of extracting and transcribing audio from video."""
    audio_path = 'extracted_audio.wav'
    
    # Step 1: Extract audio from video
    extract_audio(video_path, audio_path)
    
    # Step 2: Transcribe the extracted audio with chunking
    transcription = transcribe_audio(audio_path, segment_duration=30)
    
    return transcription

def save_transcription_to_json(video_title, transcription, json_file=r"C:\Users\Aviral\Desktop\PAAS INTERN\YTtranscribe\video_transcriptions.json"):
    """Save the transcription to a JSON file in the specified format."""
    # Load existing data
    if os.path.exists(json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)
    else:
        data = {}
    
    # Add new transcription
    data[video_title] = transcription

    # Save updated data back to the JSON file
    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)
    
    print(f"Transcription saved to {json_file}.")

# Example usage
video_link = r"https://www.youtube.com/watch?v=vLxFZZJPqw0&t=17s"
output_folder = r"C:\Users\Aviral\Desktop\PAAS INTERN\YTtranscribe\youtubedown\youtubedown\download"

# Step 1: Download the YouTube video
video_path, video_title = download_youtube_video(video_link, output_folder)

# Verify if the file exists
if os.path.exists(video_path):
    # Step 2: Transcribe the video
    transcription = transcribe_video(video_path)
    print("Transcription:\n", transcription)

    # Step 3: Save the transcription to a JSON file
    save_transcription_to_json(video_title, transcription)
else:
    print(f"Error: The video file was not found at {video_path}")
