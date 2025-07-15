import streamlit as st
from transformers import pipeline
import soundfile as sf
import numpy as np
import requests
import os

# Load the pipeline
st.markdown(
"""
<style>
.stApp {
    background-image: url("https://wallpapercave.com/wp/wp9577482.jpg");
    background-size: cover;
    background-position: center;
}
</style>
""",unsafe_allow_html=True
)

audio_classifier = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
@st.cache_resource
def load_pipeline():
    return pipeline(
        task="audio-classification",
        model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        framework="pt"  # Force PyTorch backend
    )


audio_classifier = load_pipeline()

# Read WAV file without FFmpegpip install streamlit transformers soundfile requests

def read_audio(file_path):
    data, samplerate = sf.read(file_path)
    return data, samplerate

# Streamlit app
st.title("Speech Emotion Recognition ")
st.write("Analyze emotions in speech by uploading an audio file or providing a URL link.")

# File uploader or URL input
uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])
audio_url = st.text_input("Or paste a link to a WAV audio file")

if uploaded_file or audio_url:
    audio_path = None

    # Handle uploaded file
    if uploaded_file:
        audio_path = "temp_uploaded.wav"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.read())
        st.audio(audio_path, format="audio/wav")
        st.write("Processing the uploaded audio file...")

    # Handle audio URL
    elif audio_url:
        try:
            st.write("Downloading the audio file from the URL...")
            response = requests.get(audio_url)
            if response.status_code == 200:
                audio_path = "temp_audio.wav"
                with open(audio_path, "wb") as f:
                    f.write(response.content)
                st.audio(audio_path, format="audio/wav")
            else:
                st.error("Failed to download the audio. Please check the URL.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Perform emotion analysis
    if audio_path:
        try:
            # Read audio data using soundfile
            audio_data, sample_rate = read_audio(audio_path)

            # Pass audio data directly to the pipeline
            inputs = {"array": np.array(audio_data), "sampling_rate": sample_rate}
            predictions = audio_classifier(inputs)

            # Format predictions for display
            st.write("### Emotion Probabilities")
            st.bar_chart({pred["label"]: pred["score"] for pred in predictions})

            # Display the most likely emotion
            dominant_emotion = max(predictions, key=lambda x: x["score"])
            st.write("### Dominant Emotion")
            st.write(f"**{dominant_emotion['label']}** with confidence {dominant_emotion['score'] * 100:.2f}%")
        except Exception as e:
            st.error(f"An error occurred during emotion analysis: {e}")

        # Clean up temporary file
        os.remove(audio_path)
