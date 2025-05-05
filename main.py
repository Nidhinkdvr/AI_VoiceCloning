import streamlit as st
import torch
import os
import sounddevice as sd
import wave
from TTS.api import TTS
from pydub import AudioSegment
from datetime import datetime
import numpy as np
import plotly.graph_objects as go

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load TTS model
@st.cache_resource
def load_tts():
    return TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

tts = load_tts()

# Streamlit UI
st.title("AI Voice Cloning")
st.markdown("Convert text to speech using a cloned voice.")

# Language selection
language = st.selectbox("Select Language", ["English"], index=0)

# Function to record audio
def record_audio(filename, duration=5, samplerate=22050):
    st.info("Recording... Speak now!")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(recording.tobytes())
    st.success("Recording complete!")

# Section for uploading or recording speaker audio
st.subheader("Upload or Record Speaker Audio")

# Upload speaker audio file
uploaded_audio = st.file_uploader("Upload a speaker audio file (.wav)", type=["wav"])

# Record button for speaker audio
record_button = st.button("Record Voice")
recorded_audio_path = "recorded_speaker.wav"

if record_button:
    record_audio(recorded_audio_path, duration=5)
    st.audio(recorded_audio_path, format="audio/wav")

# Section for uploading text file or entering text
st.subheader("Upload or Enter Text")

# Text input box
text_input = st.text_area("Enter the text to convert to speech", "")

# Upload text file
text_file = st.file_uploader("Or upload a text file (.txt)", type=["txt"])

# Generate button
if st.button("Generate Voice"):
    speaker_audio_path = None

    # Handle recorded audio
    if os.path.exists(recorded_audio_path):
        speaker_audio_path = recorded_audio_path

    # Handle uploaded audio
    elif uploaded_audio is not None:
        speaker_audio_path = "speaker.wav"
        with open(speaker_audio_path, "wb") as f:
            f.write(uploaded_audio.read())

    # Ensure a valid speaker audio file exists
    if not speaker_audio_path or not os.path.exists(speaker_audio_path):
        st.error("Error: No valid speaker audio provided. Please upload or record an audio file.")
        st.stop()

    # Read text from file if uploaded
    if text_file:
        text_input = text_file.read().decode("utf-8")

    if not text_input.strip():
        st.error("Error: Please enter some text or upload a text file.")
        st.stop()

    # Create output directory
    output_dir = "generated_audio"
    os.makedirs(output_dir, exist_ok=True)

    # Generate a unique timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_path = os.path.join(output_dir, f"cloned_{timestamp}.wav")

    try:
        # Run voice cloning
        tts.tts_to_file(text=text_input, speaker_wav=speaker_audio_path, language="en", file_path=output_file_path)
    except Exception as e:
        st.error(f"Error during voice cloning: {e}")
        st.stop()

    # Display audio player
    st.audio(output_file_path, format="audio/wav")
    st.success("✅ Voice cloning completed!")

    # Display interactive waveform of the generated audio
    audio = AudioSegment.from_wav(output_file_path)
    samples = np.array(audio.get_array_of_samples())
    time = np.linspace(0, len(samples) / audio.frame_rate, num=len(samples))

    # Reduce the number of data points for large audio files
    max_points = 5000  # Limit the number of points to 5000 for better performance
    if len(samples) > max_points:
        step = len(samples) // max_points
        samples = samples[::step]
        time = time[::step]

    # Create a plotly graph for the waveform
    fig = go.Figure()

    # Add waveform trace
    fig.add_trace(go.Scatter(x=time, y=samples, mode='lines', line=dict(color='royalblue', width=1)))

    # Set plot title and labels
    fig.update_layout(
        title="Waveform of Generated Audio",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        template="plotly_dark",
        autosize=True,
        hovermode="closest"
    )

    # Display the interactive plot
    st.plotly_chart(fig)
