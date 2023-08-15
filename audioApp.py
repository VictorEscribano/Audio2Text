import streamlit as st
import tempfile
import os
import time
import matplotlib.pyplot as plt
import whisper
from langcodes import Language
import torch

# Function to transcribe audio

def transcribe_audio(audio_path, model_size):
    
    # Transcription logic here
    model = whisper.load_model(model_size)
    
    #check if device is cuda
    if model.device.type == "cuda":
        # print GPU Available. Using [tyoe and info of the gpu]
        st.write(f"Using GPU: {model.device}")
        # name of the gpu and memory usage
        st.write(f"GPU Name: {torch.cuda.get_device_name(0)}")
        st.write(f"GPU Memory Usage: {torch.cuda.memory_allocated(0) / 1024 ** 2:.0f} MB")
    else:
        st.write("CPU not vailable at the moment")


    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)    

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    return result.text, detected_language

# Streamlit app
def main():
    st.title("Audio Transcription")

    uploaded_file = st.file_uploader("Upload an audio file (.mp3)")

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/mp3")

        # Save the uploaded audio file to a temporary location
        temp_audio = tempfile.NamedTemporaryFile(delete=False)
        temp_audio.write(uploaded_file.read())
        temp_audio_path = temp_audio.name
        temp_audio.close()

        # Model size selection dropdown
        model_size = st.selectbox("Select Model Size", ["tiny", "base", "small", "medium", "large"], index=3)

        # Transcribe audio and display text
        if st.button("Transcribe"):
            # Display loading animation while transcribing
            with st.spinner("Transcribing..."):
                transcribed_text, detected_language = transcribe_audio(temp_audio_path, model_size)
                time.sleep(2)  # Simulate processing time

            # Display detected language and transcribed text in a box
            st.subheader("Detected Language:")
            try:
                language = Language.get(detected_language)
                full_language_name = language.display_name()
            except:
                full_language_name = "Unknown"
            st.write(full_language_name)

            st.subheader("Transcribed Text:")
            st.text_area("Copy Transcribed Text", value=transcribed_text, height=150)

        # Clean up temporary audio file
        os.unlink(temp_audio_path)


if __name__ == "__main__":
    main()
