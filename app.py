

import streamlit as st
import os
import torch
import torchaudio
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from jiwer import wer

# Load ground truth transcriptions from a CSV file
ground_truth_path = "D:/Project/streamlit/cv-valid-train-with-threats.csv"
gt_df = pd.read_csv(ground_truth_path)

# Load pretrained model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Load threat words
threat_words = pd.read_csv("D:/Project/streamlit/threaten_word.csv")["Words/Phrases"].str.lower().tolist()

# FGSM Attack (simplified version)
def fgsm_attack(audio, epsilon=0.002):
    audio_tensor = torch.tensor(audio, dtype=torch.float32)
    perturbed = audio_tensor + epsilon * torch.sign(torch.randn_like(audio_tensor))
    return perturbed.numpy()

#  PGD attack
def pgd_attack(audio, epsilon=0.002, alpha=0.0005, iters=5):
    audio_tensor = torch.tensor(audio, dtype=torch.float32)
    original = audio_tensor.clone()
    for _ in range(iters):
        perturb = alpha * torch.sign(torch.randn_like(audio_tensor))
        audio_tensor = audio_tensor + perturb
        audio_tensor = torch.clamp(audio_tensor, original - epsilon, original + epsilon)
    return audio_tensor.numpy()

# Pre-emphasis filter
def pre_emphasis(audio, coeff=0.97):
    emphasized_audio = np.append(audio[0], audio[1:] - coeff * audio[:-1])
    return emphasized_audio

# Transcribe function
def transcribe(audio, sample_rate):
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0].lower()
        return transcription

# Threat word detection
def detect_threat_words(transcription):
    found =[word for word in threat_words if word in transcription]
    return found

# Streamlit UI
st.title("🎧 Adversarial Attacks & Defense in Speech Recognition")
st.write("Upload, apply attacks, transcribe with Wav2Vec2, and detect threat words.")

uploaded_file = st.file_uploader("Upload an MP3/WAV audio file", type=["wav", "mp3"])
model_option = st.selectbox("Choose Model", ["Wav2Vec2 (pretrained)"])
attack_option = st.selectbox("Choose Adversarial Attack", ["None", "FGSM", "PGD"])
defense_option = st.selectbox("Apply Defense Technique", ["None", "Pre-Emphasis"])

if uploaded_file:
    st.audio(uploaded_file)

    ground_truth = None
    filename = os.path.basename(uploaded_file.name)
    # Match filename (case-insensitive)
    gt_df['base_filename'] = gt_df['filename'].apply(lambda x: os.path.basename(x))
    matched = gt_df[gt_df['base_filename'].str.lower() == filename.lower()]

    if not matched.empty:
        ground_truth = matched['text'].values[0].lower()
        st.success(f"✅ Ground truth loaded for `{filename}`")
        st.markdown("**Ground Truth:**")
        st.write(f"{ground_truth}")
       
    else:
        st.warning("No matching ground truth found for this file.")

    # Load and resample audio
    audio, sr = librosa.load(uploaded_file, sr=16000)
    st.write(f"Audio duration: {len(audio)/sr:.2f} sec")

    # Keep original for comparison
    original_audio = audio.copy()
    attacked_audio = audio.copy() 

# Apply attack
    if attack_option == "FGSM":
        audio = fgsm_attack(audio)
        st.warning("FGSM attack applied.")
    elif attack_option == "PGD":
        audio = pgd_attack(audio)
        st.warning("PGD attack applied.")

# Apply defense
    if defense_option == "Pre-Emphasis":
        audio = pre_emphasis(audio)
        st.info("Pre-emphasis defense applied.")

# Transcription
    #transcription = transcribe(audio, 16000)

# Metrics 
    final_transcription = transcribe(audio, 16000)  # possibly defended
    attacked_transcription = transcribe(attacked_audio, 16000)  # no defense
    clean_transcription = transcribe(original_audio, 16000)

    st.subheader("📝 Transcription:")
    
    st.markdown("**Original (Clean) Transcript:**")
    st.info(clean_transcription)

    st.markdown(f"**Attacked Transcript ({attack_option}):**")
    st.warning(attacked_transcription)

    st.markdown(f"**Final Transcript ({defense_option} Defense):**")
    st.success(final_transcription)

    st.subheader("🔍 Threat Detection")
    detected_attacked = detect_threat_words(attacked_transcription)
    detected_defended = detect_threat_words(final_transcription)

    st.markdown(f"**Threats in Final ({defense_option}) Transcript:**")
    if detected_defended:
        st.error(f"⚠️ Detected: {', '.join(detected_defended)}")
    else:
        st.success("✅ No threat words found.")

    st.subheader("📊 Evaluation Metrics")
    if ground_truth:
        wer_attacked = wer(ground_truth, attacked_transcription)
        wer_defended = wer(ground_truth, final_transcription)

        st.write(f"- WER (Attacked vs Ground Truth): `{wer_attacked:.2f}` | Accuracy: `{(1 - wer_attacked) * 100:.2f}%`")
        st.write(f"- WER (Final - After {defense_option} vs Ground Truth): `{wer_defended:.2f}` | Accuracy: `{(1 - wer_defended) * 100:.2f}%`")
    else:
        st.warning("No ground truth available for evaluation.")

# Waveform Plots
    st.subheader("📈 Waveform Comparison")

    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].plot(original_audio)
    ax[0].set_title("Original Audio")
    ax[1].plot(audio)
    ax[1].set_title(f"Attacked Audio ({attack_option})")
    plt.tight_layout()
    st.pyplot(fig)