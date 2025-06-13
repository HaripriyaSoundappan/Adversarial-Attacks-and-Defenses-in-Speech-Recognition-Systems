# 🎧 Adversarial Attacks and Defenses in Speech Recognition Systems

This project investigates the vulnerabilities of speech recognition systems to adversarial attacks and explores defense strategies to enhance their robustness. 
By applying various perturbation techniques and defense mechanisms, we aim to understand and mitigate the impact of malicious manipulations on transcription accuracy.

---

## 📌 Project Objective

- Analyze how adversarial attacks (FGSM, PGD) affect speech recognition performance.
- Evaluate the resilience of pretrained (Wav2Vec2).
- Propose and test defenses such as signal enhancement, denoising, and robust training.
- Measure effectiveness using metrics like Word Error Rate (WER), and accuracy.

---

## 🧠 Key Features

- ✅ Preprocessing with Common Voice dataset
- ✅ Implementation of adversarial attacks: FGSM, PGD, background noise injection
- ✅ Defense mechanisms: noise reduction, audio filtering, adversarial training
- ✅ Model comparison: pretrained Wav2Vec2 vs. custom CNN-RNN
- ✅ Evaluation dashboard using Streamlit

---

## 🗃️ Dataset

- **Name**: Mozilla Common Voice  
- **Format**: `.mp3` audio clips with corresponding text  
- **Languages**: English (subset used)  
- [🔗 Download Here]([https://commonvoice.mozilla.org/en/datasets])(https://www.kaggle.com/datasets/mozillaorg/common-voice)

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Hugging Face Transformers (Wav2Vec2)
- NumPy, Pandas, Librosa
- Streamlit (for UI)
- Matplotlib, Seaborn (for visualization)

---

## 🚀 How to Run the Project

1. Install dependencies:
  !pip install torchaudio transformers librosa jiwer -q

2. Run the Streamlit app:
     streamlit run app.py
