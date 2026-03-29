# 🎵 Music Mood Classifier

## 📌 Overview

The **Music Mood Classifier** is a machine learning project that analyzes audio signals and predicts the emotional mood of a song.
It classifies music into **Happy, Energetic and Calm** using audio feature extraction and a trained ML model.

This project includes:

* End-to-end training pipeline
* Model evaluation with visual outputs
* Interactive web app using Streamlit

---

## 🎯 Key Features

* 🎧 Audio feature extraction using Librosa
* 🤖 Machine learning model (Random Forest)
* 📊 Performance evaluation (confusion matrix, metrics)
* 💾 Model saving and reuse
* 🌐 Interactive Streamlit web application

---

## 📂 Dataset

* **Name:** GTZAN Dataset - Music Genre Classification
* **Source:** Kaggle
* **Link:** https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

### 📥 Dataset Setup

1. Download the dataset from the link above
2. Extract the ZIP file
3. Navigate to `genres_original/`
4. Copy all `.wav` files into the `data/` folder

> ⚠️ Dataset is not included in this repository due to size limitations

---

## ⚙️ Installation & Setup

### 1️⃣ Install Python

* Use **Python 3.11**
* ⚠️ Python 3.13 is not supported (librosa compatibility issue)

---

### 2️⃣ Install Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib librosa soundfile tqdm streamlit
```

---

## 🚀 How to Run the Project

### 🔹 Step 1: Train the Model

Run the full pipeline:

```bash
python train_pipeline.py
```

This will:

* Extract features from audio files
* Train the model
* Save the model in `models/mood_classifier.pkl`
* Generate evaluation outputs in `outputs/`

---

### 🔹 Step 2: Launch the Web App

```bash
streamlit run app.py
```

This will:

* Open a browser-based UI
* Allow audio file upload (`.wav`)
* Predict the mood of the music

---

## 📁 Project Structure

```bash
MUSIC_MOOD/
│── data/                      # Dataset (not included)
│── models/                   # Trained model
│   └── mood_classifier.pkl
│── outputs/                  # Graphs & evaluation results
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── performance_metrics.png
│── scripts/                  # Core ML modules
│   ├── audio_features.py
│   ├── dataset.py
│   ├── model_trainer.py
│── train_pipeline.py         # Training pipeline
│── app.py                    # Streamlit UI
│── README.md
```

---

## 🧠 Tech Stack

* **Librosa** – Audio feature extraction
* **Scikit-learn** – Machine learning (Random Forest)
* **Pandas & NumPy** – Data processing
* **Matplotlib & Seaborn** – Visualization
* **Joblib** – Model persistence
* **Streamlit** – Web interface

---

## 📊 Workflow

```text
Audio Files (.wav)
        ↓
Feature Extraction (Librosa)
        ↓
Feature Dataset
        ↓
Model Training (Random Forest)
        ↓
Evaluation & Visualization
        ↓
Saved Model (.pkl)
        ↓
Streamlit App → Mood Prediction
```

---

## ⚠️ Important Notes

* Ensure `.wav` files are inside `data/` before training
* Run `train_pipeline.py` before launching the app
* Model file must exist in `models/` for predictions
* Dataset is not uploaded to GitHub

---

## 📌 Future Improvements

* 🎨 Improve UI/UX of Streamlit app
* ☁️ Deploy using AWS / Docker
* 🧠 Use deep learning (CNN/RNN) for better accuracy
* 📱 Build mobile app integration

---

## 👨‍💻 Author

**Praneet A R**

---

## ⭐ Acknowledgements

* GTZAN Dataset contributors
* Open-source ML and audio processing libraries

---

## 📬 Contact

Feel free to connect for collaboration or suggestions!
