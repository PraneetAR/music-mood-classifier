```markdown
# Music Mood Classifier

## Project Overview
AI system that detects emotional mood (happy, sad, energetic, calm, neutral) from music using audio feature extraction and machine learning.

## Dataset
**Source:** GTZAN Dataset - Music Genre Classification (Kaggle)  
**URL:** https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification  
**Setup:** Download dataset, extract ZIP, copy all .wav files from `genres_original` folder into `data/` folder in project directory.

## Setup Instructions

**1. Install Python 3.11** (Python 3.13 has compatibility issues with librosa)

**2. Install dependencies:**
```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib librosa soundfile tqdm
```

**3. Extract audio features:**
```bash
python main.py
```

**4. Train machine learning model:**
```bash
python enhanced_model.py
```

**5. Test predictions:**
```bash
python predict_mood.py
```

## Technologies Used
- **Librosa** - Audio feature extraction
- **Scikit-learn** - Random Forest classifier
- **Pandas/NumPy** - Data processing
- **Matplotlib/Seaborn** - Visualization
- **Joblib** - Model persistence

## Notes
- Python 3.11 required (3.13 not supported due to aifc module removal)
- Place all .wav files from `genres_original` folder into `data/` directory before running
```
