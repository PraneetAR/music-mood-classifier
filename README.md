# 🎵 FMA Music Mood Classifier

A machine learning system that analyzes audio features from the Free Music Archive (FMA) dataset to automatically detect whether a song conveys an ENERGETIC, HAPPY, or RELAXED mood.

The model extracts acoustic characteristics including tempo, rhythm, spectral features, MFCCs, and harmony metrics, then uses a Random Forest classifier to predict the emotional mood independent of lyrics. The system achieves 65-75% accuracy on the FMA Small dataset (8,000 tracks, 30-second clips).

## Mood Categories

- **ENERGETIC:** Fast tempo (>120 BPM), strong rhythmic drive, high spectral centroid, high percussive energy
- **HAPPY:** Major key signatures, bright timbre, moderate-to-high energy, positive melodic contours  
- **RELAXED:** Slow tempo (<90 BPM), smooth dynamics, low percussive energy, high harmonic ratio

## Key Features Extracted

- Tempo (BPM) and beat strength
- RMS energy (mean and standard deviation)
- Spectral centroid, rolloff, and bandwidth
- 20 MFCC coefficients (mean and standard deviation)
- Chroma features for harmony analysis
- Zero crossing rate
- Harmonic-percussive ratio

The trained model can predict mood for any uploaded audio file (MP3, WAV, FLAC) and provides confidence scores and probability distributions across all three mood categories.
