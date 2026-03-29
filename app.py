import streamlit as st
import librosa
import numpy as np
import tempfile
import os
from scripts.model_trainer import HybridMoodClassifier

st.set_page_config(page_title="Music Mood Classifier", page_icon="🎵", layout="wide")

@st.cache_resource
def load_model():
    try:
        return HybridMoodClassifier.load('models/mood_classifier.pkl')
    except:
        st.error("Model not found. Please run the training pipeline first.")
        return None

def extract_features(file_path):
    """Extract audio features from file"""
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=30)
        
        features = {}
        
        # Basic features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo)
        
        rms = librosa.feature.rms(y=y)
        features['energy_mean'] = float(np.mean(rms))
        features['energy_std'] = float(np.std(rms))
        
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['brightness_mean'] = float(np.mean(centroid))
        features['brightness_std'] = float(np.std(centroid))
        
        # Harmonic ratio
        harmonic, percussive = librosa.effects.hpss(y)
        h_rms = np.mean(librosa.feature.rms(y=harmonic))
        p_rms = np.mean(librosa.feature.rms(y=percussive))
        features['harmonic_ratio'] = float(h_rms / (h_rms + p_rms + 1e-8))
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfcc[i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfcc[i]))
        
        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = float(np.mean(chroma))
        features['chroma_std'] = float(np.std(chroma))
        
        # Other features
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['rolloff_mean'] = float(np.mean(rolloff))
        features['rolloff_std'] = float(np.std(rolloff))
        
        return features
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return None

def main():
    st.title("🎵 Music Mood Classifier")
    st.write("Upload an audio file to analyze its mood")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.write("This app classifies music into three moods:")
        st.write("• 😌 **CALM**: Relaxing, peaceful")
        st.write("• 😊 **HAPPY**: Cheerful, upbeat")
        st.write("• ⚡ **ENERGETIC**: Intense, powerful")
        
        st.divider()
        st.write("**How it works:**")
        st.write("1. Upload MP3/WAV file")
        st.write("2. Extract audio features")
        st.write("3. Apply hybrid AI model")
        st.write("4. Get mood prediction")
    
    # Load model
    classifier = load_model()
    if not classifier:
        return
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Upload Audio")
        uploaded_file = st.file_uploader("Choose a file", type=['mp3', 'wav'])
        
        if uploaded_file:
            st.audio(uploaded_file)
            st.caption(f"File: {uploaded_file.name} ({uploaded_file.size/1024/1024:.2f} MB)")
    
    with col2:
        st.subheader("🎯 Results")
        
        if uploaded_file and st.button("Analyze Mood", type="primary"):
            with st.spinner("Analyzing..."):
                # Save temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                try:
                    # Extract and predict
                    features = extract_features(tmp_path)
                    
                    if features:
                        prediction, confidence = classifier.predict(features)
                        
                        # Display result
                        emoji = {'calm': '😌', 'energetic': '⚡', 'happy': '😊'}[prediction]
                        color = {'calm': 'blue', 'energetic': 'orange', 'happy': 'green'}[prediction]
                        
                        st.success(f"## {emoji} {prediction.upper()}")
                        st.metric("Confidence", f"{confidence*100:.1f}%")
                        
                        # Feature display
                        st.divider()
                        st.write("**Key Features:**")
                        
                        feat_col1, feat_col2 = st.columns(2)
                        with feat_col1:
                            st.metric("Tempo", f"{features['tempo']:.0f} BPM")
                            st.metric("Energy", f"{features['energy_mean']:.3f}")
                        with feat_col2:
                            st.metric("Brightness", f"{features['brightness_mean']:.0f}")
                            st.metric("Chroma", f"{features['chroma_mean']:.2f}")
                        
                        # Description
                        st.divider()
                        descriptions = {
                            'calm': "🎼 Relaxing and peaceful music. Good for meditation, studying, or unwinding.",
                            'happy': "🎉 Cheerful and upbeat music. Perfect for parties, driving, or lifting your mood.",
                            'energetic': "🔥 Intense and powerful music. Great for workouts, gaming, or getting energized."
                        }
                        st.info(descriptions[prediction])
                
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
        
        elif not uploaded_file:
            st.info("👆 Upload a file to begin")

    # Footer
    st.divider()
    st.caption("Built with Python, Librosa, and Scikit-learn")

if __name__ == "__main__":
    main()