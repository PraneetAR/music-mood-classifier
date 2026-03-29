import librosa
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class FeatureExtractor:
    def __init__(self, sr=22050, duration=30, n_mfcc=13):
        self.sr = sr
        self.duration = duration
        self.n_mfcc = n_mfcc
    
    def _safe_stats(self, data, default_mean=0, default_std=0.1):
        """Calculate mean and std safely"""
        try:
            data = np.array(data).flatten()
            if len(data) == 0:
                return default_mean, default_std
            return float(np.mean(data)), float(np.std(data))
        except:
            return default_mean, default_std
    
    def extract_features(self, file_path):
        """Extract enhanced audio features from file"""
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
            
            # Harmonic ratio (this is where harmonic is defined)
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
            
            # NEW: Additional features for better mood discrimination
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features['spectral_contrast_mean'] = float(np.mean(contrast))
            features['spectral_contrast_std'] = float(np.std(contrast))
            
            # Spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            features['bandwidth_mean'] = float(np.mean(bandwidth))
            features['bandwidth_std'] = float(np.std(bandwidth))
            
            # Spectral flatness
            flatness = librosa.feature.spectral_flatness(y=y)
            features['flatness_mean'] = float(np.mean(flatness))
            features['flatness_std'] = float(np.std(flatness))
            
            # Rhythm patterns (for happy vs energetic distinction)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            features['onset_strength_mean'] = float(np.mean(onset_env))
            features['onset_strength_std'] = float(np.std(onset_env))
            
            # Pulse (beat strength consistency)
            pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
            features['pulse_strength'] = float(np.mean(pulse))
            
            # Tonnetz (tonal features) - FIXED: use harmonic component
            tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)  # Now harmonic is defined
            features['tonnetz_mean'] = float(np.mean(tonnetz))
            features['tonnetz_std'] = float(np.std(tonnetz))
            
            return features
        except Exception as e:
            print(f"Feature extraction failed for {file_path}: {e}")
            return None
    
    def extract_batch(self, audio_files, moods):
        """Extract features for multiple audio files"""
        features_list = []
        
        for i, (file_path, mood) in enumerate(zip(audio_files, moods)):
            print(f"Processing: {i}/{len(audio_files)}")
            
            features = self.extract_features(file_path)
            if features:
                features['mood'] = mood
                features_list.append(features)
        
        if not features_list:
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(features_list)
        
        # Fill NaN values only for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        print(f"✓ Extracted {len(df)} samples with {len(df.columns)-1} features")
        return df