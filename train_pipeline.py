import os
from scripts.dataset import DatasetLoader
from scripts.audio_features import FeatureExtractor
from scripts.model_trainer import HybridMoodClassifier

def main():
    print("=" * 60)
    print("Music Mood Classifier Training Pipeline")
    print("=" * 60)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # Load dataset
    print("\n[Step 1] Loading Dataset")
    print("-" * 60)
    loader = DatasetLoader()
    audio_files, moods = loader.load_dataset(samples_per_class=100)
    
    # Extract features
    print("\n[Step 2] Extracting Audio Features")
    print("-" * 60)
    extractor = FeatureExtractor()
    features_df = extractor.extract_batch(audio_files, moods)
    
    if features_df.empty:
        print("Error: No features extracted!")
        return
    
    # Train model
    print("\n[Step 3] Training Hybrid Classifier")
    print("-" * 60)
    classifier = HybridMoodClassifier()
    accuracy = classifier.train(features_df)
    classifier.save()
    
    # Test predictions
    print("\n[Step 4] Testing Predictions")
    print("-" * 60)
    
    test_cases = [
        ("Classical Music", {
            'tempo': 60, 'energy_mean': 0.02, 'brightness_mean': 1200,
            'harmonic_ratio': 0.8, 'chroma_mean': 0.48
        }),
        ("Rock Music", {
            'tempo': 130, 'energy_mean': 0.15, 'brightness_mean': 3500,
            'harmonic_ratio': 0.4, 'chroma_mean': 0.38
        }),
        ("Pop Music", {
            'tempo': 115, 'energy_mean': 0.08, 'brightness_mean': 3000,
            'harmonic_ratio': 0.65, 'chroma_mean': 0.58
        })
    ]
    
    for name, features in test_cases:
        pred, conf = classifier.predict(features)
        emoji = {'calm': '😌', 'energetic': '⚡', 'happy': '😊'}[pred]
        print(f"{emoji} {name}: {pred.upper()} ({conf:.1%})")
    
    print("\n" + "=" * 60)
    print(f"Training Complete! Final Accuracy: {accuracy:.1%}")
    print("=" * 60)
    print("\nGenerated outputs:")
    print("  • models/mood_classifier.pkl")
    print("  • outputs/confusion_matrix.png")
    print("  • outputs/feature_importance.png")
    print("  • outputs/performance_metrics.png")

if __name__ == "__main__":
    main()