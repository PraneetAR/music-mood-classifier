import os
import numpy as np
from collections import defaultdict

class DatasetLoader:
    def __init__(self, data_path='./data/genres_original'):
        self.data_path = data_path
        self.genre_to_mood = {
            'classical': 'calm',
            'jazz': 'calm',
            'blues': 'calm',
            'country': 'happy',
            'pop': 'happy',
            'reggae': 'happy',
            'rock': 'energetic',
            'metal': 'energetic',
            'disco': 'energetic',
            'hiphop': 'energetic'
        }
    
    def load_dataset(self, samples_per_class=100):
        """Load and balance the dataset"""
        audio_files = []
        moods = []
        
        print("Loading audio files...")
        for genre in os.listdir(self.data_path):
            genre_path = os.path.join(self.data_path, genre)
            if not os.path.isdir(genre_path):
                continue
            
            for file in os.listdir(genre_path):
                if file.endswith(('.wav', '.au')):
                    audio_files.append(os.path.join(genre_path, file))
                    mood = self.genre_to_mood.get(genre.lower(), 'happy')
                    moods.append(mood)
        
        print(f"Found {len(audio_files)} audio files")
        
        # Balance dataset
        return self._balance_dataset(audio_files, moods, samples_per_class)
    
    def _balance_dataset(self, files, moods, samples_per_class):
        """Balance classes"""
        mood_groups = defaultdict(list)
        for file, mood in zip(files, moods):
            mood_groups[mood].append(file)
        
        print("\nClass distribution:")
        for mood, file_list in mood_groups.items():
            print(f"  {mood}: {len(file_list)} files")
        
        balanced_files = []
        balanced_moods = []
        
        for mood in ['calm', 'energetic', 'happy']:
            if mood in mood_groups:
                available = len(mood_groups[mood])
                n_samples = min(samples_per_class, available)
                selected = np.random.choice(mood_groups[mood], n_samples, replace=False)
                balanced_files.extend(selected)
                balanced_moods.extend([mood] * n_samples)
        
        print(f"\nBalanced dataset: {len(balanced_files)} files")
        return balanced_files, balanced_moods