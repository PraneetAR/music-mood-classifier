import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

class HybridMoodClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.selector = None
        self.feature_names = []
        
    def _rule_predict(self, features):
        """Enhanced rule-based prediction with better thresholds"""
        tempo = features.get('tempo', 100)
        energy = features.get('energy_mean', 0.05)
        chroma = features.get('chroma_mean', 0.5)
        brightness = features.get('brightness_mean', 2000)
        harmonic_ratio = features.get('harmonic_ratio', 0.5)
        
        # More sophisticated rules
        calm_score = 0
        energetic_score = 0
        happy_score = 0
        
        # Calm rules
        if tempo < 85:
            calm_score += 1
        if energy < 0.06:
            calm_score += 1
        if harmonic_ratio > 0.7:
            calm_score += 1
        if brightness < 2500:
            calm_score += 1
            
        # Energetic rules
        if tempo > 110:
            energetic_score += 1
        if energy > 0.10:
            energetic_score += 1
        if brightness > 3000:
            energetic_score += 1
        if harmonic_ratio < 0.5:
            energetic_score += 1
            
        # Happy rules
        if 90 <= tempo <= 130:
            happy_score += 1
        if 0.45 <= chroma <= 0.65:
            happy_score += 1
        if 0.05 <= energy <= 0.09:
            happy_score += 1
            
        scores = {'calm': calm_score, 'energetic': energetic_score, 'happy': happy_score}
        max_score = max(scores.values())
        
        if max_score >= 3:  # Strong rule match
            prediction = max(scores, key=scores.get)
            confidence = min(0.85, 0.6 + (max_score * 0.08))
            return prediction, confidence
        
        return None, 0
    
    def _extract_advanced_features(self, df):
        """Create engineered features to better distinguish moods"""
        # Rhythm complexity (helps distinguish energetic vs happy)
        df['rhythm_complexity'] = df['zcr_std'] * df['energy_std']
        
        # Spectral contrast (brightness variation)
        df['spectral_contrast'] = df['brightness_std'] / (df['brightness_mean'] + 1e-8)
        
        # Harmonic dominance
        df['harmonic_dominance'] = df['harmonic_ratio'] * df['energy_mean']
        
        # Tempo-energy interaction
        df['tempo_energy_ratio'] = df['tempo'] * df['energy_mean']
        
        # MFCC variance patterns (captures timbral complexity)
        mfcc_std_cols = [f'mfcc_{i}_std' for i in range(13)]
        df['mfcc_variance'] = df[mfcc_std_cols].std(axis=1)
        
        return df
    
    def train(self, df):
        """Train the model """
        # Apply feature engineering
        df = self._extract_advanced_features(df)
        
        X = df.drop('mood', axis=1)
        y = df['mood']
        self.feature_names = X.columns.tolist()
        
        print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Classes: {y.value_counts().to_dict()}")
        
        # Feature selection - be more aggressive to reduce overfitting
        k = min(20, len(self.feature_names))  # Reduced from 25 to 20
        self.selector = SelectKBest(f_classif, k=k)
        X_selected = self.selector.fit_transform(X, y)
        
        # Get selected feature names for debugging
        selected_mask = self.selector.get_support()
        selected_features = [self.feature_names[i] for i in range(len(selected_mask)) 
                           if selected_mask[i]]
        print(f"Selected {len(selected_features)} best features")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.25, random_state=42, stratify=y  # Increased test size
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Hyperparameter tuning with regularization
        param_grid = {
            'n_estimators': [100, 150],  # Reduced options
            'max_depth': [10, 15, None],  # Limit depth
            'min_samples_split': [5, 10],  # Increased from 2
            'min_samples_leaf': [2, 4],    # Added leaf constraint
            'max_features': ['sqrt', 0.5]  # Limit features per tree
        }
        
        print("Performing hyperparameter tuning...")
        self.model = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Use GridSearchCV for better parameters
        grid_search = GridSearchCV(
            self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train_scaled, y_train)
        
        self.model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        print(f"\nPerformance Metrics:")
        print(f"Training Accuracy: {train_acc:.3f}")
        print(f"Testing Accuracy: {test_acc:.3f}")
        print(f"Cross-Validation: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        # Generate visualizations
        self._plot_confusion_matrix(y_test, y_test_pred)
        self._plot_feature_importance(selected_features)
        self._plot_performance_metrics(train_acc, test_acc, cv_scores)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_test_pred))
        
        
        return test_acc
    
    def _plot_feature_importance(self, selected_features):
        """Plot top feature importances"""
        if not self.model:
            return
        
        importance_df = pd.DataFrame({
            'feature': selected_features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Importance')
        plt.title('Top 15 Feature Importances')
        plt.tight_layout()
        plt.savefig('outputs/feature_importance.png', dpi=300)
        plt.close()
        print("Saved: feature_importance.png")
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot enhanced confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['calm', 'energetic', 'happy'],
                    yticklabels=['calm', 'energetic', 'happy'])
        plt.title('Confusion Matrix - Improved Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('outputs/confusion_matrix.png', dpi=300)
        plt.close()
        print("Saved: confusion_matrix.png")
    
    def _plot_performance_metrics(self, train_acc, test_acc, cv_scores):
        """Plot performance comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy comparison
        metrics = ['Training', 'Testing', 'CV Mean']
        values = [train_acc, test_acc, cv_scores.mean()]
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        ax1.bar(metrics, values, color=colors, alpha=0.7)
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy Comparison - Regularized')
        for i, v in enumerate(values):
            ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        # Cross-validation scores
        ax2.plot(range(1, 6), cv_scores, marker='o', linewidth=2)
        ax2.axhline(y=cv_scores.mean(), color='r', linestyle='--', 
                   label=f'Mean: {cv_scores.mean():.3f}')
        ax2.fill_between(range(1, 6), 
                        cv_scores.mean() - cv_scores.std(),
                        cv_scores.mean() + cv_scores.std(),
                        alpha=0.2)
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Cross-Validation Scores')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/performance_metrics.png', dpi=300)
        plt.close()
        print("Saved: performance_metrics.png")
    
    def predict(self, features):
        """Hybrid prediction: enhanced rules + ML"""
        # Try enhanced rules first
        rule_pred, conf = self._rule_predict(features)
        if rule_pred and conf > 0.7:  # Slightly lowered threshold
            return rule_pred, conf
        
        # Fall back to ML
        if self.model and self.selector:
            try:
                # Add engineered features for prediction
                features = self._add_engineered_features(features)
                
                feature_vector = [features.get(f, 0) for f in self.feature_names]
                feature_array = np.array(feature_vector).reshape(1, -1)
                feature_selected = self.selector.transform(feature_array)
                feature_scaled = self.scaler.transform(feature_selected)
                
                pred = self.model.predict(feature_scaled)[0]
                proba = self.model.predict_proba(feature_scaled)[0]
                conf = np.max(proba)
                return pred, conf
            except Exception as e:
                print(f"ML prediction failed: {e}")
        
        return 'happy', 0.5
    
    def _add_engineered_features(self, features):
        """Add engineered features for prediction"""
        features = features.copy()
        
        # Add the same engineered features used in training
        features['rhythm_complexity'] = features.get('zcr_std', 0) * features.get('energy_std', 0)
        features['spectral_contrast'] = features.get('brightness_std', 0) / (features.get('brightness_mean', 1) + 1e-8)
        features['harmonic_dominance'] = features.get('harmonic_ratio', 0) * features.get('energy_mean', 0)
        features['tempo_energy_ratio'] = features.get('tempo', 0) * features.get('energy_mean', 0)
        
        # Calculate MFCC variance
        mfcc_stds = [features.get(f'mfcc_{i}_std', 0) for i in range(13)]
        features['mfcc_variance'] = np.std(mfcc_stds) if mfcc_stds else 0
        
        return features
    
    def save(self, path='models/mood_classifier.pkl'):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'selector': self.selector,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, path)
        print(f"✓ Model saved to {path}")
    
    @staticmethod
    def load(path='models/mood_classifier.pkl'):
        """Load a trained model"""
        model_data = joblib.load(path)
        classifier = HybridMoodClassifier()
        classifier.model = model_data['model']
        classifier.scaler = model_data['scaler']
        classifier.selector = model_data['selector']
        classifier.feature_names = model_data['feature_names']
        return classifier