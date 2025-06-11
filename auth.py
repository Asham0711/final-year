import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class VoiceAuthenticator:
    def __init__(self):
        # Load dataset features and labels
        self.features, self.labels = self.load_dataset()
        # Train model
        self.clf = self.train_model(self.features, self.labels)

    def extract_features(self, audio_file):
        y, sr = librosa.load(audio_file, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        return mfcc.T

    def load_dataset(self):
        # List all your voice files and other voice files
        your_voice_files = [
            f"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/neha{i}.wav" for i in range(29)
        ]
        your_voice_files[0] = "C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/neha.wav"  # first file doesn't have number

        other_voice_files = [
            "C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/asham.wav",
            "C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/girl1.wav",
            "C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/girl2.wav",
            "C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/girl3.wav",
            "C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/girl4.wav",
            "C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/girl5.wav",
            "C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/other2.wav",
            "C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/other3.wav",
            "C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/other4.wav",
            "C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/other5.wav",
            "C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/other6.wav",
            "C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/other7.wav",
            "C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/other8.wav",
            "C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/other9.wav",
            "C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/other10.wav",
            "C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/other11.wav",
            "C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/other12.wav",
            "C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/other13.wav",
            "C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/other14.wav"
        ]

        # Extract features and labels for your voice
        your_features = []
        your_labels = []
        for file in your_voice_files:
            features = self.extract_features(file)
            your_features.append(features)
            your_labels.append(np.zeros(features.shape[0]))  # label 0

        # Extract features and labels for others
        other_features = []
        other_labels = []
        for file in other_voice_files:
            features = self.extract_features(file)
            other_features.append(features)
            other_labels.append(np.ones(features.shape[0]))  # label 1

        # Stack all features and labels
        features = np.vstack(your_features + other_features)
        labels = np.hstack(your_labels + other_labels)
        return features, labels

    def train_model(self, features, labels):
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained with accuracy: {accuracy * 100:.2f}%")
        return clf

    def predict_voice(self, audio_file):
        features = self.extract_features(audio_file)
        predictions = self.clf.predict(features)
        labels = ['MYVOICE' if pred == 0 else 'Other' for pred in predictions]
        # Take the majority vote
        final_label = max(set(labels), key=labels.count)
        return final_label

    def predict_chunks(self, directory):
        chunk_files = sorted([f for f in os.listdir(directory) if f.startswith("chunk") and f.endswith(".wav")])
        if not chunk_files:
            return "No audio chunks found for testing."

        results = {}
        for chunk in chunk_files:
            try:
                chunk_path = os.path.join(directory, chunk)
                prediction = self.predict_voice(chunk_path)
                results[chunk] = prediction
            except Exception as e:
                results[chunk] = f"Error: {e}"
        return results

# Usage example if run directly
if __name__ == "__main__":
    va = VoiceAuthenticator()
    recorded_chunks_directory = r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/chunks"
    results = va.predict_chunks(recorded_chunks_directory)
    for chunk, prediction in results.items():
        print(f"{chunk}: {prediction}")
