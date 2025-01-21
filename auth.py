import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Function to extract MFCC features from audio file
def extract_features(audio_file):
    y, sr = librosa.load(audio_file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # Using 20 MFCC coefficients
    return mfcc.T  # Transpose to have features as rows and time frames as columns


# Load and extract features from dataset audio files
your_voice_features1 = extract_features(r"C:/Users/MD ASHAM IMAD/Downloads/final zip/dataset/neha.wav")
your_voice_features2 = extract_features(r"C:/Users/MD ASHAM IMAD/Downloads/final zip/dataset/neha1.wav")
your_voice_features3 = extract_features(r"C:/Users/MD ASHAM IMAD/Downloads/final zip/dataset/neha2.wav")
your_voice_features4 = extract_features(r"C:/Users/MD ASHAM IMAD/Downloads/final zip/dataset/neha3.wav")

other_voice_features1 = extract_features(r"C:/Users/MD ASHAM IMAD/Downloads/final zip/dataset/asham.wav")
other_voice_features2 = extract_features(r"C:/Users/MD ASHAM IMAD/Downloads/final zip/dataset/girl1.wav")
other_voice_features3 = extract_features(r"C:/Users/MD ASHAM IMAD/Downloads/final zip/dataset/girl2.wav")
other_voice_features4 = extract_features(r"C:/Users/MD ASHAM IMAD/Downloads/final zip/dataset/girl3.wav")
other_voice_features5 = extract_features(r"C:/Users/MD ASHAM IMAD/Downloads/final zip/dataset/girl4.wav")
other_voice_features6 = extract_features(r"C:/Users/MD ASHAM IMAD/Downloads/final zip/dataset/girl5.wav")

# Create labels
your_voice_labels1 = np.zeros(your_voice_features1.shape[0])  # Label 0 for "your voice"
your_voice_labels2 = np.zeros(your_voice_features2.shape[0])  # Label 0 for "your voice"
your_voice_labels3 = np.zeros(your_voice_features3.shape[0])  # Label 0 for "your voice"
your_voice_labels4 = np.zeros(your_voice_features4.shape[0])  # Label 0 for "your voice"

other_voice_labels1 = np.ones(other_voice_features1.shape[0])  # Label 1 for "other voice"
other_voice_labels2 = np.ones(other_voice_features2.shape[0])  # Label 1 for "other voice"
other_voice_labels3 = np.ones(other_voice_features3.shape[0])  # Label 1 for "other voice"
other_voice_labels4 = np.ones(other_voice_features4.shape[0])  # Label 1 for "other voice"
other_voice_labels5 = np.ones(other_voice_features5.shape[0])  # Label 1 for "other voice"
other_voice_labels6 = np.ones(other_voice_features6.shape[0])  # Label 1 for "other voice"

# Concatenate features and labels
features = np.vstack([your_voice_features1, your_voice_features2, your_voice_features3, your_voice_features4,
                      other_voice_features1, other_voice_features2, other_voice_features3, other_voice_features4,
                      other_voice_features5, other_voice_features6])

labels = np.hstack([your_voice_labels1, your_voice_labels2, your_voice_labels3, your_voice_labels4, other_voice_labels1,
                    other_voice_labels2, other_voice_labels3, other_voice_labels4, other_voice_labels5,
                    other_voice_labels6])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the classifier on the test set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on test set: {accuracy * 100:.2f}%')


# Process recorded audio chunks for testing
def process_chunks(directory):
    chunk_files = sorted([f for f in os.listdir(directory) if f.startswith("chunk") and f.endswith(".wav")])
    if not chunk_files:
        print("No audio chunks found for testing.")
        return

    for chunk in chunk_files:
        chunk_path = os.path.join(directory, chunk)
        print(f"Processing {chunk_path}...")

        try:
            chunk_features = extract_features(chunk_path)
            predictions = clf.predict(chunk_features)
            prediction_labels = ['MYVOICE' if label == 0 else 'Other' for label in predictions]
            final_prediction = max(set(prediction_labels), key=prediction_labels.count)
            print(f"Prediction for {chunk}: {final_prediction}")
        except Exception as e:
            print(f"Error processing {chunk}: {e}")


# Main function to test recorded chunks
if __name__ == "__main__":
    recorded_chunks_directory = r"C:/Users/MD ASHAM IMAD/Downloads/final zip"  # Replace with the directory used by rec.py
    process_chunks(recorded_chunks_directory)
