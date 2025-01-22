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
your_voice_features1 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/neha.wav")
your_voice_features2 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/neha1.wav")
your_voice_features3 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/neha2.wav")
your_voice_features4 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/neha3.wav")
your_voice_features5 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/neha5.wav")
your_voice_features6 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/neha6.wav")
your_voice_features7 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/neha7.wav")
your_voice_features8 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/neha8.wav")
your_voice_features9 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/neha9.wav")
your_voice_features10 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/neha10.wav")
your_voice_features11 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/neha11.wav")
your_voice_features12 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/neha12.wav")
your_voice_features13 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/neha13.wav")
your_voice_features14 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/neha14.wav")
your_voice_features15 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/neha15.wav")
your_voice_features16 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/neha16.wav")
your_voice_features17 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/neha17.wav")
your_voice_features18 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/neha18.wav")
your_voice_features19 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/neha19.wav")
your_voice_features20 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/neha20.wav")
your_voice_features21 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/neha21.wav")
your_voice_features22 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/neha22.wav")
your_voice_features23 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/neha23.wav")
your_voice_features24 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/neha24.wav")
your_voice_features25 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/neha25.wav")
your_voice_features26 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/neha26.wav")
your_voice_features27 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/neha27.wav")
your_voice_features28 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/neha28.wav")


other_voice_features1 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/asham.wav")
other_voice_features2 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/girl1.wav")
other_voice_features3 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/girl2.wav")
other_voice_features4 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/girl3.wav")
other_voice_features5 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/girl4.wav")
other_voice_features6 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/girl5.wav")
other_voice_features7 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/other2.wav")
other_voice_features8 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/other3.wav")
other_voice_features9 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/other4.wav")
other_voice_features10 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/other5.wav")
other_voice_features11 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/other6.wav")
other_voice_features12 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/other7.wav")
other_voice_features13 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/other8.wav")
other_voice_features14 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/other9.wav")
other_voice_features15 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/other10.wav")
other_voice_features16 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/other11.wav")
other_voice_features17 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/other12.wav")
other_voice_features18 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/other13.wav")
other_voice_features19 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/other14.wav")
# other_voice_features12 = extract_features(r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip/dataset/other15.wav")

# Create labels
your_voice_labels1 = np.zeros(your_voice_features1.shape[0])  # Label 0 for "your voice"
your_voice_labels2 = np.zeros(your_voice_features2.shape[0])  # Label 0 for "your voice"
your_voice_labels3 = np.zeros(your_voice_features3.shape[0])  # Label 0 for "your voice"
your_voice_labels4 = np.zeros(your_voice_features4.shape[0])  # Label 0 for "your voice"
your_voice_labels5 = np.zeros(your_voice_features5.shape[0])  # Label 0 for "your voice"
your_voice_labels6 = np.zeros(your_voice_features6.shape[0])  # Label 0 for "your voice"
your_voice_labels7 = np.zeros(your_voice_features7.shape[0])  # Label 0 for "your voice"
your_voice_labels8 = np.zeros(your_voice_features8.shape[0])  # Label 0 for "your voice"
your_voice_labels9 = np.zeros(your_voice_features9.shape[0])  # Label 0 for "your voice"
your_voice_labels10 = np.zeros(your_voice_features10.shape[0])  # Label 0 for "your voice"
your_voice_labels11 = np.zeros(your_voice_features11.shape[0])  # Label 0 for "your voice"
your_voice_labels12 = np.zeros(your_voice_features12.shape[0])  # Label 0 for "your voice"
your_voice_labels13 = np.zeros(your_voice_features13.shape[0])  # Label 0 for "your voice"
your_voice_labels14 = np.zeros(your_voice_features14.shape[0])  # Label 0 for "your voice"
your_voice_labels15 = np.zeros(your_voice_features15.shape[0])  # Label 0 for "your voice"
your_voice_labels16 = np.zeros(your_voice_features16.shape[0])  # Label 0 for "your voice"
your_voice_labels17 = np.zeros(your_voice_features17.shape[0])  # Label 0 for "your voice"
your_voice_labels18 = np.zeros(your_voice_features18.shape[0])  # Label 0 for "your voice"
your_voice_labels19 = np.zeros(your_voice_features19.shape[0])  # Label 0 for "your voice"
your_voice_labels20 = np.zeros(your_voice_features20.shape[0])  # Label 0 for "your voice"
your_voice_labels21 = np.zeros(your_voice_features21.shape[0])  # Label 0 for "your voice"
your_voice_labels22 = np.zeros(your_voice_features22.shape[0])  # Label 0 for "your voice"
your_voice_labels23 = np.zeros(your_voice_features23.shape[0])  # Label 0 for "your voice"
your_voice_labels24 = np.zeros(your_voice_features24.shape[0])  # Label 0 for "your voice"
your_voice_labels25 = np.zeros(your_voice_features25.shape[0])  # Label 0 for "your voice"
your_voice_labels26 = np.zeros(your_voice_features26.shape[0])  # Label 0 for "your voice"
your_voice_labels27 = np.zeros(your_voice_features27.shape[0])  # Label 0 for "your voice"
your_voice_labels28 = np.zeros(your_voice_features28.shape[0])  # Label 0 for "your voice"

other_voice_labels1 = np.ones(other_voice_features1.shape[0])  # Label 1 for "other voice"
other_voice_labels2 = np.ones(other_voice_features2.shape[0])  # Label 1 for "other voice"
other_voice_labels3 = np.ones(other_voice_features3.shape[0])  # Label 1 for "other voice"
other_voice_labels4 = np.ones(other_voice_features4.shape[0])  # Label 1 for "other voice"
other_voice_labels5 = np.ones(other_voice_features5.shape[0])  # Label 1 for "other voice"
other_voice_labels6 = np.ones(other_voice_features6.shape[0])  # Label 1 for "other voice"
other_voice_labels7 = np.ones(other_voice_features7.shape[0])
other_voice_labels8 = np.ones(other_voice_features8.shape[0])
other_voice_labels9 = np.ones(other_voice_features9.shape[0])
other_voice_labels10 = np.ones(other_voice_features10.shape[0])
other_voice_labels10 = np.ones(other_voice_features10.shape[0])
other_voice_labels11 = np.ones(other_voice_features11.shape[0])
other_voice_labels12 = np.ones(other_voice_features12.shape[0])
other_voice_labels13 = np.ones(other_voice_features13.shape[0])
other_voice_labels14 = np.ones(other_voice_features14.shape[0])
other_voice_labels15 = np.ones(other_voice_features15.shape[0])
other_voice_labels16 = np.ones(other_voice_features16.shape[0])
other_voice_labels17 = np.ones(other_voice_features17.shape[0])
other_voice_labels18 = np.ones(other_voice_features18.shape[0])
other_voice_labels19 = np.ones(other_voice_features19.shape[0])

# Concatenate features and labels
features = np.vstack([your_voice_features1, your_voice_features2, your_voice_features3, your_voice_features4, your_voice_features5,
                      your_voice_features6, your_voice_features7, your_voice_features8, your_voice_features9, your_voice_features10,
                      your_voice_features11, your_voice_features12, your_voice_features13, your_voice_features14, your_voice_features15,
                      your_voice_features16, your_voice_features17, your_voice_features18, your_voice_features19, your_voice_features20,
                      your_voice_features21, your_voice_features22, your_voice_features23, your_voice_features24, your_voice_features25,
                      your_voice_features26, your_voice_features27, your_voice_features28,
                      other_voice_features1, other_voice_features2, other_voice_features3, other_voice_features4,
                      other_voice_features5, other_voice_features6, other_voice_features7, other_voice_features8, other_voice_features9, 
                      other_voice_features10, other_voice_features11, other_voice_features12, other_voice_features13, other_voice_features14,
                      other_voice_features15, other_voice_features16, other_voice_features17, other_voice_features18, other_voice_features19])

labels = np.hstack([your_voice_labels1, your_voice_labels2, your_voice_labels3, your_voice_labels4, your_voice_labels5, 
                    your_voice_labels6, your_voice_labels7, your_voice_labels8, your_voice_labels9, your_voice_labels10,
                    your_voice_labels11, your_voice_labels12, your_voice_labels13, your_voice_labels14, your_voice_labels15,
                    your_voice_labels16, your_voice_labels17, your_voice_labels18, your_voice_labels19, your_voice_labels20,
                    your_voice_labels21, your_voice_labels22, your_voice_labels23, your_voice_labels24, your_voice_labels25,
                    your_voice_labels26, your_voice_labels27, your_voice_labels28, other_voice_labels1,
                    other_voice_labels2, other_voice_labels3, other_voice_labels4, other_voice_labels5,
                    other_voice_labels6, other_voice_labels7, other_voice_labels8, other_voice_labels9, other_voice_labels10,
                    other_voice_labels11, other_voice_labels12, other_voice_labels13, other_voice_labels14, other_voice_labels15,
                    other_voice_labels16, other_voice_labels17, other_voice_labels18, other_voice_labels19])

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
    recorded_chunks_directory = r"C:/Users/MD ASHAM IMAD/OneDrive/Desktop/final zip"  # Replace with the directory used by rec.py
    process_chunks(recorded_chunks_directory)
