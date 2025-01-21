import os
import speech_recognition as sr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Step 1: Prepare training data
training_phrases = [
    "help", "please help", "can you help me", "need assistance",
    "hello", "good morning", "no emergency", "just checking",
]
labels = [1, 1, 1, 1, 0, 0, 0, 0]

# Step 2: Vectorize text data
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(training_phrases)

# Step 3: Train a classifier
classifier = LogisticRegression()
classifier.fit(X_train, labels)


# Step 4: Function to recognize speech from an audio file and classify it
def recognize_from_audio_file(file_path):
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(file_path) as source:
            print(f"Processing {file_path}...")
            audio = recognizer.record(source)  # Record the entire audio file

            # Recognize text from audio using Google's Speech Recognition
            text = recognizer.recognize_google(audio).lower()
            print(f"Recognized text: {text}")

            # Vectorize the recognized text
            X_test = vectorizer.transform([text])

            # Predict using the trained classifier
            prediction = classifier.predict(X_test)

            if prediction == 1:
                return "Emergency detected!"
            else:
                return "No emergency detected."

    except sr.UnknownValueError:
        return f"Could not understand the audio in {file_path}."
    except sr.RequestError:
        return "Could not request results from Google Speech Recognition service."
    except FileNotFoundError:
        return f"Audio file {file_path} not found."


# Step 5: Process multiple audio files starting with "chunk"
def process_audio_chunks(directory):
    audio_files = sorted([f for f in os.listdir(directory) if f.startswith("chunk") and f.endswith(".wav")])
    if not audio_files:
        print("No audio files starting with 'chunk' found.")
        return

    for file_name in audio_files:
        file_path = os.path.join(directory, file_name)
        result = recognize_from_audio_file(file_path)
        print(f"{file_name}: {result}")


# Step 6: Run the pipeline
if __name__ == "__main__":
    audio_directory = "C:/Users/MD ASHAM IMAD/Downloads/final zip/"  # Replace with your directory path
    process_audio_chunks(audio_directory)
