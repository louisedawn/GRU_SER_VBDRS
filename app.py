from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import librosa
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load GRU model (Assume the model is saved as 'gru_emotion_model.h5')
MODEL_PATH = 'gru_emotion_model.h5'
model = load_model(MODEL_PATH)

# Emotion labels
emotion_labels = ['Fear', 'Angry', 'Disgust', 'Neutral', 'Sad', 'Pleasantly Surprised', 'Happy']

def preprocess_audio(file_path):
    """
    Preprocess audio: Load file, extract MFCCs.
    """
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)
    return np.expand_dims(mfcc, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    audio_file = request.files['audio']
    if not audio_file:
        return jsonify({'error': 'No audio file provided'}), 400

    # Save the audio file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_audio.wav')
    audio_file.save(file_path)

    # Preprocess and predict
    audio_features = preprocess_audio(file_path)
    predictions = model.predict(audio_features)[0]

    # Check if the detected emotion is "Fear"
    fear_index = emotion_labels.index('Fear')
    is_danger = predictions[fear_index] > 0.5  # Threshold of 50%

    return jsonify({'danger': is_danger})

if __name__ == '__main__':
    app.run(debug=True)
