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

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the file
    filename = secure_filename(audio_file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(file_path)

    # Preprocess and predict
    audio_features = preprocess_audio(file_path)
    predictions = model.predict(audio_features)[0]
    emotion_percentages = {emotion: round(pred * 100, 2) for emotion, pred in zip(emotion_labels, predictions)}

    # Generate spectrogram
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    plt.figure(figsize=(10, 4))
    plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap='inferno', sides='default', mode='default', scale='dB')
    plt.axis('off')
    spectrogram_path = os.path.join(app.config['UPLOAD_FOLDER'], 'spectrogram.png')
    plt.savefig(spectrogram_path)
    plt.close()

    return jsonify({'emotions': emotion_percentages, 'spectrogram': spectrogram_path})

if __name__ == '__main__':
    app.run(debug=True)
