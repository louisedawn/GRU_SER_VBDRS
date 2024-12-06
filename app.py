from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import librosa
import numpy as np
from keras.models import load_model
import time  # Import time module to measure duration

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load GRU models
ORIGINAL_MODEL_PATH = 'gru_model_not_optimized.h5'
ENHANCED_MODEL_PATH = 'gru_original_model.h5'

original_model = load_model(ORIGINAL_MODEL_PATH)
enhanced_model = load_model(ENHANCED_MODEL_PATH)

# Emotion labels
emotion_labels = ['Fear', 'Angry', 'Disgust', 'Neutral', 'Sad', 'Pleasantly Surprised', 'Happy']

def preprocess_audio(file_path, model_type='original'):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    
    # For the original model, we use 10 MFCC features
    # For the enhanced model, we use 1 MFCC feature per time step
    if model_type == 'original':
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=1)  # 10 features per time step for the original model
    else:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=1)  # 1 feature per time step for the enhanced model

    # Ensure 100 frames (padding/truncating as needed)
    mfcc = librosa.util.fix_length(mfcc, size=100, axis=1)  # Fix along time axis

    # Reshape for model input: (batch_size, time_steps, features)
    return np.expand_dims(mfcc.T, axis=0)

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

    # Retrieve model selection
    selected_model = request.form.get('model', 'original')
    print("GRU Model:", selected_model)
    model = original_model if selected_model == 'original' else enhanced_model
    print(model.summary())

    # Preprocess and predict
    audio_features = preprocess_audio(file_path, model_type=selected_model)
    print(f"Audio feature shape: {audio_features.shape}")
    print(f"Model input shape: {model.input_shape}")

    start_time = time.time()  # Start time
    predictions = model.predict(audio_features)[0]
    end_time = time.time()  # End time

    # Calculate time taken for prediction
    prediction_time = end_time - start_time
    avg_time_per_step = prediction_time  # Convert to seconds

    # Debugging: Print raw predictions
    print("Predictions:", predictions)  # Add this line to debug the output probabilities

    # Determine the highest predicted emotion
    predicted_index = np.argmax(predictions)
    predicted_emotion = emotion_labels[predicted_index]

    # Get the probability of the prediction
    predicted_probability = float(predictions[predicted_index])

    # Check if the detected emotion is "Fear"
    is_danger = predicted_emotion == 'Fear'

    return jsonify({
        'danger': bool(is_danger),
        'emotion': predicted_emotion,  # Include the detected emotion in the response
        'probability': predicted_probability,  # Include the probability in the response
        'prediction_time': round(avg_time_per_step, 2)  # Add average time per step to the response
    })


if __name__ == '__main__':
    app.run(debug=True)
