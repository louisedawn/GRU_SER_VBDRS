import h5py

model_path = 'gru_emotion_model.h5'  # Replace with your model file path

try:
    with h5py.File(model_path, 'r') as h5_file:
        # Check for Keras version, 3.3.3 is needed
        if 'keras_version' in h5_file.attrs:
            keras_version = h5_file.attrs['keras_version']
            print(f"Keras version: {keras_version}")

        # Check for backend (TensorFlow)
        if 'backend' in h5_file.attrs:
            backend = h5_file.attrs['backend']
            print(f"Backend: {backend}")

        # Check model configuration
        if 'model_config' in h5_file:
            model_config = h5_file['model_config'][:]
            print("Model configuration found.")
        else:
            print("No model configuration found.")

    with h5py.File('gru_emotion_model.h5', 'r') as f:
        if 'model_weights' in f.keys():
            print("The file contains weights only.")
        else:
            print("The file contains a full model.")

except Exception as e:
    print(f"Error while analyzing the H5 file: {e}")
