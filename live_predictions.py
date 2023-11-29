import keras
import numpy as np
import librosa

class LivePredictions:
    """
    Main class for live speech emotion predictions.
    """

    def __init__(self, path, file):
        """
        Initialize the class with the path to the model and the file to predict.
        """
        self.path = path
        self.file = file
        self.loaded_model = None

    def load_model(self):
        """
        Load the pre-trained model.
        """
        self.loaded_model = keras.models.load_model(self.path)

    def extract_features(self):
        """
        Extract features from the provided audio file.
        """
        data, sampling_rate = librosa.load(self.file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, axis=1)
        x = np.expand_dims(x, axis=0)
        return x

    def make_predictions(self):
        """
        Make predictions using the loaded model and extracted features.
        """
        if self.loaded_model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        features = self.extract_features()
        predictions = self.loaded_model.model.predict(features)
        emotion_label = self.convert_class_to_emotion(predictions[0])
        return emotion_label

    @staticmethod
    def convert_class_to_emotion(pred):
        """
        Convert the predicted class to a human-readable emotion label.
        """
        label_conversion = {
            0: 'neutral',
            1: 'calm',
            2: 'happy',
            3: 'sad',
            4: 'angry',
            5: 'fearful',
            6: 'disgust',
            7: 'surprised'
        }

        return label_conversion.get(pred, 'unknown')
