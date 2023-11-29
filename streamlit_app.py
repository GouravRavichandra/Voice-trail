import streamlit as st


def main():
    st.title("Speech Emotion Recognition App")

    st.sidebar.header("Settings")
    model_path = st.sidebar.text_input("Enter Model Path", "SER_model.h5")
    
    uploaded_file = st.sidebar.file_uploader("Choose an audio file", type=["wav"])

    if st.sidebar.button("Make Prediction") and uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        try:
            prediction = make_prediction(model_path, uploaded_file)
            display_emotion_label(prediction)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

def make_prediction(model_path, audio_file):
    pred = livePredictions(model_path, audio_file)
    pred.load_model()
    prediction = pred.make_predictions()
    return prediction


def display_emotion_label(prediction):
    st.write(f"Predicted Emotion: {prediction}")
import keras
import numpy as np
import librosa

class livePredictions:
    """
    Main class of the application.
    """

    def __init__(self, path, file):
        """
        Init method is used to initialize the main parameters.
        """
        self.path = path
        self.file = file

    def load_model(self):
        """
        Method to load the chosen model.
        :param path: path to your h5 model.
        :return: summary of the model with the .summary() function.
        """
        self.loaded_model = keras.models.load_model(self.path)
        return self.loaded_model.summary()

    def makepredictions(self):
        """
        Method to process the files and create your features.
        """
        data, sampling_rate = librosa.load(self.file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, axis=1)
        x = np.expand_dims(x, axis=0)
        predict_x=model.predict(x) 
        predictions=np.argmax(predict_x,axis=1)
        print("Prediction is", " ", self.convertclasstoemotion(predictions))

    @staticmethod
    def convertclasstoemotion(pred):
        """
        Method to convert the predictions (int) into human readable strings.
        """
        
        label_conversion = {'0': 'neutral',
                            '1': 'calm',
                            '2': 'happy',
                            '3': 'sad',
                            '4': 'angry',
                            '5': 'fearful',
                            '6': 'disgust',
                            '7': 'surprised'}

        for key, value in label_conversion.items():
            if int(key) == pred:
                label = value
        return label


if __name__ == "__main__":
    main()
