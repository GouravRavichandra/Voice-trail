import streamlit as st
from live_predictions import LivePredictions

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

# def main():
#     st.title("Speech Emotion Recognition App")

#     st.sidebar.header("Settings")
#     model_path = st.sidebar.text_input("Enter Model Path", "SER_model.h5")
    
#     uploaded_file = st.sidebar.file_uploader("Choose an audio file", type=["wav"])

#     if st.sidebar.button("Make Prediction") and uploaded_file is not None:
#         st.audio(uploaded_file, format="audio/wav")
#         prediction = make_prediction(model_path, uploaded_file)
#         display_emotion_label(prediction)

def make_prediction(model_path, audio_file):
    pred = LivePredictions(model_path, audio_file)
    pred.load_model()
    return pred.make_predictions()

def display_emotion_label(prediction):
    st.write(f"Predicted Emotion: {prediction}")

if __name__ == "__main__":
    main()
