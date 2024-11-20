import streamlit as st
import os

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
        predictions = self.loaded_model.predict(x)
        # prediction = np.argmax(predictions,axis=1)
        # print("Prediction is", " ", self.convertclasstoemotion(predictions))
        return self.convertclasstoemotion(predictions)

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
        # emotions = list(label_conversions.values())

        # for key, value in label_conversion.items():
        #     if int(key) == pred:
        #         label = value
        # predicted_classes = {label_conversion[str(i)]: str(round(score*100,2))+'%' for i, score in enumerate(pred[0])}
        predicted_classes = {label_conversion[str(i)]: score for i, score in enumerate(pred[0])}
        #{label_conversion[str(i)]: pred[i] for i in range(len(pred))}
        # for i in pred[0]:
            

        return predicted_classes
# pred = livePredictions(path='testing10_model.h5',file='Ravtess/03-01-01-01-01-01-06.wav')

# pred.load_model()
# pred.makepredictions()


# Custom Audio Processor to handle the audio recording and prediction
class AudioProcessor():
    def __init__(self, model_path):
        self.model_path = model_path
        self.prediction_model = None

    def recv(self, frame):
        """
        This function is called when a new frame (audio buffer) is received.
        Here, we will directly pass the raw audio to the prediction function.
        """
        # Convert the audio frame to numpy array (raw audio data)
        audio_data = frame.to_ndarray()

        # We don't modify the audio data; we pass it directly to the prediction function
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            # Write raw audio data as bytes to temporary file
            temp_audio.write(audio_data.tobytes())
            audio_file_path = temp_audio.name

        # Use the livePredictions class to process the audio file and make predictions
        # pred = livePredictions(path=self.model_path, file=audio_file_path)
        pred = livePredictions(path=self.model_path, file=audio_file_path)
        emotion = pred.makepredictions()

        # Return predicted emotions
        return emotion




# Streamlit UI
st.title("Emotion Detection from Audio")

# Audio file upload


st.subheader('Record audio now...')
from audio_recorder_streamlit import audio_recorder
from pydub import AudioSegment

audio_bytes = audio_recorder()

st.subheader('or, upload an audio file...')
audio_file = st.file_uploader("", type=["wav", "mp3"])
if audio_file:
    # Save the uploaded audio file temporarily
    with open("audio_file.wav", "wb") as f:
        f.write(audio_file.read())
    st.audio('audio_file.wav')
    model_path = 'testing10_model.h5'
    pred = livePredictions(path=model_path, file='audio_file.wav')
    model_summary = pred.load_model()
    emotion = pred.makepredictions()

    # Display predicted emotions
    st.subheader("Predicted Emotions:")
    dct = dict()
    for emotion_label, score in emotion.items():
        # st.write(f"{emotion_label}: {score}")
        dct[emotion_label] = score
    sorted_items = sorted(dct.items(), key=lambda item: item[1], reverse=True)
    # st.write(sorted_items)
    # st.write(
    #     sorted_items[0][0]
    # )
    hide_table_headers = """
        <style>
        thead tr th {display:none}
        tbody tr th {display:none}
        table tbody tr:first-child {
        font-size: 3em; /* Larger font size */
        font-weight: bold; /* Bold text */
    }
        </style>
        """
    sorted_item = [(sorted_items[i][0],str(round(sorted_items[i][1] * 100,2)) + '%') for i in range(len(sorted_items))]
    st.markdown(hide_table_headers, unsafe_allow_html=True)
    # st.table(sorted_items)
    st.table(sorted_item)
if audio_bytes:
    
    audio_segment = AudioSegment(
        data=audio_bytes,
        # sample_width=2, # Sample width in bytes
        # frame_rate=44100, # Frame rate
        # channels=1 # Mono
        )
    x = audio_segment.export('output.wav', format='wav')
    st.audio(x.name, format="audio/wav")
    # with open(x.name, "wb") as f:
    #     f.write(audio_file.read())
    # Model path (you may need to adjust this)
    model_path = 'testing10_model.h5'

    # Initialize prediction class
    pred = livePredictions(path=model_path, file=x.name)

    # Load the model
    model_summary = pred.load_model()

    # Predict emotion
    emotion = pred.makepredictions()

    # Display predicted emotions
    st.subheader("Predicted Emotions:")
    dct = dict()
    for emotion_label, score in emotion.items():
        # st.write(f"{emotion_label}: {score}")
        dct[emotion_label] = score
    sorted_items = sorted(dct.items(), key=lambda item: item[1], reverse=True)
    print(sorted_items)
    # st.write(sorted_items)
    # st.write(
    #     sorted_items[0][0]
    # )
    hide_table_headers = """
        <style>
        thead tr th {display:none}
        tbody tr th {display:none}
        table tbody tr:first-child {
        font-size: 3em; /* Larger font size */
        font-weight: bold; /* Bold text */
    }
        </style>
       """
    sorted_item = [(sorted_items[i][0],str(round(sorted_items[i][1] * 100,2)) + '%') for i in range(len(sorted_items))]
    st.markdown(hide_table_headers, unsafe_allow_html=True)
    # st.table(sorted_items)
    st.table(sorted_item)
    # st.markdown(hide_table_headers, unsafe_allow_html=True)
    # st.table(sorted_items)

    # st.subheader(f'{list(sorted_dict.keys())[0]}({list(sorted_dict.values())[0]})')
    # for i in range(1,len(sorted_dict.keys())):
    #     st.write(f'{list(sorted_dict.keys())[i]}                      {list(sorted_dict.values())[i]}')