import streamlit as st
import tensorflow as tf
import numpy as np
import pickle 

import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import IPython.display as ipd
import librosa
import librosa.display

loaded_model = tf.saved_model.load('my_lstm_model2')

st.title("RESPIRATORY TRACK DISORDER ")
audio_file = st.file_uploader("Upload file", type=["mp3", "wav"])
if audio_file is not None:
    st.audio(audio_file)
    if st.button("PREDICT"):
        # Load the audio file and convert it to a NumPy array
        audio_data, sr = librosa.load(audio_file, sr=22050, duration=10)
        
        # Trim or pad the audio to the desired length
        target_n_samples = 22050 * 10
        if len(audio_data) < target_n_samples:
            audio_data = np.pad(audio_data, (0, target_n_samples - len(audio_data)), 'constant')
        else:
            audio_data = audio_data[:target_n_samples]
        
        # Extract the MFCC features
        n_mfcc = 20
        mfccs = librosa.feature.mfcc(y=audio_data, sr=22050, n_mfcc=n_mfcc)
        preprocessed_input = tf.cast(mfccs.T[np.newaxis, ...], tf.float32)
        
        # Make the prediction
        predict_fn = loaded_model.signatures["serving_default"]
        prediction = predict_fn(lstm_1_input=preprocessed_input)
        output = prediction['dense_1'].numpy()

        # Convert the output to labels based on the threshold
        threshold = 0.5
        label_encoder = {'Positive': 0, 'Negative': 1}
        predicted_label = 'Positive' if output[0][0] < threshold else 'Negative'
        
        # Display the prediction in the app
        st.header(predicted_label)
        print(predicted_label)
