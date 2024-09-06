import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = load_model("D:\DEPI Generative AI Professional\Week 10\Entity Recognition Task\Recognition.keras")

# Load the tokenizer used for training
with open('D:\DEPI Generative AI Professional\Week 10\Entity Recognition Task\\tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open("D:\DEPI Generative AI Professional\Week 10\Entity Recognition Task\\tag_tokenizer.pkl", 'rb') as f:
    tag_tokenizer = pickle.load(f)
max_length = 89
# Load any necessary preprocessing functions if saved
def preprocess_sentence(sentence):
    # Tokenize the input sentence
    sequence = tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    return padded_sequence

def predict_entities(sentence):
    # Preprocess the sentence
    padded_sequence = preprocess_sentence(sentence)
    
    # Predict using the loaded model
    predictions = model.predict(padded_sequence)
    predictions = np.argmax(predictions, axis=-1)
    
    # Convert prediction indices back to tags
    tags = tag_tokenizer.sequences_to_texts(predictions)
    return tags[0]

# Streamlit app
st.title('Named Entity Recognition (NER) with Streamlit')

sentence = st.text_input("Enter a sentence:")

if st.button('Predict'):
    if sentence:
        tags = predict_entities(sentence)
        st.write("Entities:")
        for word, tag in zip(sentence.split(), tags.split()):
            st.write(f"{word}: {tag}")
    else:
        st.write("Please enter a sentence.")