from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import pandas as pd

app=Flask(__name__)
model = load_model('model.h5')

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()

def preprocess_text(text, tokenizer, max_seq_length):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_seq_length)
    return padded_sequences

def predict_sentiment(processed_text, model):
    prediction = model.predict(processed_text)
    return prediction
text_collection = ['good, it is good', 'hi, how are you', 'no, no way, too bad', 'it is too good'] #demo sentence

def softmax2label(prediction):
  prediction_label = "Positive" if prediction >= 0.5 else "Negative"
  return prediction_label

@app.route('/')
def inputs():
    return render_template('index.html')

@app.route('/classify',methods=['POST'])
def classify():
    for text in text_collection:
        processed_text = preprocess_text(text, tokenizer, max_seq_length)
        sentiment = predict_sentiment(processed_text, model)
        x = softmax2label(sentiment)
    pos = len(x == "Positive")
    neg = len(x == "Negative")
    return render_template('index.html',positive=pos, negative=neg)

if __name__=='__main__':
    app.run()