import streamlit as st
import numpy as np
import re
import string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json
import tensorflow as tf


# Load model
try:
    model = load_model('resume_model.keras')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")


with open('tokenizer.json', 'r') as json_file:
    tokenizer_json = json_file.read()
tokenizer = tokenizer_from_json(tokenizer_json)

# Define preprocessing function for resumes
def preprocess_resume(resume_text, tokenizer, max_sequence_length):
    # Handle potential float values
    if isinstance(resume_text, float):
        resume_text = str(resume_text)  # Convert float to string if necessary

    # Convert text to lowercase
    resume_text = resume_text.lower()

    # Remove specific unwanted characters
    resume_text = resume_text.replace('â¢', '')
    resume_text = resume_text.replace('â', '')  # Remove additional unwanted characters

    # Remove URLs
    resume_text = re.sub(r'^https?:\/\/.*[\r\n]*', '', resume_text, flags=re.MULTILINE)

    # Remove HTML tags
    resume_text = re.sub(r'<.*?>', '', resume_text)

    # Remove emoticons and other symbols
    regex_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+", flags=re.UNICODE)
    resume_text = regex_pattern.sub(r'', resume_text)

    # Remove digits
    resume_text = ''.join([i for i in resume_text if not i.isdigit()])

    # Remove punctuation
    resume_text = ''.join([char for char in resume_text if char not in string.punctuation])

    # Tokenize text
    words = word_tokenize(resume_text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word, pos='v') for word in words]

    # Join words back into a single string
    cleaned_text = ' '.join(words)

    # Convert cleaned text to sequences and pad
    sequences = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)  # Adjust maxlen if needed

    return padded_sequences

# Define label mapping
label_mapping = {
    'Data Science': 0,
    'HR': 1,
    'Advocate': 2,
    'Arts': 3,
    'Web Designing': 4,
    'Mechanical Engineer': 5,
    'Sales': 6,
    'Health and fitness': 7,
    'Civil Engineer': 8,
    'Java Developer': 9,
    'Business Analyst': 10,
    'SAP Developer': 11,
    'Automation Testing': 12,
    'Electrical Engineering': 13,
    'Operations Manager': 14,
    'Python Developer': 15,
    'DevOps Engineer': 16,
    'Network Security Engineer': 17,
    'PMO': 18,
    'Database': 19,
    'Hadoop': 20,
    'ETL Developer': 21,
    'DotNet Developer': 22,
    'Blockchain': 23,
    'Testing': 24
}

# Reverse label mapping for decoding predictions
inverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Define the Streamlit app
st.title('Resume Classification Application')

# User input for resume text
resume_text = st.text_area("Paste the resume text here:")

if st.button("Classify Resume"):
    if resume_text:
        # Preprocess the resume
        padded_resume = preprocess_resume(resume_text, tokenizer, max_sequence_length=100)  # Adjust max_sequence_length if needed

        # Predict the class
        prediction = model.predict(padded_resume)
        class_index = np.argmax(prediction, axis=1)[0]

        # Get the predicted class label
        predicted_class = inverse_label_mapping[class_index]

        # Display the result
        st.write(f'Prediction: {predicted_class}')
    else:
        st.write("Please paste a resume text to classify.")
