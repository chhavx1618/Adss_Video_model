import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import load_model

# Download NLTK data if not already present
nltk.download('stopwords')
stopword = set(stopwords.words('english'))
stemmer = nltk.SnowballStemmer('english')

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('@[^\s]+', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

def predict_class(text):
    model = load_model("model2.h5")
    cv = CountVectorizer(vocabulary=np.load('count_vectorizer.npy', allow_pickle=True).item())
    text_vectorized = cv.transform([clean(text)])
    prediction = model.predict(text_vectorized)
    predicted_class = int(np.argmax(prediction))
    return predicted_class

x = predict_class("I will kill you")
print(x)