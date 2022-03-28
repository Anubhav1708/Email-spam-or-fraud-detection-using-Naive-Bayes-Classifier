import streamlit as st
import pickle
import string
import sklearn
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def modify_email_text(email_text):
    email_text = email_text.lower()
    email_text = nltk.word_tokenize(email_text)

    z = []
    for i in email_text:
        if i.isalnum():
            z.append(i)

    email_text = z[:]
    z.clear()

    for i in email_text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            z.append(i)

    email_text = z[:]
    z.clear()

    for i in email_text:
        z.append(ps.stem(i))

    return " ".join(z)

Tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email Spam Classifier")

input_mail = st.text_input("Enter the mail")

# 1. preprocess
modified_text = modify_email_text(input_mail)
# 2. Vectorize
vector_input = Tfidf.transform([modified_text])
# 3. predict
result = model.predict(vector_input[0])
# 4. display
if result ==1:
    st.header("Spam")
else:
    st.header("Not Spam")

