import streamlit as st
import nltk
import spacy
import string
import re
import pandas as pd
import matplotlib.pyplot as plt

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# DOWNLOAD NLTK DATA
nltk.download("punkt")
nltk.download("stopwords")

# LOAD SPACY MODEL
nlp = spacy.load("en_core_web_sm")

# STREAMLIT PAGE CONFIG
st.set_page_config(
    page_title="NLP Preprocessing App",
    layout="wide"
)

# APP TITLE
st.title("NLP Preprocessing App")
st.write("Tokenization, Cleaning, Stemming, Lemmatization, BoW, TF-IDF, Regex & Embeddings")

# USER INPUT
text = st.text_area(
    "Enter text for NLP processing",
    height=150,
    placeholder="Example: Aman is the HOD of HIT and loves NLP"
)

# SIDEBAR OPTIONS
option = st.sidebar.radio(
    "Select NLP Technique",
    [
        "Tokenization",
        "Text Cleaning",
        "Regular Expression",
        "Stemming",
        "Lemmatization",
        "Bag of Words",
        "TF-IDF",
        "Word Embedding"
    ]
)

# PROCESS BUTTON
if st.button("Process Text"):
    if text.strip() == "":
        st.warning("Please enter some text.")

    # TOKENIZATION
    elif option == "Tokenization":
        st.subheader("Tokenization Output")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Sentence Tokenization")
            st.write(sent_tokenize(text))

        with col2:
            st.markdown("### Word Tokenization")
            st.write(word_tokenize(text))

        with col3:
            st.markdown("### Character Tokenization")
            st.write(list(text))

    # TEXT CLEANING
    elif option == "Text Cleaning":
        st.subheader("Text Cleaning Output")

        text_lower = text.lower()
        cleaned_text = "".join(
            ch for ch in text_lower
            if ch not in string.punctuation and not ch.isdigit()
        )

        doc = nlp(cleaned_text)
        final_words = [token.text for token in doc if not token.is_stop]

        st.markdown("### Cleaned Text")
        st.write(" ".join(final_words))

    # REGULAR EXPRESSION
    elif option == "Regular Expression":
        st.subheader("Regular Expression Output")

        emails = re.findall(r"\S+@\S+", text)
        numbers = re.findall(r"\d+", text)

        st.markdown("### Extracted Emails")
        st.write(emails if emails else "No email found")

        st.markdown("### Extracted Numbers")
        st.write(numbers if numbers else "No numbers found")

    # STEMMING
    elif option == "Stemming":
        st.subheader("Stemming Output")

        words = word_tokenize(text)
        porter = PorterStemmer()
        lancaster = LancasterStemmer()

        df = pd.DataFrame({
            "Word": words,
            "Porter Stemmer": [porter.stem(w) for w in words],
            "Lancaster Stemmer": [lancaster.stem(w) for w in words]
        })

        st.dataframe(df, use_container_width=True)

    # LEMMATIZATION
    elif option == "Lemmatization":
        st.subheader("Lemmatization Output")

        doc = nlp(text)
        data = [(token.text, token.pos_, token.lemma_) for token in doc]

        df = pd.DataFrame(data, columns=["Word", "POS", "Lemma"])
        st.dataframe(df, use_container_width=True)

    # BAG OF WORDS
    elif option == "Bag of Words":
        st.subheader("Bag of Words (BoW)")

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform([text])

        df = pd.DataFrame({
            "Word": vectorizer.get_feature_names_out(),
            "Frequency": X.toarray()[0]
        }).sort_values(by="Frequency", ascending=False)

        st.dataframe(df, use_container_width=True)

    # TF-IDF
    elif option == "TF-IDF":
        st.subheader("TF-IDF Representation")

        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform([text])

        df = pd.DataFrame({
            "Word": tfidf.get_feature_names_out(),
            "TF-IDF Score": X.toarray()[0]
        }).sort_values(by="TF-IDF Score", ascending=False)

        st.dataframe(df, use_container_width=True)

    # WORD EMBEDDING
    elif option == "Word Embedding":
        st.subheader("Word Embeddings (spaCy)")

        doc = nlp(text)
        vectors = [(token.text, token.vector_norm) for token in doc]

        df = pd.DataFrame(vectors, columns=["Word", "Vector Norm"])
        st.dataframe(df, use_container_width=True)

        st.info("spaCy small model has limited word vectors. Norm shows relative magnitude.")