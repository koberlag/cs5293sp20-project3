import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import spacy
import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath("app.py")), 'project3')
punctuations = string.punctuation
nlp=spacy.load("en_core_web_sm")
parser = English()


def spacy_tokenizer(sentence):
    """Uses the spaCy library to tokenize sentences, lemmatizing each token and converting to lowercase."""

     # Creating our token object, which is used to create documents with linguistic annotations.
    tokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    tokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens ]

    # Removing stop words
    tokens = [ word for word in tokens if word not in STOP_WORDS and word not in punctuations ]

    # return preprocessed list of tokens
    return tokens


def evaluate_model(classifier, features, labels):
    """Used for testing model classifier. Uses cross_val_score method to cross validate
    the classifier using the features and labels from the yummly data."""

    # Cross-validate the classifier
    cv_scores = cross_val_score(classifier, features, labels, cv=5)
    cv_mean_score = np.mean(cv_scores)
    print('CV Accuracy (5-fold):', cv_scores)
    print('Mean CV Accuracy:', cv_mean_score)


def create_model_vec_matrix(df):
    """Creates the vectorizer, classification model, and feature matrix from the yummly data frame.
    Uses the pickle functionality to persist the three objects."""

    # Create vectorizer object
    vectorizer = create_vectorizer()
   
    # Combine ingredients into a single string, per cuisine instance (id)
    recipes = [' '.join(ingredient) for ingredient in df['ingredients']] # the features we want to analyze
    # Get the cuisine labels
    cuisines = np.array(df['cuisine']) # the labels, or answers, we want to test against

    # fit and transform recipes into features
    feature_matrix = vectorizer.fit_transform(recipes)

    # Create Classifier
    classifier = create_classifier()

    # Fit model from classifier, features and labels
    model = classifier.fit(feature_matrix, cuisines)

    # Create persisted files for the three objects
    pickle_model_vec_matrix(model, vectorizer, feature_matrix)

    return model, vectorizer, feature_matrix


def get_model_vec_matrix(df):
    """Retrieves the pickle files that are persisted for the vectorizer, classification model, and feature matrix, or
    if they do yet exists, a the objects are created and persisted."""

    model = None
    vectorizer = None
    feature_matrix = None
    try:
        # Load the persisted objects
        model = pickle.load(open(PROJECT_DIR + os.path.sep + "models" + os.path.sep +  "finalized_model.sav", 'rb'))
        vectorizer = pickle.load(open(PROJECT_DIR + os.path.sep + "models" + os.path.sep +  "vectorizer.pickle", "rb"))
        feature_matrix = pickle.load(open(PROJECT_DIR + os.path.sep + "models" + os.path.sep +  "feature_matrix.pickle", "rb"))
    except Exception as ex:
        print(ex)

    if(model is None):
        # Create objects and persist them
        model, vectorizer, feature_matrix = create_model_vec_matrix(df)
    return model, vectorizer, feature_matrix


def pickle_model_vec_matrix(model, vectorizer, feature_matrix):
    """Creates the persisted files for the model, vecorizer, and feature matrix"""

    # save the model and vectorizer to disk
    pickle.dump(model, open(PROJECT_DIR + os.path.sep + "models" + os.path.sep + 'finalized_model.sav', 'wb'))
    pickle.dump(vectorizer, open(PROJECT_DIR + os.path.sep + "models" + os.path.sep +  "vectorizer.pickle", "wb"))
    pickle.dump(feature_matrix, open(PROJECT_DIR + os.path.sep + "models" + os.path.sep + "feature_matrix.pickle", "wb"))
    

def create_vectorizer():
    """Creates and returns a count vectorizer."""

    return CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))


def create_classifier():
    """Creates and returns a logistics regression classifier"""

    classifier = None
    # Logistic Regression
    classifier = LogisticRegression(penalty='l2', max_iter=1000, C=1, random_state=42)

    return classifier