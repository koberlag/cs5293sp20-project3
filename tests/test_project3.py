# import project3
from project3 import reader, modeler, analyzer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

df = reader.read_yummly_json()

def test_read_yummly_json():
    df = reader.read_yummly_json()
    assert isinstance(df, pd.DataFrame)
    

def test_spacy_tokenizer():
    tokens = modeler.spacy_tokenizer("A token test")
    # A is a stop word and should not be returned
    assert tokens[0] == "token"
    assert tokens[1] == "test"
    


def test_create_model_vec_matrix():

    model, vec, matrix = modeler.create_model_vec_matrix(df)

    assert isinstance(model, LogisticRegression)
    assert isinstance(vec, CountVectorizer)
    assert matrix is not None


def test_get_model_vec_matrix():
   
    model, vec, matrix = modeler.get_model_vec_matrix(df)

    assert isinstance(model, LogisticRegression)
    assert isinstance(vec, CountVectorizer)
    assert matrix is not None


def test_create_vectorizer():
    assert isinstance(modeler.create_vectorizer(), CountVectorizer)


def test_create_classifier():
    assert isinstance(modeler.create_classifier(), LogisticRegression)



def test_get_best_match_cuisine():

    model, vec, matrix = modeler.get_model_vec_matrix(df)
    features = vec.transform(["tomato"])
    best_match_cuisine, best_match_cuisine_prob = analyzer.get_best_match_cuisine(vec, model, features)
    assert best_match_cuisine is not None
    assert best_match_cuisine_prob > 0

def test_get_closest_recipes():
    model, vec, matrix = modeler.get_model_vec_matrix(df)
    features = vec.transform(["tomato"])
    closest_recipes = analyzer.get_closest_recipes(matrix, features, df, 2)
    assert closest_recipes is not None
    assert len(closest_recipes) == 2

  
def test_analyze_ingredients():
    best_match, closest_matches = analyzer.analyze_ingredients(["tomato"])
    assert best_match[0] is not None
    assert best_match[1] > 0
    assert closest_matches[0][0] is not None
    assert closest_matches[0][1] >= 0
    assert len(closest_matches) == 4