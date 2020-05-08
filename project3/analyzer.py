import os
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle
import numpy.linalg as LA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine

punctuations = string.punctuation
nlp=spacy.load("en_core_web_sm")
parser = English()

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath("app.py")), 'project3')

def readYummlyJson(): 
    """Load the yummly.json data into a pandas dataframe and return"""

    df = pd.DataFrame()
    try:
        file_path = PROJECT_DIR + os.path.sep + "yummly.json"
        df = pd.read_json(file_path)
    except Exception as ex:
        print(ex)
    return df


# Creating our tokenizer function
def spacy_tokenizer(sentence):
     # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in STOP_WORDS and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens

# Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()

# def analyze_ingredients(ingredients):
#     df = readYummlyJson()
#     # df = df.head(100)
#     # cv = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
#     cv = TfidfVectorizer(tokenizer = spacy_tokenizer)

#     # bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
#     # tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)
#     # # df= df.explode('ingredients').reset_index()
#     # X = [' '.join(x) for x in df['ingredients']] #  the features we want to analyze
#     # ylabels = np.array(df['cuisine']) # the labels, or answers, we want to test against
#     # # ynames = df['id']
#     # # X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(X, ylabels,ynames, test_size=0.3)
#     # X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3)
   
#     X = [' '.join(x) for x in df['ingredients']] # the features we want to analyze
#     ylabels = np.array(df['cuisine']) # the labels, or answers, we want to test against
#     # ynames = df['id']
#     train_corpus, test_corpus, train_labels, test_labels = train_test_split(X, ylabels, test_size=0.3)
#     # train_corpus, test_corpus, train_label_nums, test_label_nums, train_label_names, test_label_names = train_test_split(X, ylabels,ynames, test_size=0.3)

#     cv_train_features = cv.fit_transform(X)
#     # transform test articles into features
#     cv_test_features = cv.transform(test_corpus)

#     # print('BOW model:> Train features shape:', cv_train_features.shape,
#     #     ' Test features shape:', cv_test_features.shape)

#     # Logistic Regression
#     lr = LogisticRegression(penalty='l2', max_iter=100, C=1, random_state=42)
#     model = lr.fit(cv_train_features, ylabels)
#     lr_bow_cv_scores = cross_val_score(lr, cv_train_features, ylabels, cv=5)
#     lr_bow_cv_mean_score = np.mean(lr_bow_cv_scores)
#     print('CV Accuracy (5-fold):', lr_bow_cv_scores)
#     print('Mean CV Accuracy:', lr_bow_cv_mean_score)
#     # lr_bow_test_score = lr.score(cv_test_features, test_labels)
#     # print('Test Accuracy:', lr_bow_test_score)
#     # print('Top Test Cuisine', model.predict(cv_test_features)[0])
#     # print('Top Test Cuisine Prob', model.predict_proba(cv_test_features).max())

#     user_corpus = [' '.join(x) for x in ingredients] 
#     user_features = cv.transform(ingredients)

#     print('Top Test Cuisine', model.predict(user_features)[0])
#     print('Top Test Cuisine Prob', model.predict_proba(user_features).max())
#     dd = 0

def train_model():
    pass
def evaluate_model(classifier, features, labels):
    cv_scores = cross_val_score(classifier, features, labels, cv=5)
    cv_mean_score = np.mean(cv_scores)
    print('CV Accuracy (5-fold):', cv_scores)
    print('Mean CV Accuracy:', cv_mean_score)
    # test_score = lr.score(test_features, test_labels)
    # print('Test Accuracy:', test_score)
    # print('Top Test Cuisine', model.predict(test_features)[0])
    # print('Top Test Cuisine Prob', model.predict_proba(test_features).max())

# def test_model():
#     df = readYummlyJson()
#     # df = df.head(100)
#     vectorizer = createVectorizer("")
   
#     X = [' '.join(x) for x in df['ingredients']] # the features we want to analyze
#     ylabels = np.array(df['cuisine']) # the labels, or answers, we want to test against
#     # ynames = df['id']

#     # fit and transform ingredients into features
#     features = vectorizer.fit_transform(X)

#     # transform cuisines into features
#     labels = vectorizer.transform(ylabels)

#     # Logistic Regression
#     classifier = LogisticRegression(penalty='l2', max_iter=100, C=1, random_state=42)
#     model = classifier.fit(features, labels)
    
#     evaluate_model(classifier, features, labels)

#     user_corpus = [' '.join(x) for x in ingredients] 
#     user_features = vectorizer.transform(user_corpus)

#     print('Top Test Cuisine', model.predict(user_features)[0])
#     print('Top Test Cuisine Prob', model.predict_proba(user_features).max())
#     dd = 0


def get_best_match_cuisine(vectorizer, model, feature_matrix):

    predicted_cuisines = model.predict(feature_matrix)
    best_match_cuisine = predicted_cuisines[0]
    best_match_cuisine_prob = model.predict_proba(feature_matrix).max()
    return best_match_cuisine, best_match_cuisine_prob

def analyze_ingredients(user_ingredients):

    # number of similar recipes to return
    n_recipes = 4

    # Read Yummly Json Returns data frame
    df = readYummlyJson()

    # Get the classifeier model, vectorizer and yummly data feature matrix
    model, vectorizer, feature_matrix = get_model_vec_matrix(df)

    # Get feature matrix of user input
    user_recipe_feature_matrix = vectorizer.transform(user_ingredients)

    # Get the best match cuisine and probability
    best_match_cuisine, best_match_cuisine_prob = get_best_match_cuisine(vectorizer, model, user_recipe_feature_matrix)

    # Get the top n closest recipes
    closest_recipes = get_closest_recipes(feature_matrix, user_recipe_feature_matrix, df, n_recipes)

    # create tuple of cuisine and prob
    best_match = (best_match_cuisine, round(best_match_cuisine_prob, 2))

    # create tuple of id and similarity score
    closest_matches = [(r_id,round(similarity,2)) for r_id, similarity in zip(closest_recipes["id"], closest_recipes['similarity'])]

    return (best_match, closest_matches)

def get_closest_recipes(feature_matrix, user_recipe_feature_matrix, df, n_recipes):
    cosine = cosine_similarity(feature_matrix.todense(), user_recipe_feature_matrix.todense())
    df["similarity"] = pd.DataFrame(cosine)[0]
    sorted_df = df.sort_values(by=["similarity"], ascending=False)
    closest_recipes = sorted_df.head(n_recipes)
    return closest_recipes



def get_model_vec_matrix(df):
    model = None
    vectorizer = None
    feature_matrix = None
    try:
        model = pickle.load(open(PROJECT_DIR + os.path.sep + "models" + os.path.sep +  "finalized_model.sav", 'rb'))
        vectorizer = pickle.load(open(PROJECT_DIR + os.path.sep + "models" + os.path.sep +  "vectorizer.pickle", "rb"))
        feature_matrix = pickle.load(open(PROJECT_DIR + os.path.sep + "models" + os.path.sep +  "feature_matrix.pickle", "rb"))
    except Exception as ex:
        print(ex)

    if(model is None):
        model, vectorizer, feature_matrix = create_model_vec_matrix(df)
    return model, vectorizer, feature_matrix


def pickle_model_vec_matrix(model, vectorizer, feature_matrix):
    # save the model and vectorizer to disk
    pickle.dump(model, open(PROJECT_DIR + os.path.sep + "models" + os.path.sep + 'finalized_model.sav', 'wb'))
    pickle.dump(vectorizer, open(PROJECT_DIR + os.path.sep + "models" + os.path.sep +  "vectorizer.pickle", "wb"))
    pickle.dump(feature_matrix, open(PROJECT_DIR + os.path.sep + "models" + os.path.sep + "feature_matrix.pickle", "wb"))
    

def create_vectorizer():
    return CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))


def create_classifier():
    classifier = None
    # Logistic Regression
    classifier = LogisticRegression(penalty='l2', max_iter=1000, C=1, random_state=42)

    return classifier


def create_model_vec_matrix(df):

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

    pickle_model_vec_matrix(model, vectorizer, feature_matrix)

    return model, vectorizer, feature_matrix
  

    # df = readYummlyJson()
    # # df = df.head(100)
    # vec = createVectorizer("")

    # # bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
    # # tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)
    # # # df= df.explode('ingredients').reset_index()
    # # X = [' '.join(x) for x in df['ingredients']] #  the features we want to analyze
    # # ylabels = np.array(df['cuisine']) # the labels, or answers, we want to test against
    # # # ynames = df['id']
    # # # X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(X, ylabels,ynames, test_size=0.3)
    # # X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3)
   
    # X = [' '.join(x) for x in df['ingredients']] # the features we want to analyze
    # ylabels = np.array(df['cuisine']) # the labels, or answers, we want to test against
    # # ynames = df['id']
    # train_corpus, test_corpus, train_labels, test_labels = train_test_split(X, ylabels, test_size=0.3)
    # # train_corpus, test_corpus, train_label_nums, test_label_nums, train_label_names, test_label_names = train_test_split(X, ylabels,ynames, test_size=0.3)

    # cv_train_features = cv.fit_transform(train_corpus)
    # # transform test articles into features
    # cv_test_features = cv.transform(test_corpus)

    # # print('BOW model:> Train features shape:', cv_train_features.shape,
    # #     ' Test features shape:', cv_test_features.shape)

    # # Logistic Regression
    # lr = LogisticRegression(penalty='l2', max_iter=100, C=1, random_state=42)
    # model = lr.fit(cv_train_features, train_labels)
    # lr_bow_cv_scores = cross_val_score(lr, cv_train_features, train_labels, cv=5)
    # lr_bow_cv_mean_score = np.mean(lr_bow_cv_scores)
    # print('CV Accuracy (5-fold):', lr_bow_cv_scores)
    # print('Mean CV Accuracy:', lr_bow_cv_mean_score)
    # lr_bow_test_score = lr.score(cv_test_features, test_labels)
    # print('Test Accuracy:', lr_bow_test_score)
    # print('Top Test Cuisine', model.predict(cv_test_features)[0])
    # print('Top Test Cuisine Prob', model.predict_proba(cv_test_features).max())

    # user_corpus = [' '.join(x) for x in ingredients] 
    # user_features = cv.transform(user_corpus)

    # print('Top Test Cuisine', model.predict(user_features)[0])
    # print('Top Test Cuisine Prob', model.predict_proba(user_features).max())
    # dd = 0


    # # predicted = model.predict()
    # # Logistic Regression Classifier
    # classifier = LogisticRegression()

    # # Create pipeline using Bag of Words
    # pipe = Pipeline([("cleaner", predictors()),
    #                 ('vectorizer', tfidf_vector),
    #                 ('classifier', classifier)])

    # # model generation
    # pipe.fit(X_train,y_train)
    # pred = pipe.predict_proba(X_test)

    # from sklearn import metrics
    # # Predicting with a test dataset
    # predicted = pipe.predict(X_test)

    # # Model Accuracy
    # print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
    # print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted,average='micro'))
    # print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted, average='micro'))
    # # Predicting with a test dataset
    # predicted = pipe.predict(' '.join(ingredients))
    # print(predicted)
   


# def getUniqueIngredients():
#     ingredients = set()
#     data = loadYummlyJson()
#     for d in data:
#         ingredients.update(d["ingredients"])
#     ingredients = list(ingredients)
#     ingredients.sort()
#     return ingredients


# def writeUniqueIngredientsToJson():
#     ingredients = getUniqueIngredients()
#     # Serializing json  
#     json_object = json.dumps(ingredients, indent = 4) 
  
#     # Writing to ingredients.json 
#     with open("ingredients.py", "w") as outfile: 
#         outfile.write(json_object) 
