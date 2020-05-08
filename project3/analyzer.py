import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from project3.reader import read_yummly_json
from project3.modeler import get_model_vec_matrix



def get_best_match_cuisine(vectorizer, model, feature_matrix):
    """Used the trained model and the feature matrix of the user input, to determine
    the best matching cuisine."""

    # Gets the predicted cuisine array
    predicted_cuisines = model.predict(feature_matrix)
    # Gets the first element
    best_match_cuisine = predicted_cuisines[0]
    # Get the highest probability of the model
    best_match_cuisine_prob = model.predict_proba(feature_matrix).max()
    return best_match_cuisine, best_match_cuisine_prob

def get_closest_recipes(feature_matrix, user_recipe_feature_matrix, df, n_recipes):
    """Uses the feature matrix from the user input and the feature matrix used to build the model
    as input into the cosine similarity function. Returns the ids and the similarity score as a tuple."""

    # Gets similarity score between feature matrix and the user_recipe_feature_matrix
    cosine = cosine_similarity(feature_matrix.todense(), user_recipe_feature_matrix.todense())
    # Add similarity data to original data frame
    df["similarity"] = pd.DataFrame(cosine)[0]
    #Sort data frame by similarity, descending
    sorted_df = df.sort_values(by=["similarity"], ascending=False)
    # Get the top n recipes
    closest_recipes = sorted_df.head(n_recipes)
    return closest_recipes

  
def analyze_ingredients(user_ingredients):
    """Accepts a list of user input ingredients and creates a feature matrix
    for using in the existing model. The model will return the best match for
    cuisine type. In addition a closest recipes are determined using cosine similarity.
    Returns the cuisine type and prob as a tuple as well as the closest recipes as a tuple
    of the recipe id and similarity score."""

    # Set user input to be lower case
    user_ingredients = [ingredient.lower() for ingredient in user_ingredients]

    # number of similar recipes to return
    n_recipes = 4

    # Read Yummly Json Returns data frame
    df = read_yummly_json()

    # Get the classifier model, vectorizer and yummly data feature matrix
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