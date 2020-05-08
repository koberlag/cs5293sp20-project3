from flask import Flask, render_template, request, redirect, url_for
from project3.ingredients import INGREDIENTS
from project3.analyzer import analyze_ingredients

app=Flask(__name__)



@app.route("/")
def home():
    return render_template("home.html", ingredient_list=INGREDIENTS)

@app.route("/analyze",  methods=['POST'])
def analyze():
    ingredients = request.form['ingredients'].split(",")
    best_match_cuisine, closest_match_recipes = analyze_ingredients(ingredients)
    return render_template("analyze.html", best_match_cuisine=best_match_cuisine, closest_match_recipes=closest_match_recipes)
    
if __name__ == "__app__":
    app.run(debug=True)