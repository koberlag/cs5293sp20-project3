from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/analyze",  methods=['POST'])
def add():
    ingredients = request.form['ingredients'].split(",")
    return redirect(url_for("home"))
    
if __name__ == "__app__":
    app.run(debug=True)