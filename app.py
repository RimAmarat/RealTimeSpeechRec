# app.py
from flask import Flask
from main import print_hi
import os

app = Flask(__name__)

@app.route("/")
def index():
    result = print_hi('PyCharm')

    return f"<h1>{result}</h1>"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
