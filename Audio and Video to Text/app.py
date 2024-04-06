import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# creating flask app
app = Flask(__name__)

# Load the pickle model
model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def Home():
    return render_template("index.html", processed_text = transcript.text)


import assemblyai as aai

aai.settings.api_key = "38949b69a551445a81d16241e78e1c9d"
transcriber = aai.Transcriber()

transcript = transcriber.transcribe("./Happy Birthday song.mp3")
# transcript = transcriber.transcribe("./my-local-audio-file.wav")https://storage.googleapis.com/aai-web-samples/news.mp4

print(transcript.text)


@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text="The result is {}".format(prediction))


if __name__ == "__main__":
    app.run(debug=False)
