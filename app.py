from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)


@app.route('/')
def main():
    return render_template("main.html")


@app.route("/result", methods=["POST"])
def result():
    if request.method == "POST":
        gender = request.form.get("gender")
        ageCategory = request.form.get("Age Category")
        race = request.form.get("Race")
        bmi = request.form.get("Bmi")
        sleepTime = request.form.get("Sleep Time")
        smoking = request.form.get("Smoking")
        alcoholDrinking = request.form.get("Alcohol Drinking")
        stroke = request.form.get("Stroke")
        diabetic = request.form.get("Diabetic")
        asthma = request.form.get("Asthma")
        kidneyDisease = request.form.get("Kidney Disease")
        skinCancer = request.form.get("Skin Cancer")
        difficultyWalking = request.form.get("Difficulty Walking")
        physicalActivity = request.form.get("Physical Activity")
        physicalHealth = request.form.get("Physical Health")
        mentalHealth = request.form.get("Mental Health")
        generalHealth = request.form.get("General Health")

        pred_args = [bmi, smoking, alcoholDrinking, stroke, physicalHealth,
                     mentalHealth, difficultyWalking, gender, ageCategory, race, diabetic,
                     physicalActivity, generalHealth, sleepTime, asthma, kidneyDisease, skinCancer]

        data = np.array(pred_args)
        data = data.reshape(1, 17)

        model = joblib.load(open('heart_disease_pred_model.pkl', 'rb'))
        prediction = model.predict(data)

        if prediction == 1:
            return render_template("result.html", result="You probably have heart disease")
        else:
             return render_template("result.html", result="You probably don't have heart disease")"

   
if __name__ == "__main__":
    app.run(debug=True)
