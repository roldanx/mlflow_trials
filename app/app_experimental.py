from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
import yaml

model_location="/home/roldanx/code/tfm/mlruns/1/d07c2eed5f494aeebcb396fe8af89d46/artifacts/model.pkl"
registry_uri="http://localhost:8000/"
mlruns_directory="/home/roldanx/code/tfm/mlruns"
template_dir="/home/roldanx/code/mlflow_trials/app/templates"

app = Flask(__name__,template_folder=template_dir)

class NotANumber(Exception):
    def __init__(self, message="Values entered are not Numerical"):
        self.message = message
        super().__init__(self.message)

def validate_input(dict_request):
    for _, val in dict_request.items():
        try:
            val=float(val)
        except Exception as e:
            raise NotANumber
    return True

def predict(data):
    model = joblib.load(model_location)
    prediction = model.predict(data).tolist()[0]
    return prediction

def form_response(dict_request):
    try:
        if validate_input(dict_request):
            data = dict_request.values()
            data = [list(map(float, data))]
            response = predict(data)
            return response
    except NotANumber as e:
        response =  str(e)
        return response 

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            if request.form:
                dict_req = dict(request.form)
                response = form_response(dict_req)
                return render_template("index.html", response=response)
        except Exception as e:
            print(e)
            error = {"error": "Something went wrong!! Try again later!"}
            error = {"error": e}
            return render_template("404.html", error=error)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
