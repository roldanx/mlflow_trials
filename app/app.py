# Original repo: https://github.com/shanakaChathu/churn_model

from flask import Flask, render_template, request #, jsonify
import mlflow
import numpy as np
import os
import yaml

registry_uri = os.getenv('REGISTRY_URI')
mlruns_directory = os.getenv('MLRUNS_DIRECTORY')
model_location= os.path.join(mlruns_directory, '1/d07c2eed5f494aeebcb396fe8af89d46/artifactsX')
template_dir='./templates'

app = Flask(__name__,template_folder=template_dir)

class NotANumber(Exception):
    def __init__(self, message="Values entered are not Numerical"):
        self.message = message
        super().__init__(self.message)

# This function only allows numerical input. This could rise the next warning:
# "UserWarning: X does not have valid feature names"
# This means that, for SKLearn, input should be of type {column:value}.
def validate_input(dict_request):
    for _, val in dict_request.items():
        try:
            val=float(val)
        except Exception as e:
            raise NotANumber
    return True

def predict(data):
    model = mlflow.sklearn.load_model(model_location) # --> folder required
    # model = joblib.load(model_location) --> .pkl file required
    # model = pickle.load(open(model_location, 'rb')) --> .pkl file required
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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            if request.form:
                dict_req = dict(request.form)
                response = form_response(dict_req)
                return render_template('index.html', response=response)
        except Exception as e:
            print(e)
            error = {'error': "Something went wrong!! Try again later!"}
            error = {'error': e}
            return render_template('404.html', error=error)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
