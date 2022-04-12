# Original repo: https://github.com/shanakaChathu/churn_model

from flask import Flask, render_template, request #, jsonify
import mlflow
import numpy as np
import os
import yaml

registry_uri = os.getenv('REGISTRY_URI')
# UNUSED: mlruns_directory = os.getenv('MLRUNS_DIRECTORY')
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

def get_model_path(model_name):
    client = mlflow.tracking.MlflowClient(registry_uri=registry_uri)
    model_path = client.get_latest_versions(name=model_name, stages=['Production'])
    if model_path:
        print("Prod loaded")
        return model_path[0].source
    model_path = client.get_latest_versions(name=model_name, stages=['Staging'])
    if model_path:
        print("Staging loaded")
        return model_path[0].source
    model_path = client.get_latest_versions(name=model_name, stages=['None'])
    if model_path:
        print("Unclassified model loaded")
        return model_path[0].source
    else:
        print("No model named '" + model_name + "' found on this MLFlow server")

def predict(data, model_path):
    model = mlflow.sklearn.load_model(model_path) # --> folder required
    # model = joblib.load(model_path) --> .pkl file required
    # model = pickle.load(open(model_path, 'rb')) --> .pkl file required
    prediction = model.predict(data).tolist()[0]
    return prediction

def form_response(dict_request, model_name):
    try:
        if validate_input(dict_request):
            data = dict_request.values()
            data = [list(map(float, data))]
            model_path = get_model_path(model_name)
            response = predict(data, model_path)
            return response
    except NotANumber as e:
        response =  str(e)
        return response 

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')    
    
@app.route('/tree', methods=['GET', 'POST'])
def tree():
    if request.method == 'POST':
        try:
            if request.form:
                dict_req = dict(request.form)
                response = form_response(dict_req, "tree_model")
                return render_template('tree_model.html', response=response)
        except Exception as e:
            print(e)
            error = {'error': "Something went wrong!! Try again later!"}
            error = {'error': e}
            return render_template('404.html', error=error)
    else:
        return render_template('tree_model.html')
    
@app.route('/nn', methods=['GET', 'POST'])
def nn():
    if request.method == 'POST':
        try:
            if request.form:
                dict_req = dict(request.form)
                response = form_response(dict_req, "neural_network")
                return render_template('nn_model.html', response=response)
        except Exception as e:
            print(e)
            error = {'error': "Something went wrong!! Try again later!"}
            error = {'error': e}
            return render_template('404.html', error=error)
    else:
        return render_template('nn_model.html')
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
