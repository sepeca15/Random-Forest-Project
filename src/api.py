import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open('../models/titanic_model.pickle', 'rb'))

input_data = (3,0,35,0,0)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)

def create_app(enviroment):
    app = Flask(__name__)
    return app

app = create_app()

app.route('/api/v1/users', methods=['GET'])
def get_users():
    response = {'message': 'success'}
    return jsonify(response)
""" app.route('/api/v1/users', methods=['GET'])
def get_users():
    response = {'message': 'success'}
    return jsonify(response) """