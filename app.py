from flask import Flask, render_template, request
import pickle 
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')

def hello_world():
    return render_template('wine.html')

