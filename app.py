import pickle 
from flask import Flask,request,app,jsonify,url_for,render_template

import numpy as np
import pandas as pd

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',method=["POST"])
def predict_api():
    data=request.json['data']
    output=model.predict(np.array(list(data.values())).reshape(1,-1))
    return jsonify(output[0])

if __name__=='__main__':
    app.run(debug=True)