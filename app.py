from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__, template_folder='templates')
model = pickle.load(open('model2.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    Age = int(request.form.get('age'))
    Workclass = request.form.get('workclass')
    Education = request.form.get('education')
    marital_status = request.form.get('marital_status')
    occupation = request.form.get('occupation')
    sex = request.form.get('sex')
    hours_per_week = int(request.form.get('hours_per_week'))
    col=["Age", "hours_per_week", "Workclass", "Education", "marital_status", "occupation", "sex"]
    output = model.predict(pd.DataFrame(np.array([Age, hours_per_week, Workclass, Education, marital_status, occupation, sex]).reshape(1, 7),columns=col))[0]
    if output == 1:
        result = "INCOME IS >50K"
    else:
        result = "INCOME IS <=50K"
    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
