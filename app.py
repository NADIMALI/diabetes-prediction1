#app.py
from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    # receive the values send by user in three text boxes thru request object -> requesst.form.values()
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    #print(final_features)
    #prediction = model.predict(final_features)
    #output = round(prediction[0], 2)
    
    prediction=model.predict_proba(final_features)
    output='{0:.{1}f}'.format(prediction[0][1], 2)
   
    #print(output )

    return render_template('index.html', pred='Probability of having Diabetes is :  {}'.format(output))

if __name__ == '__main__':
    app.run(debug=False)