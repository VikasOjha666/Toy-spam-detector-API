from flask import Flask,render_template,request,url_for
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import pandas as pd
app=Flask(__name__)

clf=joblib.load('model.pkl')
cnt_vec=CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("count_vec.pkl", "rb")))

@app.route('/')
def home():
    return render_template("MessageSubmission.html")

@app.route('/messagesubmission',methods=['POST','GET'])
def messagesubmission():
    message=request.form['msgbox']
    cust_data=pd.DataFrame([message],columns=['SMS'])
    vec=cnt_vec.transform(cust_data['SMS'])

    predicted_label=pd.DataFrame([clf.predict(vec)],columns=['pred'])
    predicted_label=predicted_label['pred'].map({1:'Spam',0:'Not Spam'}).values


    return render_template("prediction.html",pred=predicted_label[0])

app.run()
