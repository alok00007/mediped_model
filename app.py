from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np

#create flask app
app=Flask(__name__)

#load the pickle model
model=pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
@app.route("/")
def home():
    return render_template("index.html")
    

@app.route("/predict",methods=["POST"])
def predict():
    float_features=[float(x) for x in request.form.values()]
    features=[np.array(float_features)]
    features=scaler.transform(features)
    prediction=model.predict(features)
    return render_template("index.html",prediction_text="The flower is {}".format(prediction))
    

if __name__ == "__main__":
    #host = '0.0.0.0'  # Listen on all network interfaces
    #port = 58812       # Custom port number
    app.run(debug=True)