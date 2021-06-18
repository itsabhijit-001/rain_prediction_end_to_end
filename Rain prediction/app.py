from flask import Flask,render_template,request
import pickle
import xgboost
import numpy as np
# import x

model1 = pickle.load(open('model1.pkl', 'rb'))
model2=pickle.load(open('model2.pkl','rb'))
direction_encoder=pickle.load(open('direction_encoder.pkl','rb'))
prediction_encoder=pickle.load(open('prediction_encoder.pkl','rb'))
app=Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/prediction_of_rain",methods=['GET','POST'])
def predict_rain():
    inputs=[]
    inputs2=[]
    tomorrow_prediction=-1
    today_prediction=-1
    today=0
    tomorrow=0
    if request.method=='POST':
        # print('NOt working')
        gust_dir=int(direction_encoder.transform([request.form.get('gustDirection')])[0])
        # print(gust_dir)
        gust_speed=int(request.form.get('gustSpeed'))
        wind_dir_9=int(direction_encoder.transform([request.form.get('direction9')])[0])
        wind_dir_3=int(direction_encoder.transform([request.form.get('direction3')])[0])
        wind_speed_9=int(request.form.get('speed9'))
        wind_speed_3=int(request.form.get('speed3'))
        humidity_9=int(request.form.get('humidity9'))
        humidity_3=int(request.form.get('humidity3'))
        pressure=int(request.form.get('pressure'))
        temperature=int(request.form.get('maxTemp'))

        inputs=[gust_dir,gust_speed,wind_dir_9,wind_dir_3,wind_speed_9,wind_speed_3,humidity_9,humidity_3,pressure,temperature]
        # print(inputs)
        inputs_np=np.array([inputs])
        # print(inputs_np.shape)
        # print(inputs_np)
        today_prediction=model1.predict(inputs_np)[0] 
        today=model1.predict_proba(inputs_np)[0][1]
        print(today)
        # print(inputs)
        inputs2=[gust_dir,gust_speed,wind_dir_9,wind_dir_3,wind_speed_9,wind_speed_3,humidity_9,humidity_3,pressure,temperature,today_prediction]
        inputs2_np=np.array([inputs2])
        # print(inputs2)
        # print(inputs2_np)
        tomorrow_prediction=model2.predict(inputs2_np)
        tomorrow=model2.predict_proba(inputs2_np)[0][1]
        print(tomorrow)
    else:
        print('NOt working')
    return render_template('index.html',today_rain=today_prediction,tomorrow_rain=tomorrow_prediction,today=today,tomorrow=tomorrow)

if __name__=='__main__':
    app.run(debug=True)