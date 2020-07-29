from flask import Flask, make_response, request
import pandas as pd, numpy as np, io, operator, csv
from sklearn.preprocessing import MinMaxScaler
import keras
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense ,Dropout
from keras.optimizers import RMSprop
app = Flask(__name__)
def transform():
    cols=['week','center_id','num_orders']
    df1=pd.read_csv('train.csv',usecols=cols)
    uniques=df1.center_id.unique()
    orders_per_center=[]
    for i in range(1,146):
        for j in uniques:
            orders_per_center.append(df1[df1['week']==i][df1['center_id']==j]['num_orders'].sum())
    weeks=[]
    centerids=[]
    for i in range(1,146):
        for j in range(0,77):
            centers=uniques[j]
            weeks.append(i)
            centerids.append(centers)
    finals=pd.DataFrame({'week':weeks,'center_id':centerids,'num_orders':orders_per_center})
    df_final=finals
    df=df_final.dropna()
    df.set_axis(df['week'], inplace=True)
    last_dates = df['week'].values[-1]
    uniques=df.center_id.unique()
    close_data = df['num_orders'].values
    close_data = close_data.reshape((-1,1))
    sc= MinMaxScaler(feature_range=(0,1))
    close_data = sc.fit_transform(close_data)
    split_percent = 0.98
    split = int(split_percent*len(close_data))
    close_train = close_data[:split]
    close_test = close_data[split:]
    date_train = df[['week','center_id']][:split]
    date_test = df[['week','center_id']][split:]
    print(len(close_train))
    print(len(close_test))
    look_back = 200
    train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=64)     
    test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)
    model = Sequential()
    model.add(LSTM(32,return_sequences=True,input_shape=(look_back,1)))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    num_epochs = 10
    model.fit_generator(train_generator, epochs=num_epochs, verbose=1)
    prediction = model.predict_generator(test_generator)
    close_train = close_train.reshape((-1))
    close_test = close_test.reshape((-1))
    prediction = prediction.reshape((-1))
    close_data = close_data.reshape((-1))
    def predict(num_prediction, model):
        prediction_list = close_data[-look_back:]
        for _ in range(num_prediction*77):
            x = prediction_list[-look_back:]
            x = x.reshape((1, look_back, 1))
            out = model.predict(x)[0][0]
            prediction_list = np.append(prediction_list, out)
        prediction_list = prediction_list[look_back-1:]
        return prediction_list
    def predict_ids(num_prediction):
        centerid=[]
        for i in range(1,num_prediction+1):
            for j in range(0,77):
                centerid.append(uniques[j])
        return centerid
    num_prediction = 10
    forecast = predict(num_prediction, model)
    forecast_ids = predict_ids(num_prediction)
    forecast_vals=sc.inverse_transform(forecast.reshape(-1,1))
    forecast=forecast_vals.reshape((-1))
    forecast=forecast.tolist()
    forecast.pop(0)
    forecastss=np.array(forecast)
    weeeks=[]
    for i in range(1,11):
        for j in range(0,77):
            weeeks.append(last_dates+i)
    finals=pd.DataFrame({'weeks':weeeks,'center_id':forecast_ids,'num_orders':np.round(forecastss,0)})
    return finals
@app.route('/')
def form():
  return """
    <html>
        <body style="background-color:#33FFEC;">
            <h1 style="text-align: center; color:#FF4933;" ><strong>OPTIMIZED WAREHOUSE MANAGEMENT</strong></h1>
<hr style="border: 2px dashed black;width:80%" />
<br>
<h3><em><span style="text-decoration: underline;margin-left:70px">Please Upload The CSV File</span></em></h3>
<form action="/transform" method="post" enctype="multipart/form-data">
<input type="file" name="data_file" style="background-color: #555;color: #fff;border-radius: 30px;text-align: center;border: 3px outset buttonface;fontstyle:bold;font-size:16px;margin-top:15px;margin-left:70px"/>


<input type="submit" value="SUBMIT FILE" style="margin-left:110px;background-color: #FFDE7A;border-radius: 30px;text-align: center;border: 2px outset buttonface;fontstyle:bold;font-size:16px;margin-top:15px"/>

<hr style="border: 1.25px dashed black;margin-top:25px;width:90%" />
<br>


<p><span style="color: #ff0000;text-decoration: underline;margin-left:70px"><em>INSTRUCTIONS</em></span></p>
<ol>
<li><span style="color: #050200;"><em>Please Upload only the Historical Data file.</em></span></li><br>
<li><span style="color: #050200;"><em>The Historical Data file must contain two columns: week and corresponding demand</em></span></li><br>
<li><span style="color: #050200;"><em>Click on the Choose file button to upload the file.</em></span></li><br>
<li><span style="color: #050200;"><em>Click on the SUBMIT FILE button to submit the file.</em></span></li><br>
<li><span style="color: #050200;"><em>Please wait for some time while training takes place.</em></span></li><br>
<li><span style="color: #050200;"><em>The prediction file automatically gets downloaded.</em></span></li><br>
<li><span style="color: #050200;"><em>The predictions are in the result.csv file. </em></span></li>
</ol>
            </form>
        </body>
    </html>
"""

@app.route('/transform', methods=["POST"])
def transform_view():
  request_file = request.files['data_file']
  request_file.save("train.csv")
  if not request_file:
    return "No file"
  result = transform()
  print(result)
  response = make_response(result.to_csv())
  response.headers["Content-Disposition"] = "attachment; filename=result.csv"
  return response
if __name__ == '__main__':
    app.run(debug=True)#app.run(host='0.0.0.0', port=8005, debug=True)
