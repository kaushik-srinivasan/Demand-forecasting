# Demand-forecasting
IBM hack challenge 2020

Developed custom neural networks for real-time training.
Reshaped training data to conform with LSTM input shape.

Install Python, preferably Python 3.6.x
Install the following libraries: Flask, Pandas, Numpy, Scikit-Learn, Sklearn, Tensorflow 1.14.0, Keras 2.2.5

If you are using Windows without any GPU then run the following command in the command prompt. 
> pip install numpy pandas scikit-Learn sklearn flask tensorflow==1.14.0 keras==2.2.5

If you are using Windows with a GPU (3 GB and above) then download and install required CUDA and CUDNN files. (required versions keep changing, please look it up)
Then run the following command in the command prompt. 
> pip install numpy pandas scikit-Learn sklearn flask tensorflow-gpu==1.14.0 keras==2.2.5

Run app.py and copy the local host IP Address (127.0.0.1:#port number displayed) URL in the browser (any browser, tested with chrome, edge and opera).

Upload the train.csv (hystorical data to be trained) file, throught the web page.
Click the "Submit" button to initiate training.
Wait for few minutes. The result.csv file will automatically download.

Link for project video- 1) https://www.youtube.com/watch?v=YAodXxWqkQM
                        2) https://www.youtube.com/watch?v=Erwu8kggJqU

You can also try the prediction.py on the same train dataset for predictions corresponding to a centre-meal id pair.




