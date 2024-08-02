from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('Diabetesmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [float(x) for x in request.form.values()]
    input_data_as_numpy_array = np.asarray(input_data)
    input_reshape = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_reshape)

    if prediction[0] == 0:
        return render_template('index.html', prediction_text='The person is not diabetic')
    else:
        return render_template('index.html', prediction_text='The person is diabetic')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
