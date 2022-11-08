from flask import Flask, render_template, request
from models.neural_net import *
import pandas as pd
from data import *

app = Flask(__name__)
model = torch.load('./long-range.pt')
df = pd.read_csv('data/long-range.csv')
X = np.array(df[['KMS', 'Age']])
scaler = MinMaxScaler()
scaler.fit_transform(X)


def predict_input(kms, age):
    pred = torch.Tensor(scaler.transform([[kms, age]]))
    pred = model(pred).detach().numpy()[0][0]
    return int(pred)


@app.route('/', methods=['GET', 'POST'])
def index():
    # if request.method == 'POST':
    try:
        kms = request.form.get('kms')
        age = request.form.get('age')
        prediction = predict_input(kms, age)
    except:
        prediction = 0

    return render_template('base.html', prediction_info=(prediction, kms, age))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # kms = request.form.get('kms')
    # age = request.form.get('age')
    # prediction = predict_input(kms, age)
    return 'prediction xD'


if __name__ == '__main__':
    app.run(debug=True)