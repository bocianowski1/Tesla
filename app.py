from flask import Flask, render_template, request
from models.neural_net import *
import pandas as pd
import numpy as np
import locale

# Norway
locale.setlocale(locale.LC_ALL, 'no_NO.ISO8859-1')

from data import *

app = Flask(__name__)
model = torch.load('./long-range.pt')
df = pd.read_csv('data/long-range.csv')
X = np.array(df[['KMS', 'Age']])
scaler = MinMaxScaler()
scaler.fit_transform(X)


def number_to_locale(price: int or float) -> str:
    try:
        price = locale.currency(int(price), grouping=True)
        return price[2:-3]
    except:
        return price

def predict_input(kms: int, age: int) -> int:
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

    return render_template('base.html', prediction=prediction, 
                            price=number_to_locale(prediction), 
                            kilometers=number_to_locale(kms), age=age)

if __name__ == '__main__':
    app.run(debug=True)