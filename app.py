from flask import Flask, render_template, request
import pickle 
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')

def hello_world():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    input_values = request.form.values()
    float_features = [float(x) if x.replace('.', '', 1).isdigit() else 0.0 for x in input_values]
    
    final = [np.array(float_features)]
    # print(float_features)
    # print(final)
    
    prediction = model.predict(final)
    
    if len(prediction) > 0:
        output = '{0:.2f}'.format(prediction[0])
        
        if float(output) >= 8:
            return render_template('index.html', pred='Best Quality Wine as Quality Value is {}'.format(output))
        elif float(output) in range(6,8):
            return render_template('index.html', pred='Good Quality Wine as Quality Value is {}'.format(output))
        elif float(output) in range(4,6):
            return render_template('index.html', pred='Average Quality Wine as Quality Value is {}'.format(output))
        else:
            return render_template('index.html', pred='Bad Quality Wine as Quality Value is {}'.format(output))
    else:
        return render_template('index.html', pred='Prediction Error')


if __name__ == '__main__':
    app.run(debug=True)