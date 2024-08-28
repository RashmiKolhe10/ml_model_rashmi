from flask import Flask, render_template, request
import pickle
import numpy as np
import sklearn
# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_placement():
    try:
        
        cgpa = float(request.form.get('cgpa'))
        iq = int(request.form.get('iq'))
        profile_score = int(request.form.get('profile_score'))
        
        
        print(f"CGPA: {cgpa}, IQ: {iq}, Profile Score: {profile_score}")
        print(f"Prediction Array: {np.array([cgpa, iq, profile_score]).reshape(1, 3)}")
        
        # Prediction
        result = model.predict(np.array([cgpa, iq, profile_score]).reshape(1, 3))

        
        if result[0] == 1:
            result = 'placed'
        else:
            result = 'not placed'
    except Exception as e:
       
        print(f"Error during prediction: {e}")
        result = 'Error in prediction'

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
