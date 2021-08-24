# import libraries
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from flask import Flask, request
import flasgger
from flasgger import Swagger

# flask app initialization
app = Flask(__name__)
Swagger(app)

# load saved model object 
with open('./model.pkl', 'rb') as model_pkl:
   lr_model= pickle.load(model_pkl)

# create api endpoint for the function of prediction
@app.route('/')
def welcome():
   return "Hello!"

@app.route('/predict', methods=["Get"])
def predict_churn():
   """Let's assign label  0 or 1 to the observations.
    This is using docstrings for specifications.
    ---
    parameters:
      - name: gen
        in: query
        type: number
        required: true
      - name: nl
        in: query
        type: number
        required: true
      - name: pt
        in: query
        type: number
        required: true
      - name: pf
        in: query
        type: number
        required: true
      - name: ph
        in: query
        type: number
        required: true
      - name: cp
        in: query
        type: number
        required: true
      - name: gv
        in: query
        type: number
        required: true
      - name: age
        in: query
        type: number
        required: true
      - name: aac
        in: query
        type: number
        required: true
      - name: mte
        in: query
        type: number
        required: true
      - name: lf
        in: query
        type: number
        required: true
      - name: aft
        in: query
        type: number
        required: true
      - name: afc
        in: query
        type: number
        required: true
    
    responses:
        200:
            description: The output values
         
   """       
   gen = request.args.get('gen')
   nl = request.args.get('nl')
   pt = request.args.get('pt')
   pf = request.args.get('pf')
   ph = request.args.get('ph')
   cp = request.args.get('cp')
   gv = request.args.get('gv')
   age = request.args.get('age')
   aac = request.args.get('aac')
   mte = request.args.get('mte')
   lf = request.args.get('lf')
   aft = request.args.get('aft')
   afc = request.args.get('afc')

# create test data
   test = np.array([[gen, nl, pt, pf, ph, cp, gv, age, aac, mte, lf, aft, afc]])
   prediction = lr_model.predict(test)
# return the resulted prediction
   return 'Predicted result for observation ' + str(test) + ' is: ' + str(prediction)


# If we want to preedict for observations in the file
@app.route('/predict_file',methods=["POST"])
def predict_churn_file():
   """Let's predict label -0 or 1 - for the observations.
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true

    responses:
        200:
            description: The output values

   """
   # create dataframe from imported file
   df_test=pd.read_csv(request.files.get("file"))
   print(df_test.head())
   prediction=lr_model.predict(df_test)
   return 'Predicted result for observations is: ' + str(list(prediction))

if __name__ == '__main__':
   app.run()

# features meaning
#    gender - 'gen'
#    near_location -'nl'
#    partner - 'pt'
#    promo_friends - 'pf'
#    phone - 'ph'
#    contract_period - 'cp'
#    group_visits - 'gv'
#    age - 'age'
#    avg_add_charges - 'aac'
#    months_to_end - 'mte'
#    lifetime - 'lf'
#    avg_frequency_total - 'aft'
#    avg_frequency_current - 'afc'

# example of test observation 1, 0, 0, 1, 1, 12, 1, 27, 100, 6, 6, 1.05, 0.95

# example of request in browser
#http://127.0.0.1:5000/predict?gen=1&nl=0&pt=0&pf=1&ph=1&cp=12&gv=1&age=27&aac=100&mte=6&lf=6&aft=1.05&afc=0.95
