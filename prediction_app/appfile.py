
import pandas as pd
import numpy as np
import pickle
from fastapi import FastAPI
import uvicorn
from typing import List
from pydantic import BaseModel
from sklego.meta import Thresholder

# define class which describes test data measurements
class class_testdata(BaseModel):
    
    DaysEmployed: float
    ExtSource3: float
    DaysBirth: float
    ApartmentAvg: float
    DaysRegistration: float
    CreditTypeMicroloan: float
    PrevDaysDecisionMin: float
    IdClient: int
    
    #class class_item(BaseModel):
    #    prediction: float        
    # app object creation
app = FastAPI()
pickle_in = open('C:\\Users\\USER\\Documents\\P7dataset\\mlflow_model\\model.pkl', 'rb') # load threshold model for probability prediction\n",
classifier = pickle.load(pickle_in)
# index route, opens automatically on
@app.get('/')
def index():
    return {'message': 'Welcome!'}

# route with a single parameter
@app.get('/{number}')
def get_name(name: int):
    return {'Welcome to page': {number}}

# make prediction functionality\n",
#@app.post('/predict', response_model=class_item) # the API's name
@app.post('/predict')
async def client_predict(data: class_testdata):
    data = data.dict()
    IdClient= data['IdClient']
    DaysEmployed= data['DaysEmployed']
    ExtSource3= data['ExtSource3']
    DaysBirth= data['DaysBirth']
    ApartmentAvg= data['ApartmentAvg']
    DaysRegistration= data['DaysRegistration']
    CreditTypeMicroloan= data['CreditTypeMicroloan']
    PrevDaysDecisionMin = data['PrevDaysDecisionMin']
    
    prediction = classifier.predict_proba([[DaysEmployed, ExtSource3,  
                                   DaysBirth, ApartmentAvg, DaysRegistration, CreditTypeMicroloan, PrevDaysDecisionMin]])
    
    prediction_default = prediction[0][0]
   
    return prediction_default
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port= 8000)
