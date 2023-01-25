
import pickle
from fastapi import FastAPI
import uvicorn
import gunicorn
from pydantic import BaseModel
from lightgbm import LGBMClassifier
from sklego.meta import Thresholder


app = FastAPI()
pickle_in = open('common_files/mlflow_model/model.pkl', 'rb') # load threshold model for probability prediction\n",
classifier = pickle.load(pickle_in)

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

# index route, opens automatically on
@app.get('/')
def root():
    return {'message': 'Welcome! This is an fastapi application. You can make prdiction on the client credit default probability by entering values in listed features!'}


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

#if __name__ == '__main__':
#    uvicorn.run(app, host='127.0.0.1', port= 8000)
