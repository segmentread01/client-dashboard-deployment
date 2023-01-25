
#import sys
#sys.path.append("prediction_app/")
from appfile import client_predict

import pandas as pd
import numpy as np
import streamlit as st
import requests
import json

import plotly
import plotly.graph_objects as go
#import plotly.express as px
from matplotlib import pyplot as plt

import pickle
import shap
from streamlit_shap import st_shap

# config
st.set_page_config(layout = 'wide')

reduce_header_height_style = """
    <style>
        div.block-container {padding-top:0rem;}
    </style>
"""
st.markdown(reduce_header_height_style, unsafe_allow_html=True)

prediction_data = pd.read_csv("common_files/features_test.csv")
#prediction_data = prediction_data.drop(columns='AmtIncomeTotal')
# loading the client info file prepared in modeling code
client_info = pd.read_csv('common_files/test_client_info.csv')

prediction_data_info = pd.merge(prediction_data, client_info, how='inner', left_on='IdClient', right_on='SK_ID_CURR')
#prediction_data_info = prediction_data_info.drop(columns='TARGET')
prediction_data_info.set_index('IdClient', inplace=True)

#avoid int64 erreur: convert to int
prediction_data_info['DaysEmployed'] = prediction_data_info['DaysEmployed'].astype('float')
prediction_data_info['DaysBirth'] = prediction_data_info['DaysBirth'].astype('float')

    #api_choice = st.sidebar.selectbox(
    #    'API name',
    #    ['MLflow'])


IdClient = st.sidebar.selectbox('Client ID to predict', [el for el in list(prediction_data_info.index)])

DaysEmployed= prediction_data_info.loc[IdClient, 'DaysEmployed']
ExtSource3= prediction_data_info.loc[IdClient, 'ExtSource3']
DaysBirth= prediction_data_info.loc[IdClient, 'DaysBirth']
ApartmentAvg= prediction_data_info.loc[IdClient, 'ApartmentAvg']
DaysRegistration= prediction_data_info.loc[IdClient, 'DaysRegistration']
CreditTypeMicroloan= prediction_data_info.loc[IdClient, 'CreditTypeMicroloan']
PrevDaysDecisionMin = prediction_data_info.loc[IdClient, 'PrevDaysDecisionMin']



Gender = prediction_data_info.loc[IdClient, 'CODE_GENDER']
Age = prediction_data_info.loc[IdClient, 'AGE']
Revenue = prediction_data_info.loc[IdClient, 'AMT_INCOME_TOTAL']
RevenueType = prediction_data_info.loc[IdClient, 'NAME_INCOME_TYPE']

Annuity = prediction_data_info.loc[IdClient, 'AMT_ANNUITY']
ContractType = prediction_data_info.loc[IdClient, 'NAME_CONTRACT_TYPE']
PreviousApp = prediction_data_info.loc[IdClient, 'PREVIOUS_APPLICATIONS']


inputs = {
        'DaysEmployed': DaysEmployed,
        'ExtSource3': ExtSource3,
        'DaysBirth': DaysBirth,
        'ApartmentAvg': ApartmentAvg,
        'DaysRegistration': DaysRegistration,
        'CreditTypeMicroloan': CreditTypeMicroloan,
        'PrevDaysDecisionMin': PrevDaysDecisionMin,
        'IdClient': IdClient
        }


@st.cache
def get_shap_data():

    # loading the trained model for compute shap value
    pickle_in = open('common_files/mlflow_model_0/model.pkl', 'rb') # load no-threshold model for shap computation
    model = pickle.load(pickle_in)

    train_sample = pd.read_csv('common_files/features_train.csv')
    X = train_sample.drop(columns=['IdClient', 'Target'])
    y = train_sample['Target']
    model.fit(X,y)

    inputs_dict = {k:[v] for k,v in inputs.items()} # dictionary for a dataframe
    pd_inputs = pd.DataFrame(inputs_dict).drop(columns='IdClient') # ensure only input features, Id client/Target excluded

    explainer = shap.TreeExplainer(model['Classifier'])
    sv_explainer = explainer(pd_inputs) # apply treeExplainer on test instance
    return explainer, sv_explainer, pd_inputs


def main():
    #fast_uri = "https://fast-api-app-deployment.herokuapp.com"
    fast_uri = "http://127.0.0.1:8000/predict"

    # sidebar
    st.sidebar.write('Client: ', IdClient)
    #if Gender == 'F':
    #    Gender == 'Female'
    #    st.sidebar.write('Gender:', 'Female')
    #else:
    #    Gender == 'Male'
    #    st.sidebar.write('Gender:', 'Male')
    st.sidebar.write('Gender:', Gender)
    st.sidebar.write('Age:', Age)
    st.sidebar.write('Revenu:', Revenue)
    st.sidebar.write('Revenu Type:', RevenueType)

    # main page
    #PageTitle='<p style="padding-top:0; font-family:courier; font-size: 42px; font-weight: bold; text-align: center;">Client Credit Default Prediction Dashbord</p>'
    #st.markdown(PageTitle, unsafe_allow_html=True)

   # #column1, column2, column3
    with open('style.css') as w:
        st.markdown(f'<style>{w.read()}</style>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    #value1 ='<p style="font-weight: bold; font-family:courier; font-size: 24px; text-align: center;">f'{PreviousApp}'</p>'
    #col1.subheader(f'{PreviousApp}')
    col1.markdown(f"""
                  <p style="font-weight: bold; font-family:courier; font-size: 24px; text-align: center;">{PreviousApp}</p>
                    """, unsafe_allow_html=True)
    col1.markdown('Previous Applications', unsafe_allow_html=True)
    #col2.subheader(f'{Annuity}')
    col2.markdown(f"""
                  <p style="font-weight: bold; font-family:courier; font-size: 24px; text-align: center;">{Annuity}</p>
                    """, unsafe_allow_html=True)
    col2.markdown('Annuity Amount', unsafe_allow_html=True)
    #col3.subheader(f'{ContractType}')
    col3.markdown(f"""
                  <p style="font-weight: bold; font-family:courier; font-size: 24px; text-align: center;">{ContractType}</p>
                    """, unsafe_allow_html=True)
    col3.markdown('Contract Type', unsafe_allow_html=True)


    #if st.sidebar.selectbox('Client ID to predict', [el for el in list(prediction_data.index)]):
    if st.sidebar.button('Predict'):
        # convert input dictionnary to json string
        inputs_json = json.dumps(inputs)

        # send api request
        headers = {"Content-Type": "application/json"}

        response = requests.post(url=fast_uri, data=inputs_json, headers= headers)

        if response.status_code != 200:
            raise Exception(
                "Request failed with status {}, {}".format(response.status_code, response.text))

        # convert string to float
        proba = np.array(response.text, dtype=float).item() # got np.ndarray, use item to get the first item which should return a float value
        #proba_flatten = proba.flatten()
        proba = round(proba, 2)
        st.sidebar.write(f'The default probability predicted is {proba}')

        # graphics plotting
        col1, col2 = st.columns([1,3])

        # plot plotly gauge chart on predicted value ---------------------------------------------------------------------------------------


        with col1:
            head1 ='<p style="margin-top: 25px; font-weight: bold; font-family:courier; font-size: 18px; text-align: center;">Predicted Credit Default Probability</p>'
            st.markdown(head1, unsafe_allow_html=True)
            layout = go.Layout(autosize=False, width=360, height=150,
                              margin= go.layout.Margin(t=0, b=25))
            fig = go.Figure(layout=layout)
            fig.add_trace(go.Indicator(value = proba,
                                       number = {'valueformat': '.2%', 'font':{'size':24}},
                                       gauge = {'axis': {'range': [0, 1],'dtick': 0.76},
                                                'bar': {'color': 'blue'},
                                                'steps': [
                                                    {'range': [0, 0.76], 'color': 'red'},
                                                        {'range': [0.76, 1], 'color': 'green'}
                                                ]}
                                                    ))

            fig.update_layout(
                              grid = {'rows': 1, 'columns': 1},
                              template = {'data' : {'indicator': [{
                                                                   'mode' : "number+gauge"}]}})
                                                                   #'domain': {'x': [0, 1], 'y': [0, 1]}


            col1.plotly_chart(fig, use_contrainer_width=True)

        #graphic 2
        with col2:
            head2 ='<p style="margin-top: 25px; font-weight: bold; font-family:courier; font-size: 18px; text-align: center;">Revenue Amounts by Revenue Types</p>'
            st.markdown(head2, unsafe_allow_html=True)

            #def revenu_plot():
            fig, ax= plt.subplots(constrained_layout=True)
            infog= client_info.loc[:, ['AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE']].groupby('NAME_INCOME_TYPE').mean()
            infog.plot(kind= 'bar', ax=ax, label='Type', color='b')

            ax.set_xlabel('Client Income Types')
            ax.set_ylabel('Total Revenus')
            ax.set_xticks(range(0,len(infog.index)), list(infog.index), rotation=30)
            ax.set_ylim(0, 600000)
            ax.legend(['Type'])

            #add client
            tc = Revenue
            ax.axhline(tc, color='orange', label='Client')

            #get the two labels as legends
            handles,_ = ax.get_legend_handles_labels()
            plt.legend(handles= handles, labels = ['Client', 'Type'])
            fig.set_tight_layout(True) # to prevent axis labels overlapping
            #ax.figure.savefig(plt.gcf())
            #plt.savefig()

            #return  plt.gcf()#plotly.io.to_image(plt.gcf(), engine='kaleido', format='jpg')
            #plotly.io.to_image(fig, engine='kaleido', validate=False)
            #fig= revenu_plot()
            #st.pyplot(plt.gcf(), use_column_width=True) #return oversize figure

            #st.pyplot(fig)
            st_shap(plt.gcf(), height=280, width=1000)
            #st.image(fig)
            #col2.plotly_chart(fig, use_contrainer_width=True)

        # plot shap value -------------------------------------------------------------------------------------------------------
        # Shap Waterfall_plot on class 0
        head3 ='<p style="margin-top: 25px; font-weight: bold; font-family:courier; font-size: 18px; text-align: center;">Default Probability Influence Factors</p>'
        st.markdown(head3, unsafe_allow_html=True)

        explainer, sv_explainer, pd_inputs = get_shap_data()

        shap.initjs()
        fig = plt.figure()
        #shap.waterfall_plot(shap.Explanation(sv_explainer[0,:,0], sv_explainer.base_values[:,0], pd_inputs, feature_names=pd_inputs.columns), show=False)
        st_shap(shap.waterfall_plot(shap.Explanation(sv_explainer[0,:,0], sv_explainer.base_values[:,0], pd_inputs, feature_names=pd_inputs.columns)), height=390, width=1100)
        #plt.gcf().set_size_inches(8,0.8)
        #plt.tight_layout()
        #st.pyplot(plt.gcf())

        # hide streamlit style
        hide_st_style = """
                    <style>
                    MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    header {visibility: hidden;}
                    </style>
                    """
        st.markdown(hide_st_style, unsafe_allow_html= True)

if __name__ == '__main__':
    main()
