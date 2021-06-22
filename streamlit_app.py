
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock
plt.style.use('default')

pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

def predict_heart(age,sex,chest_pain_type,resting_blood_pressure,cholesterol,st_slope,rest_ecg,max_heart_rate_achieved,exercise_induced_angina, st_depression, thalassemia):
    prediction=classifier.predict([[age,sex,chest_pain_type,resting_blood_pressure,cholesterol,st_slope,rest_ecg,max_heart_rate_achieved,exercise_induced_angina, st_depression, thalassemia]])
    print(prediction)
    return prediction


def main():

    selected_box = st.sidebar.selectbox("Health",["Home", "Heart Disease Prediction"])

    if selected_box == "Home":
        home()

    if selected_box == "Heart Disease Prediction":
        heart()

    
def home():
    
    row1_spacer1, row1_1, row1_spacer2, row1_2, row1_spacer3 = st.beta_columns((.1, 2, 1, 1, .1))
    row1_1.title('Health Care')
    st.image('hlth.webp')
    st.markdown(""" ## Heart Disease Prediction

Heart disease refers to a variety of conditions that affect the heart — from infections to genetic defects and blood-vessel diseases.t disease refers to a variety of conditions that affect the heart — from infections to genetic defects and blood-vessel diseases. Heart disease is responsible for most deaths worldwide for both men and women of all races.

“Heart disease” isn’t a single disease. Rather, the term encompasses a variety of diseases that affect your cardiovascular system. With modern lifestyle people are more prone to heart diseases. 

## Some Facts about Heart disease

An estimated 17.9 million people died from CVDs in 2019, representing 32% of all global deaths. Of these deaths, 85% were due to heart attack and stroke. Along with being the leading cause of death, cardiovascular disease (CVD) -- especially ischemic heart disease and stroke -- is a major cause of disability and rising health care costs.

## Risk Factors

The most important behavioural risk factors of heart disease and stroke are unhealthy diet, physical inactivity, tobacco use and harmful use of alcohol. The effects of behavioural risk factors may show up in individuals as raised blood pressure, raised blood glucose, raised blood lipids, and overweight and obesity. These “intermediate risks factors” can be measured in primary care facilities and indicate an increased risk of heart attack, stroke, heart failure and other complications.


## Prevention

Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol.
It is important to detect cardiovascular disease as early as possible so that management with counselling and medicines can begin.

## About Health Care App

#### Health Care App can be used to check if you have high risk of heart disease or not by the help of which early detection and prevention of heart disease can be done by scheduling regular check-ups with your health care provider and getting advice on controlling your risk factors. 



        """, True)
    

    
    with row1_2:
        st.write('')
        row1_2.subheader('A web app by [Mansi Maurya](https://github.com/mansi75)')

     
   
    

def heart():
    st.markdown(""" # Heart Disease Prediction

We can see from data that number of male patients with heart disease are more as compared to female patients. With increase of cholesterol there is a higher risk of a heart disease and higher blood pressure. Higher blood sugar leads to higher risk of heart disease.
People with Non-Anginal Chest Pain have more chances of a heart disease than other chest pain type. People with cholesterol not only have chances of heart disease but also have higher heart rate.


        """, True)
    





    df = pd.read_csv("heart.csv")
    fig = Figure(figsize=(18, 16))
    ((ax1, ax2),(ax3, ax4), (ax5, ax6)) = fig.subplots(3,2)
    #((ax4, ax3),(ax1, ax2)) = fig.subplots(2,2)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig.subplots_adjust(left=0.9,
                    bottom=1.9, 
                    right=1, 
                    top=2, 
                    wspace= 14, 
                    hspace=10)
        
    
    sns.countplot(x='sex', hue = "target", data=df, palette = ["brown", "red"], ax = ax1)
    ax1.legend()
    ax1.set_xlabel('Sex (0 = female, 1= male)', fontsize=25)
    ax1.set_ylabel('count', fontsize=25)
    ax1.grid(zorder=0,alpha=.2)
    
    sns.scatterplot(x="chol", y = "trestbps", hue="target", data=df, palette= ["brown","red"], ax = ax2)
    ax2.legend()
    ax2.set_xlabel('cholesterol', fontsize=25)
    ax2.set_ylabel('Rest Blood Pressure', fontsize=25)
    ax2.grid(zorder=0,alpha=.2)
    
    
    
    st.set_option('deprecation.showPyplotGlobalUse', False)

    ''
    
    sns.countplot(x="fbs", hue="target", data=df, palette = ["brown", "red"], ax = ax3)
    ax3.legend()
    ax3.set_xlabel('Fasting Blood Sugar(1 = true; 0 = false)', fontsize=25)
    ax3.set_ylabel('count', fontsize=25)
    ax3.grid(zorder=0,alpha=.2)

    sns.scatterplot(data=df, x="age", y="thalach", hue="target", palette=['brown','red'], ax = ax4)
    ax4.legend()
    ax4.set_xlabel('Age', fontsize=25)
    ax4.set_ylabel('Maximum Heart Rate', fontsize=25)
    ax4.grid(zorder=0,alpha=.2)
    
    
    sns.countplot(x="cp", hue="target", data=df, palette = ["brown", "red"], ax = ax5)
    ax5.legend()
    ax5.set_xlabel('Heart Disease According To Chest Pain Type', fontsize=25)
    ax5.set_ylabel('count', fontsize=25)
    ax5.grid(zorder=0,alpha=.2)


    sns.scatterplot(data=df, x='chol', y = "thalach", hue = "target", palette = ['brown','red'], ax = ax6)
    ax6.legend()
    ax6.set_xlabel('cholesterol', fontsize=25)
    ax6.set_ylabel('Maximum Heart Rate', fontsize=25)
    ax6.grid(zorder=0,alpha=.2)

    


    
    with _lock:
        #fig.suptitle('Heart disease')
        fig.tight_layout(rect=[0, 0.05, 1.0, 1.75])
        st.pyplot(fig)

    st.write("Fill the below data to check if you have heart disease or not.")
    sex = st.text_input("sex (0 if female and 1 if male)")
    age = st.text_input("age")
    chest_pain_type = st.text_input("chest pain type(0 = Typical Angina,1 =Atypical Angina,2 = Non-Anginal Pain,4 = Asymptomatic)")
    resting_blood_pressure = st.text_input("resting blood pressure")
    cholesterol = st.text_input("cholesterol")
    st_slope = st.text_input("st_slope(0 = unslopping, 1= flat, 2 = downslopping)")
    rest_ecg = st.text_input("rest_ecg(0 = Normal, 1 = ST-T Wave Abnormality, 2 = Left Ventricular Hypertrophy)")
    max_heart_rate_achieved = st.text_input("maximum heart rate achieved")
    exercise_induced_angina   = st.text_input("exercise induced angina(1 = Yes, 0 = No)")
    st_depression = st.text_input("st_depression")
    thalassemia = st.text_input("thalassemia(0 = Normal, 1= Fixed Defect, 2 = Reversable Defect)")
    result=""
    if st.button("Predict"):
        result=predict_heart(age,sex,chest_pain_type,resting_blood_pressure,cholesterol, st_slope, rest_ecg,max_heart_rate_achieved,exercise_induced_angina,st_depression, thalassemia)
        if(result == 0):
            st.success('You are healthy, you do not have a chance of heart disease')
        elif(result == 1):
            st.success('You have a chance of heart disease, kindly visit cardiologist')

    
    

    
    

if __name__ == '__main__':
    main()

