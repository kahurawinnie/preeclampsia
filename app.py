import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from PIL import Image

# Set up the sidebar
# Logo image
image = Image.open('files/muk.png')
st.sidebar.image(image, width=300)

# Main content
st.write("""
### PREDICTION OF RISK OF PRE-ECLAMPSIA AMONG PREGNANT WOMEN AT NAGURU HOSPITAL
         
##### Project by WINFRED KAHURA
##### Registration: 2020/HD07/19481U 
""")

# Project Description and Column Key
st.write("""
#### Objective
To develop, train, and test a machine-learning model to predict women’s risk of pre-eclampsia, to be used by healthcare workers.
""")

# Collects user input features into dataframe
def user_input_features():
    age        = st.sidebar.slider('Age (Years)', 16, 60, 36)
    dipstick   = st.sidebar.selectbox('Dipstick Proteinuria', ('Positive', 'Negative'))
    systolic   = st.sidebar.slider('Systolic BP (mmHg)', 90, 150, 120)
    diastolic  = st.sidebar.slider('Diastolic BP (mmHg)', 60, 100, 80)
    apain      = st.sidebar.selectbox('Abdominal Pain', ('No', 'Yes'))
    headache   = st.sidebar.selectbox('Headache/Visual Disturbances', ('No', 'Yes'))
    cpain      = st.sidebar.selectbox('Chest Pain', ('No', 'Yes'))
    parity     = st.sidebar.slider('Parity', 1, 10, 3)
    bleeding   = st.sidebar.selectbox('Vaginal Bleeding', ('Yes', 'No'))
    
    data = {'Age': age,
            'Dipstick Proteinuria': dipstick,
            'systolic': systolic,
            'diastolic': diastolic,
            'Abdominal Pain': apain,
            'Headache/Visual Disturbances': headache,
            'Chest Pain': cpain,
            'Parity': parity,
            'Vaginal Bleeding': bleeding}
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Encode categorical variables using map()
dfn = input_df.copy()  # Keep the original input for display

# Map binary categorical values to 0 and 1
encoding_dict = {
    'Dipstick Proteinuria': {'Positive': 1, 'Negative': 0},
    'Abdominal Pain': {'Yes': 1, 'No': 0},
    'Headache/Visual Disturbances': {'Yes': 1, 'No': 0},
    'Chest Pain': {'Yes': 1, 'No': 0},
    'Vaginal Bleeding': {'Yes': 1, 'No': 0}
}

for col, mapping in encoding_dict.items():
    input_df[col] = input_df[col].map(mapping)

# Displays the user input features before encoding
st.subheader('User Input features')
st.write(dfn)


# Load the saved classification model
load_clf = load('files/preeclampsia.joblib')

# Apply model to make predictions
prediction = load_clf.predict(input_df)
prediction_proba = load_clf.predict_proba(input_df)

# Instead of using prediction probability, calculate the prediction score
# The prediction score is the highest probability of the two possible outcomes
prediction_score = np.max(prediction_proba)

# Prediction results
st.subheader('Prediction')
prediction_text = np.array(['No Pre-eclampsia', 'Pre-eclampsia'])
st.write(prediction_text[prediction])

st.subheader('Prediction Confidence Score')
st.write(f"The model's confidence in this prediction is: **{prediction_score * 100:.2f}%**")

# Dynamic narrative for non-technical users
st.subheader('Explanation of Results')

if prediction[0] == 1:
    st.markdown(f"""
    **Risk of Pre-eclampsia:** Based on the input values, there is a **high risk** that the woman may develop pre-eclampsia. 
    The model is **{prediction_score * 100:.2f}%** confident in this prediction. This means that pre-eclampsia is likely. 
    
    It is important to take this prediction seriously and consult healthcare professionals for further assessments and treatment.
    """)
else:
    st.markdown(f"""
    **Risk of Pre-eclampsia:** The model predicts that the woman is **unlikely** to develop pre-eclampsia. 
    The model is **{prediction_score * 100:.2f}%** confident in this prediction. This suggests a lower risk, but regular monitoring and check-ups are still recommended.
    
    Although the risk appears low, it’s important to consider all factors in consultation with a healthcare provider.
    """)

