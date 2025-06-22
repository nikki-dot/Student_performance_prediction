import streamlit as st
import pickle
import numpy as np

st.title('Student Exam pass prediction')

logistic = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('scaler.pkl','rb'))

study = st.number_input('Hours_studied',min_value=0)
attendance = st.number_input('Attendance_percentage',min_value=0)
sleep = st.number_input('Sleep_hours',min_value=0)
assignments = st.number_input('Assignments_completed',min_value=0)
st.text_input('Give Feedback',max_chars=50)

input_data = ([study, attendance, sleep, assignments])
input_data_as_numpy = np.array(input_data)
input_data_reshape = input_data_as_numpy.reshape(1,-1)
std_sc = sc.transform(input_data_reshape)
print(std_sc)
predicted_model = logistic.predict(std_sc)
print(predicted_model)
if st.button('Submit'):
    if(predicted_model)[0]==0:
        st.header("Fail")
    else:
        st.header("Pass")