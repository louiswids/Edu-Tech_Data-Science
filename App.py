# Libraries
import streamlit as st
import numpy as np
import pandas as pd
import joblib


# Categorical Encoders
encoder_Application_mode = joblib.load('model/encoder_Application_mode.joblib')
encoder_Course = joblib.load('model/encoder_Course.joblib')
encoder_Daytime_evening_attendance = joblib.load('model/encoder_Daytime_evening_attendance.joblib')
encoder_Debtor = joblib.load('model/encoder_Debtor.joblib')
encoder_Displaced = joblib.load('model/encoder_Displaced.joblib')
encoder_Gender = joblib.load('model/encoder_Gender.joblib')
encoder_Scholarship_holder = joblib.load('model/encoder_Scholarship_holder.joblib')
encoder_Tuition_fees_up_to_date = joblib.load('model/encoder_Tuition_fees_up_to_date.joblib')

# Numerical Scaler
scaler_Admission_grade = joblib.load('model/scaler_Admission_grade.joblib')
scaler_Age_at_enrollment = joblib.load('model/scaler_Age_at_enrollment.joblib')
scaler_Curricular_units_1st_sem_approved = joblib.load('model/scaler_Curricular_units_1st_sem_approved.joblib')
scaler_Curricular_units_1st_sem_enrolled = joblib.load('model/scaler_Curricular_units_1st_sem_enrolled.joblib')
scaler_Curricular_units_1st_sem_evaluations = joblib.load('model/scaler_Curricular_units_1st_sem_evaluations.joblib')
scaler_Curricular_units_1st_sem_grade = joblib.load('model/scaler_Curricular_units_1st_sem_grade.joblib')
scaler_Curricular_units_2nd_sem_approved = joblib.load('model/scaler_Curricular_units_2nd_sem_approved.joblib')
scaler_Curricular_units_2nd_sem_enrolled = joblib.load('model/scaler_Curricular_units_2nd_sem_enrolled.joblib')
scaler_Curricular_units_2nd_sem_evaluations = joblib.load('model/scaler_Curricular_units_2nd_sem_evaluations.joblib')
scaler_Curricular_units_2nd_sem_grade = joblib.load('model/scaler_Curricular_units_2nd_sem_grade.joblib')
scaler_Previous_qualification_grade = joblib.load('model/scaler_Previous_qualification_grade.joblib')

# PCA, Modeling, and Result Encoding
pca_1 = joblib.load('model/pca_1.joblib')
pca_2 = joblib.load('model/pca_2.joblib')
gb_model = joblib.load('model/gb_model.joblib')
encoder_target = joblib.load('model/encoder_target.joblib')

pca_columns_1 = ['Curricular_units_1st_sem_enrolled', 
                 'Curricular_units_1st_sem_approved', 
                 'Curricular_units_1st_sem_grade', 
                 'Curricular_units_1st_sem_evaluations',
                 'Curricular_units_2nd_sem_enrolled', 
                 'Curricular_units_2nd_sem_approved', 
                 'Curricular_units_2nd_sem_grade', 
                 'Curricular_units_2nd_sem_evaluations']
pca_columns_2 = ['Previous_qualification_grade',
                 'Admission_grade',
                 'Age_at_enrollment',
                 'Application_mode']

training_features = ['Course', 
                     'Daytime_evening_attendance', 
                     'Displaced', 
                     'Debtor',
                     'Tuition_fees_up_to_date', 
                     'Gender', 
                     'Scholarship_holder', 
                     'pc1_1',
                     'pc1_2', 
                     'pc1_3', 
                     'pc1_4', 
                     'pc2_1']

def data_preprocessing(data):
    """Preprocessing data
 
    Args:
        data (Pandas DataFrame): Dataframe that contain all the data to make prediction 
        
    return:
        Pandas DataFrame: Dataframe that contain all the preprocessed data
    """
    data = data.copy()
    df = pd.DataFrame()

    # categorical encoding
    df["Application_mode"] = encoder_Application_mode.transform(data["Application_mode"])
    df["Course"] = encoder_Course.transform(data["Course"])
    df["Daytime_evening_attendance"] = encoder_Daytime_evening_attendance.transform(data["Daytime_evening_attendance"])
    df["Debtor"] = encoder_Debtor.transform(data["Debtor"])
    df["Displaced"] = encoder_Displaced.transform(data["Displaced"])
    df["Gender"] = encoder_Gender.transform(data["Gender"])
    df["Scholarship_holder"] = encoder_Scholarship_holder.transform(data["Scholarship_holder"])
    # df["target"] = encoder_target.transform(data["target"])
    df["Tuition_fees_up_to_date"] = encoder_Tuition_fees_up_to_date.transform(data["Tuition_fees_up_to_date"])

    #PCA 1
    data['Curricular_units_1st_sem_enrolled'] = scaler_Curricular_units_1st_sem_enrolled.transform(np.asarray(data["Curricular_units_1st_sem_enrolled"]).reshape(-1, 1))[0]
    data['Curricular_units_1st_sem_approved'] = scaler_Curricular_units_1st_sem_approved.transform(np.asarray(data["Curricular_units_1st_sem_approved"]).reshape(-1, 1))[0]
    data['Curricular_units_1st_sem_grade'] = scaler_Curricular_units_1st_sem_grade.transform(np.asarray(data["Curricular_units_1st_sem_grade"]).reshape(-1, 1))[0]
    data['Curricular_units_1st_sem_evaluations'] = scaler_Curricular_units_1st_sem_evaluations.transform(np.asarray(data["Curricular_units_1st_sem_evaluations"]).reshape(-1, 1))[0]
    data['Curricular_units_2nd_sem_enrolled'] = scaler_Curricular_units_2nd_sem_enrolled.transform(np.asarray(data["Curricular_units_2nd_sem_enrolled"]).reshape(-1, 1))[0]
    data['Curricular_units_2nd_sem_approved'] = scaler_Curricular_units_2nd_sem_approved.transform(np.asarray(data["Curricular_units_2nd_sem_approved"]).reshape(-1, 1))[0]
    data['Curricular_units_2nd_sem_grade'] = scaler_Curricular_units_2nd_sem_grade.transform(np.asarray(data["Curricular_units_2nd_sem_grade"]).reshape(-1, 1))[0]
    data['Curricular_units_2nd_sem_evaluations'] = scaler_Curricular_units_2nd_sem_evaluations.transform(np.asarray(data["Curricular_units_2nd_sem_evaluations"]).reshape(-1, 1))[0]
    df[["pc1_1", "pc1_2", "pc1_3", "pc1_4"]] = pca_1.transform(data[pca_columns_1])


    
    # PCA 2
    data['Previous_qualification_grade'] = scaler_Previous_qualification_grade.transform(np.asarray(data["Previous_qualification_grade"]).reshape(-1, 1))[0]
    data['Admission_grade'] = scaler_Admission_grade.transform(np.asarray(data["Admission_grade"]).reshape(-1, 1))[0]
    data["Age_at_enrollment"] = scaler_Age_at_enrollment.transform(np.asarray(data["Age_at_enrollment"]).reshape(-1, 1))[0]
    data["Application_mode"] = np.asarray(df["Application_mode"]).reshape(-1, 1)[0]
    df[["pc2_1"]] = pca_2.transform(data[pca_columns_2])

    df = df[training_features]
    
    return df

def prediction(data):
    """Making prediction
 
    Args:
        data (Pandas DataFrame): Dataframe that contain all the preprocessed data
 
    Returns:
        str: Prediction result (Good, Standard, or Poor)
    """
    result = gb_model.predict(data)
    final_result = encoder_target.inverse_transform(result)[0]
    return final_result


col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://media.licdn.com/dms/image/D5603AQG3r7y4FIt2IQ/profile-displayphoto-shrink_200_200/0/1712126841723?e=2147483647&v=beta&t=U_iDrfRoLYs9gzOFdcTuXuxIgl88iRRf0b8onRYVrk0", width=100, )
with col2:
    st.header('Student Performance Outcome App (Prototype)')
    st.write('By: Louis Widi Anandaputra')
# Input fields
data = pd.DataFrame()
# Create Streamlit form for user input
with st.form(key='dropout_form'):
    col1, col2 = st.columns(2)
    Age_at_enrollment = st.number_input('Age at Enrollment', min_value=0, value=30)
    data["Age_at_enrollment"] = [Age_at_enrollment]

    Previous_qualification_grade = st.number_input('Previous Qualification Grade', min_value=0, value=0)
    data["Previous_qualification_grade"] = [Previous_qualification_grade]

    Admission_grade = st.number_input('Admission Grade', min_value=0, max_value=200, value=0)
    data["Admission_grade"] = [Admission_grade]

    Application_mode = st.selectbox('Application Mode', options=encoder_Application_mode.classes_, index=14)
    data["Application_mode"] = [Application_mode]
    
    Course = st.selectbox('Course', options=encoder_Course.classes_, index=10)
    data["Course"] = [Course]
    
    with col1:
        Curricular_units_1st_sem_enrolled = st.number_input('Curricular Units 1st Sem Enrolled', min_value=0, value=0)
        data["Curricular_units_1st_sem_enrolled"] = [Curricular_units_1st_sem_enrolled]
        
        Curricular_units_1st_sem_approved = st.number_input('Curricular Units 1st Sem Approved', min_value=0, value=0)
        data["Curricular_units_1st_sem_approved"] = [Curricular_units_1st_sem_approved]
        
        Curricular_units_1st_sem_evaluations = st.number_input('Curricular Units 1st Sem Evaluations', min_value=0, value=0)
        data["Curricular_units_1st_sem_evaluations"] = [Curricular_units_1st_sem_evaluations]
        
        Curricular_units_1st_sem_grade = st.number_input('Curricular Units 1st Sem Grade', min_value=0, value=0)
        data["Curricular_units_1st_sem_grade"] = [Curricular_units_1st_sem_grade]

    with col2:
        Curricular_units_2nd_sem_enrolled = st.number_input('Curricular Units 2nd Sem Enrolled', min_value=0, value=0)
        data["Curricular_units_2nd_sem_enrolled"] = [Curricular_units_2nd_sem_enrolled]

        Curricular_units_2nd_sem_approved = st.number_input('Curricular Units 2nd Sem Approved', min_value=0, value=0)
        data["Curricular_units_2nd_sem_approved"] = [Curricular_units_2nd_sem_approved]

        Curricular_units_2nd_sem_grade = st.number_input('Curricular Units 2nd Sem Grade', min_value=0, value=0)
        data["Curricular_units_2nd_sem_grade"] = [Curricular_units_2nd_sem_grade]
        
        Curricular_units_2nd_sem_evaluations = st.number_input('Curricular Units 2nd Sem Evaluations', min_value=0, value=0)
        data["Curricular_units_2nd_sem_evaluations"] = [Curricular_units_2nd_sem_evaluations]
        
    Daytime = st.selectbox('Daytime_evening_attendance', options=encoder_Daytime_evening_attendance.classes_, index=0)
    data['Daytime_evening_attendance'] = [Daytime]
    
    Displaced = st.selectbox('Displaced', options=encoder_Displaced.classes_, index=0)
    data['Displaced'] = [Displaced]
    
    Debtor = st.selectbox('Debtor', options=encoder_Debtor.classes_, index=0)
    data['Debtor'] = [Debtor]
    
    Tuition = st.selectbox('Tuition_fees_up_to_date', options=encoder_Tuition_fees_up_to_date.classes_, index=0)
    data['Tuition_fees_up_to_date'] = [Tuition]

    Gender = st.selectbox('Gender', options=encoder_Gender.classes_, index=0)
    data['Gender'] = [Gender]

    Scholarship = st.selectbox('Scholarship_holder', options=encoder_Scholarship_holder.classes_, index=0)
    data['Scholarship_holder'] = [Scholarship]
    
    submit_button = st.form_submit_button(label='Predict Status')

with st.expander("View the Raw Data"):
    st.dataframe(data=data, width=800, height=10)
    
if submit_button:
    new_data = data_preprocessing(data)
    with st.expander("View the Preprocessed Data"):
        st.dataframe(data=new_data, width=800, height=10)
    st.write("Status Prediction: {}".format(prediction(new_data)))

st.caption('© 2024 - Louis Widi Anandaputra')


