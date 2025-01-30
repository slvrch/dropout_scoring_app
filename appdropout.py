import joblib
import pandas as pd
import numpy as np
import streamlit as st

#Load model
model = joblib.load("model/rdf_model.joblib")
result_target = joblib.load("model/encoder_target.joblib")
encoder_Application_mode = joblib.load("model/encoder_Application_mode.joblib")
encoder_Course = joblib.load("model/encoder_Course.joblib")
encoder_Previous_qualification = joblib.load("model/encoder_Previous_qualification.joblib")
encoder_Mothers_qualification = joblib.load("model/encoder_Mothers_qualification.joblib")
encoder_Fathers_qualification = joblib.load("model/encoder_Fathers_qualification.joblib")
encoder_Mothers_occupation = joblib.load("model/encoder_Mothers_occupation.joblib")
encoder_Fathers_occupation = joblib.load("model/encoder_Fathers_occupation.joblib")
encoder_Displaced = joblib.load("model/encoder_Displaced.joblib")
encoder_Debtor = joblib.load("model/encoder_Debtor.joblib")
encoder_Tuition_fees_up_to_date = joblib.load("model/encoder_Tuition_fees_up_to_date.joblib")
encoder_Gender = joblib.load("model/encoder_Gender.joblib")
encoder_Scholarship_holder = joblib.load("model/encoder_Scholarship_holder.joblib")

pca_1 = joblib.load("model/pca_1.joblib")
pca_2 = joblib.load("model/pca_2.joblib")

scaler_Admission_grade = joblib.load("model/scaler_Admission_grade.joblib")
scaler_Age_at_enrollment = joblib.load("model/scaler_Age_at_enrollment.joblib")
scaler_Application_order = joblib.load("model/scaler_Application_order.joblib")
scaler_Curricular_units_1st_sem_approved = joblib.load("model/scaler_Curricular_units_1st_sem_approved.joblib")
scaler_Curricular_units_1st_sem_credited = joblib.load("model/scaler_Curricular_units_1st_sem_credited.joblib")
scaler_Curricular_units_1st_sem_enrolled = joblib.load("model/scaler_Curricular_units_1st_sem_enrolled.joblib")
scaler_Curricular_units_1st_sem_evaluations = joblib.load("model/scaler_Curricular_units_1st_sem_evaluations.joblib")
scaler_Curricular_units_1st_sem_grade = joblib.load("model/scaler_Curricular_units_1st_sem_grade.joblib")
scaler_Curricular_units_2nd_sem_approved = joblib.load("model/scaler_Curricular_units_2nd_sem_approved.joblib")
scaler_Curricular_units_2nd_sem_credited = joblib.load("model/scaler_Curricular_units_2nd_sem_credited.joblib")
scaler_Curricular_units_2nd_sem_enrolled = joblib.load("model/scaler_Curricular_units_2nd_sem_enrolled.joblib")
scaler_Curricular_units_2nd_sem_evaluations = joblib.load("model/scaler_Curricular_units_2nd_sem_evaluations.joblib")
scaler_Curricular_units_2nd_sem_grade = joblib.load("model/scaler_Curricular_units_2nd_sem_grade.joblib")
scaler_GDP = joblib.load("model/scaler_GDP.joblib")
scaler_Inflation_rate = joblib.load("model/scaler_Inflation_rate.joblib")
scaler_Previous_qualification_grade = joblib.load("model/scaler_Previous_qualification_grade.joblib")
scaler_Unemployment_rate = joblib.load("model/scaler_Unemployment_rate.joblib")

pca_numerical_columns_1 = [
    'Curricular_units_1st_sem_credited',
    'Curricular_units_1st_sem_enrolled',
    'Curricular_units_1st_sem_approved',
    'Curricular_units_2nd_sem_credited',
    'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade',
    'Curricular_units_1st_sem_grade',
    'Curricular_units_1st_sem_evaluations'
]

pca_numerical_columns_2 = [
    'Curricular_units_2nd_sem_evaluations',
    'Unemployment_rate',
    'Inflation_rate',
    'Application_order',
    'Age_at_enrollment',
    'GDP'
]

def data_preprocessing(data):
    """Preprocessing data

    Args:
        data (Pandas DataFrame): Dataframe that contain all the data to make prediction

    return:
        Pandas DataFrame: Dataframe that contain all the preprocessed data
    """
    data = data.copy()
    df = pd.DataFrame()

    # Encoding dan Scaling
    df["Application_mode"] = encoder_Application_mode.transform(data["Application_mode"])
    df["Course"] = encoder_Course.transform(data["Course"])
    df["Previous_qualification"] = encoder_Previous_qualification.transform(data["Previous_qualification"])
    df["Previous_qualification_grade"] = scaler_Previous_qualification_grade.transform(
        np.asarray(data["Previous_qualification_grade"]).reshape(-1, 1)
    )[:, 0]
    df["Mothers_qualification"] = encoder_Mothers_qualification.transform(data["Mothers_qualification"])
    df["Fathers_qualification"] = encoder_Fathers_qualification.transform(data["Fathers_qualification"])
    df["Mothers_occupation"] = encoder_Mothers_occupation.transform(data["Mothers_occupation"])
    df["Fathers_occupation"] = encoder_Fathers_occupation.transform(data["Fathers_occupation"])
    df["Admission_grade"] = scaler_Admission_grade.transform(np.asarray(data["Admission_grade"]).reshape(-1, 1))[:, 0]
    df["Displaced"] = encoder_Displaced.transform(data["Displaced"])
    df["Debtor"] = encoder_Debtor.transform(data["Debtor"])
    df["Tuition_fees_up_to_date"] = encoder_Tuition_fees_up_to_date.transform(data["Tuition_fees_up_to_date"])
    df["Gender"] = encoder_Gender.transform(data["Gender"])
    df["Scholarship_holder"] = encoder_Scholarship_holder.transform(data["Scholarship_holder"])

    # PCA Transformations
    df[["pc1_1", "pc1_2", "pc1_3"]] = pca_1.transform(data[pca_numerical_columns_1])
    df[["pc2_1", "pc2_2", "pc2_3"]] = pca_2.transform(data[pca_numerical_columns_2])

    return df

def prediction(data):
    """Making prediction
 
    Args:
        data (Pandas DataFrame): Dataframe that contain all the preprocessed data
 
    Returns:
        str: Prediction result (Good, Standard, or Poor)
    """
    result = model.predict(data)
    final_result = result_target.inverse_transform(result)[0]
    return final_result


# Tampilan utama
st.title("Aplikasi Prediksi Dropout Mahasiswa")

# Tambahkan logo dan judul aplikasi
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://github.com/slvrch/dropout_scoring_app/blob/main/logo%20for%20dropout%20scoring%20application.png?raw=true", width=200)
with col2:
    st.header('Dropout Scoring App (Prototype)')
    
# Sidebar untuk input
st.sidebar.header("📥 Input Data Mahasiswa")

# Input seluruh fitur yang digunakan dalam pemodelan
data = pd.DataFrame()

data["Admission_grade"] = [st.sidebar.number_input("Admission Grade", value=127.3, key="admiss_grade")]
data["Application_order"] = [st.sidebar.number_input("Application_order", value=4, key="application_order")]
data["Age_at_enrollment"] = [st.sidebar.number_input("Age at Enrollment", value=23, key="age")]

data["Curricular_units_1st_sem_credited"] = [st.sidebar.number_input("1st Semester Credited", value=2, key="1st_sem_credited")]
data["Curricular_units_1st_sem_enrolled"] = [st.sidebar.number_input("1st Semester Enrolled", value=4, key="1st_sem_enrolled")]
data["Curricular_units_1st_sem_evaluations"] = [st.sidebar.number_input("1st Semester Evaluations", value=3, key="1st_sem_evaluations")]
data["Curricular_units_1st_sem_approved"] = [st.sidebar.number_input("1st Semester Approved", value=5, key="1st_sem_approved")]
data["Curricular_units_1st_sem_grade"] = [st.sidebar.number_input("1st Semester Grade", value=13.42, key="1st_sem_grade")]

data["Curricular_units_2nd_sem_credited"] = [st.sidebar.number_input("2nd Semester Credited", value=1, key="2nd_sem_credited")]
data["Curricular_units_2nd_sem_enrolled"] = [st.sidebar.number_input("2nd Semester Enrolled", value=3, key="2nd_sem_enrolled")]
data["Curricular_units_2nd_sem_evaluations"] = [st.sidebar.number_input("2nd Semester Evaluations", value=2, key="2nd_sem_evaluations")]
data["Curricular_units_2nd_sem_approved"] = [st.sidebar.number_input("2nd Semester Approved", value=2, key="2nd_sem_approved")]
data["Curricular_units_2nd_sem_grade"] = [st.sidebar.number_input("2nd Semester Grade", value=12.40, key="2nd_sem_grade")]

data["Previous_qualification_grade"] = [st.sidebar.number_input("Previous Qualification Grade", value=122.0, key="prev_qualification_grade")]
data["Unemployment_rate"] = [st.sidebar.number_input("Unemployment Rate", value=9.4, key="unemployment_rate")]
data["Inflation_rate"] = [st.sidebar.number_input("Inflation Rate", value=1.4, key="inflation_rate")]
data["GDP"] = [st.sidebar.number_input("GDP", value=1.74, key="GDP")]

data["Application_mode"] =[st.sidebar.selectbox("Application Mode", options=encoder_Application_mode.classes_, index=1, key="appli_mode")]
data["Course"] = [st.sidebar.selectbox("Course", options=encoder_Course.classes_, index=1, key="course")]
data["Previous_qualification"] = [st.sidebar.selectbox("Previous Qualification", options=encoder_Previous_qualification.classes_, index=2, key="prev_qualification")]
data["Mothers_qualification"] = [st.sidebar.selectbox("Mothers Qualification", options=encoder_Mothers_qualification.classes_, index=3, key="mothers_qualif")]
data["Fathers_qualification"] = [st.sidebar.selectbox("Fathers Qualification", options=encoder_Fathers_qualification.classes_, index=5, key="fathers_qualif")]
data["Mothers_occupation"] = [st.sidebar.selectbox("Mothers Occupation", options=encoder_Mothers_occupation.classes_, index=3, key="mothers_occupation")]
data["Fathers_occupation"] = [st.sidebar.selectbox("Fathers Occupation", options=encoder_Fathers_occupation.classes_, index=5, key="fathers_occupation")]

data["Tuition_fees_up_to_date"] = [st.sidebar.selectbox("Tuition Fees Up-to-date", options=encoder_Tuition_fees_up_to_date.classes_, index=0, key="tuition_fees_uptodate")]
data["Scholarship_holder"] = [st.sidebar.selectbox("Scholarship Holder", options=encoder_Scholarship_holder.classes_, index=1, key="scholarship_holder")]
data["Displaced"] = [st.sidebar.selectbox("Displaced", options=encoder_Displaced.classes_, index=1, key="displaced")]
data["Debtor"] = [st.sidebar.selectbox("Debtor", options=encoder_Debtor.classes_, index=1, key="debtor")]
data["Gender"] = [st.sidebar.selectbox("Gender", options=encoder_Gender.classes_, index=0, key="gender")]


# Tampilkan Data
st.subheader("📌 Data yang Dimasukkan")
st.dataframe(data, width=800)

if st.button("🔍 Prediksi"):
    new_data = data_preprocessing(data=data)
    with st.expander("View the Preprocessed Data"):
        st.dataframe(data=new_data, width=800, height=10)
    st.write("Status: {}".format(prediction(new_data)))
    
    #Prediksi status mahasiswa
    prediksi = prediction(new_data)
    
    # Tampilkan hasil prediksi dengan badge warna-warni
    st.subheader("🔎 Hasil Prediksi")
    if prediksi == "Dropout":
         st.error("⚠️ Mahasiswa berisiko *dropout*!", icon="⚠️")
    elif prediksi == "Graduate":
        st.success("✅ Mahasiswa diprediksi *graduate*!", icon="🎓")
    else:
        st.info("📚 Mahasiswa masih aktif *enrolled*.", icon="📚")