
#python -m streamlit run APP.py
# Function to loapd the model
import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

#python -m streamlit run APP.py
# Function to loapd the model
import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import time
import plotly.express as px
from sklearn import metrics
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder , OneHotEncoder , StandardScaler , MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn import svm
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import InstanceHardnessThreshold
lb=LabelEncoder()
MMS=MinMaxScaler()
sc=StandardScaler()
import warnings
warnings.simplefilter(action='ignore')
# Function to load the model
# Function to load the model
# Function to load the model


with open('C:/Users/LENOVO/Desktop/MY_PROJECT_2024/saved_data.pkl', 'rb') as f:
    my_df = pickle.load(f)


#my_df = pd.read_pickle('C:/Users/LENOVO/Desktop/MY_PROJECT_2024/_dataset-Copy.pkl')

#my_df=pd.read_csv('C:/Users/LENOVO/Desktop/MY_PROJECT_2024/diabetes_prediction_dataset-Copy.csv')
X=my_df.iloc[:,0:8]
X=X.dropna()
Y=my_df.iloc[:,-1]
X['gender']=lb.fit_transform(X['gender'])
from sklearn.preprocessing import OneHotEncoder
feature_cols = ['smoking_history']
encoder = OneHotEncoder()
# Perform one-hot encoding and create a DataFrame with the encoded columns
X_encoded = pd.DataFrame(encoder.fit_transform(X[feature_cols].values.reshape(-1, 1)).toarray(),
                         columns=encoder.get_feature_names_out(feature_cols))
# Concatenate the new DataFrame with the original DataFrame
X = pd.concat([X.drop(feature_cols, axis=1), X_encoded], axis=1)
num_cols = ['age', 'bmi', 'HbA1c_level','blood_glucose_level']
X[num_cols] = sc.fit_transform(X[num_cols])



st.set_page_config(
    page_title="Doctor AI",
    page_icon="C:/Users/LENOVO/Desktop/MY_PROJECT_2024/avatar_streamly.png",
    layout="wide",
    # initial_sidebar_state="expanded",
)




    

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Load your model here, replace 'My-Predictor - under_Sampling - Copy.pkl' with the actual path to your model file
model_path = "C:/Users/LENOVO/Desktop/MY_PROJECT_2024/trained_model.sav"
model = load_model(model_path)

# Create a StandardScaler instance


# Function to preprocess input data
def preprocess_input(gender, age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level):
    # Encode categorical variables
    gender_map = {"Male": 1, "Female": 0}
    hypertension_map = {"No": 0, "Yes": 1}
    heart_disease_map = {"No": 0, "Yes": 1}
    
    gender_encoded = gender_map[gender]
    hypertension_encoded = hypertension_map[hypertension]
    heart_disease_encoded = heart_disease_map[heart_disease]

    # Combine encoded categorical and scaled numerical features
    input_features = [gender_encoded, age, hypertension_encoded, heart_disease_encoded, bmi, HbA1c_level, blood_glucose_level]
    
    return np.array([input_features])


# Function to predict diabetes
def predict_diabetes(input_features):
    # Make predictions using your model
    prediction = model.predict(input_features)

    return prediction

def main():
   
# Define the icon path
    icon_path = "C:/Users/LENOVO/Desktop/MY_PROJECT_2024/avatar_streamly.png"

    # Display the icon and title with center alignment
    st.write(
        f"""
        <div style="display: flex; align-items: center; justify-content: center;">
            <h1 style="text-align: center;">Diabetes Diagnosis</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Input fields
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=1, max_value=150, step=1)
    hypertension = st.selectbox("Hypertension", ["Yes", "No"])
    heart_disease = st.selectbox("Heart Disease", ["Yes", "No"])
    smoking_history = st.selectbox("Smoking History", ["No Info", "Currently","Ever","Former","Never","Not Current"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1)
    HbA1c_level = st.number_input("HbA1c Level", min_value=0.0, max_value=20.0, step=0.1)
    blood_glucose_level = st.number_input("Blood Glucose Level", min_value=50, max_value=400, step=1)

    if(smoking_history=='No Info'):
       [No_Info,Currently,Ever,Former,Never,Not_Current]=[1,0,0,0,0,0]
    if(smoking_history=='Currently'):
       [No_Info,Currently,Ever,Former,Never,Not_Current]=[0,1,0,0,0,0]
    if(smoking_history=='Ever'):
       [No_Info,Currently,Ever,Former,Never,Not_Current]=[0,0,1,0,0,0]
    if(smoking_history=='Former'):
       [No_Info,Currently,Ever,Former,Never,Not_Current]=[0,0,0,1,0,0]
    if(smoking_history=='Never'):
       [No_Info,Currently,Ever,Former,Never,Not_Current]=[0,0,0,0,1,0]
    if(smoking_history=='Not Current'):
       [No_Info,Currently,Ever,Former,Never,Not_Current]=[0,0,0,0,0,1]
    


    #if not st.button("Predict"):
        


  
    st.markdown(
        """
        
        <style>
        
        .stButton > button {
            width: 100%;
        }
        
       
        
        </style>
        """,
        unsafe_allow_html=True
    )

    # 



    if st.button("Predict"):
        # Preprocess input values
        
        arr1 = preprocess_input(gender, age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level) 

        arr2 = np.array([[No_Info, Currently, Ever, Former, Never, Not_Current]])

        input_data = np.concatenate((arr1, arr2), axis=1)

       
        input_data_reshaped = np.array(input_data).reshape(1, -1)

# Select only the numeric columns for scaling
        num_cols_indices = [1, 4, 5, 6]  # Indices of numeric columns
        input_data_numeric = input_data_reshaped[:, num_cols_indices]

        # Scale the numeric columns using the previously fitted StandardScaler instance 'sc'
        input_data_scaled = sc.transform(input_data_numeric)

        # Replace the numeric columns with the scaled values in the original array
        input_data_reshaped[:, num_cols_indices] = input_data_scaled

        # Make prediction
        prediction = predict_diabetes(input_data_reshaped)
        
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: center;">
                <h2>Diabetes Prediction Result</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    

        if prediction == 1:
            st.write(
                f"""
                <div style="text-align: center; font-size: 30px;">
                    You have diabetes!🙁
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.write(
                f"""
                <div style="text-align: center; font-size: 30px;">
                    🎉Great news🎉!<br> You are not diabetes
                </div>
                """,
                unsafe_allow_html=True
            )


if __name__ == "__main__":
    main()
