import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from AQ_model import binary_nn


# Streamlit app title and description
st.title("Air Quality Type Prediction")
st.write("Enter the air quality measurements to predict the type (Industrial or Non-Industrial)")

# Creating the input fields for the 7 features
st.header("Input Parameters :")
co = st.number_input("CO (μg/m³)", min_value=0.0, max_value=1000.0, value=200.0, step=0.1)
no2 = st.number_input("NO2 (μg/m³)", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
so2 = st.number_input("SO2 (μg/m³)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
city = st.selectbox("City", options=['Beijing', 'Delhi', 'Moscow', 'Stockholm', 'Vancouver', 'Zurich'])
o3 = st.number_input("O3 (μg/m³)", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
pm25 = st.number_input("PM2.5 (μg/m³)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
pm10 = st.number_input("PM10 (μg/m³)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
# default input size is 12 as we have 12 parameters
default_input_size = 12 

# Button to trigger prediction
if st.button("Predict"):
    # Load the trained model
    model = binary_nn(input_size=default_input_size)
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.load_state_dict(torch.load("AQ_prediction.pth",map_location = torch.device(device)))
        model.eval()  #Setting the model to evaluation mode!
    except FileNotFoundError:
        st.error("Model file 'model.pth' not found. Please ensure the model is saved in the same directory.")
        st.stop()

    # Load the saved scaler
    try:
        with open("standard_scalar.pickle", "rb") as f:
            standar_scaler = pickle.load(f)
            f.close()
            
    except FileNotFoundError:
        st.error("Scaler file 'scaler.pkl' not found. Please ensure the scaler is saved in the same directory.")
        st.stop()
        
    # Loading the result_encoder! it's just 2 values but this would be the proper way.
    try:
        with open("result_encoder.pickle", "rb") as f:
            result_encoder = pickle.load(f)
            f.close()
            
    except FileNotFoundError:
        st.error("encoder file 'result_encoder.pkl' not found. Please ensure the encoder is saved in the same directory.")
        st.stop()

    # Loading the ohe ecoder file for city transformation to match tensor input.
    try:
        with open("city_ohe.pickle", "rb") as f:
            city_encoder = pickle.load(f)
            f.close()
            
    except FileNotFoundError:
        st.error("encoder file 'city_ohe.pkl' not found. Please ensure the encoder is saved in the same directory.")
        st.stop()
        
        
    # Prepare input data
    
    input_data = np.array([[co, no2, so2, o3, pm25, pm10]], dtype=np.float32)
    input_df = pd.DataFrame(input_data, columns=["CO", "NO2", "SO2", "O3", "PM2.5", "PM10"])
    city = city_encoder.transform([[city]])
    # st.write(city)
    df_city = pd.DataFrame(city.toarray(),columns=city_encoder.get_feature_names_out(['city']))
    # st.write(df_city)
    
    # st.write(input_df)
    # st.write(df_city)
    df_input = pd.merge(input_df,df_city,left_index=True,right_index=True)
    st.write(df_input)
    
    # Scale the input data
    input_scaled = standar_scaler.transform(df_input)
    # st.write(input_scaled)
    
    # Convert to PyTorch tensor
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    
    # Making prediction with our AQ_prediction model with saved params.
    with torch.no_grad():
        output = model(input_tensor)
        probability = output.item()
        input_map = dict(enumerate(result_encoder.classes_))
        st.write(input_map)  #I just create a simple map of result set
        st.write(f"Device in use : {device}")
    
    
    # Display the following results
    st.header("Prediction Results :")
    st.write(f"**Predicted Type**: {input_map.get(probability)}")

# # Instructions for saving model and scaler
# st.sidebar.header("Instructions")
# st.sidebar.write("1. Ensure your trained PyTorch model is saved as 'model.pth'.")
# st.sidebar.write("2. Ensure the StandardScaler used during training is saved as 'scaler.pkl'.")
# st.sidebar.write("3. Place both files in the same directory as this app.")
# st.sidebar.write("4. Input values should be within realistic ranges for air quality measurements.")