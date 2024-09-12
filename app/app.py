import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import requests
import json
import os
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from datetime import datetime, timedelta

# Gemini API configuration
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
API_KEY = "enter your gemini api here"

# Weather API configuration (replace with your preferred weather API)
WEATHER_API_URL = "weatherapi-com.p.rapidapi.com"
WEATHER_API_KEY = "2d7198105fmsha78df4c828aea6ep182ce4jsn6a513052a904"

# Function to generate synthetic data
def generate_synthetic_data(num_samples):
    soil_types = ['Clay', 'Loam', 'Sandy', 'Silt']
    weather_conditions = ['Sunny', 'Rainy', 'Cloudy', 'Windy']
    crop_types = ['Wheat', 'Rice', 'Corn', 'Soybeans']
    
    data = {
        'soil_type': np.random.choice(soil_types, num_samples),
        'temperature': np.random.uniform(10, 35, num_samples),
        'rainfall': np.random.uniform(0, 1000, num_samples),
        'humidity': np.random.uniform(30, 90, num_samples),
        'weather_condition': np.random.choice(weather_conditions, num_samples),
        'crop_type': np.random.choice(crop_types, num_samples),
        'fertilizer_used': np.random.uniform(50, 200, num_samples),
        'yield': np.random.uniform(1000, 5000, num_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add some correlations
    df.loc[df['soil_type'] == 'Clay', 'yield'] *= 0.9
    df.loc[df['soil_type'] ==  'Loam', 'yield'] *= 1.1
    df.loc[df['weather_condition'] == 'Rainy', 'yield'] *= 1.05
    df.loc[df['crop_type'] == 'Corn', 'yield'] *= 1.1
    df['yield'] += df['fertilizer_used'] * 5
    
    return df

# Function to train the crop yield prediction model
def train_model(data):
    X = data[['soil_type', 'temperature', 'rainfall', 'humidity', 'weather_condition', 'crop_type', 'fertilizer_used']]
    y = data['yield']
    
    X = pd.get_dummies(X, columns=['soil_type', 'weather_condition', 'crop_type'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, mse, r2, X.columns

# Function to predict crop yield
def predict_yield(model, scaler, input_data, feature_names):
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df, columns=['soil_type', 'weather_condition', 'crop_type'])
    
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[feature_names]
    
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    return prediction[0]

# Function to detect plant disease using Gemini API
def detect_plant_disease(image):
    image_base64 = base64.b64encode(image.getvalue()).decode('utf-8')
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": "Analyze this image and detect any plant diseases. If a disease is present, provide the following information:\n1. Name of the disease\n2. Brief description of the disease\n3. Possible causes\n4. Recommended treatment or management practices\nIf no disease is detected, state that the plant appears healthy."},
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}}
                ]
            }
        ]
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(f"{GEMINI_API_URL}?key={API_KEY}", headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"Error: {response.status_code} - {response.text}"

# Function to plot feature importance
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    return plt

# Function to plot correlation heatmap
def plot_correlation_heatmap(data):
    numeric_data = data.select_dtypes(include=[np.number])
    corr = numeric_data.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    return plt

# Function to get AI-powered crop recommendations
def get_crop_recommendations(soil_type, climate):
    prompt = f"Given a soil type of {soil_type} and a {climate} climate, what are the top 5 recommended crops to grow? For each crop, provide a brief explanation of why it's suitable."
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(f"{GEMINI_API_URL}?key={API_KEY}", headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"Error: {response.status_code} - {response.text}"

# Function to get AI-powered pest control advice
def get_pest_control_advice(crop, pest):
    prompt = f"What are the best organic pest control methods for controlling {pest} in {crop} crops? Provide at least 3 methods with brief explanations."
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(f"{GEMINI_API_URL}?key={API_KEY}", headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"Error: {response.status_code} - {response.text}"

# New function to get weather forecast
def get_weather_forecast(city):
    params = {
        'q': city,
        'appid': WEATHER_API_KEY,
        'units': 'metric'
    }
    response = requests.get(WEATHER_API_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        forecasts = data['list'][:5]  # Get forecast for next 5 time steps
        return [{'date': datetime.fromtimestamp(f['dt']),
                 'temp': f['main']['temp'],
                 'description': f['weather'][0]['description']} for f in forecasts]
    else:
        return None

# New function for soil analysis
def analyze_soil(ph, nitrogen, phosphorus, potassium):
    prompt = f"Given soil test results of pH {ph}, Nitrogen {nitrogen} ppm, Phosphorus {phosphorus} ppm, and Potassium {potassium} ppm, provide an analysis of soil health and recommendations for soil amendments and fertilizers."
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(f"{GEMINI_API_URL}?key={API_KEY}", headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"Error: {response.status_code} - {response.text}"

# New function for crop rotation planning
def plan_crop_rotation(current_crop, soil_type, climate):
    prompt = f"Suggest a 3-year crop rotation plan for a farm currently growing {current_crop} with {soil_type} soil in a {climate} climate. Explain the benefits of each crop in the rotation."
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(f"{GEMINI_API_URL}?key={API_KEY}", headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"Error: {response.status_code} - {response.text}"

# New function for market price prediction
def predict_market_price(crop):
    prompt = f"Based on historical trends and current market conditions, predict the price range for {crop} in the next 3 months. Provide a brief explanation for the prediction."
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(f"{GEMINI_API_URL}?key={API_KEY}", headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"Error: {response.status_code} - {response.text}"

# New function for carbon footprint calculation
def calculate_carbon_footprint(farm_size, crop_type, fertilizer_use, machinery_hours):
    prompt = f"Calculate the approximate carbon footprint for a {farm_size} hectare farm growing {crop_type}, using {fertilizer_use} kg/ha of fertilizer annually, and {machinery_hours} hours of machinery operation per week. Provide the result in CO2 equivalent and suggest ways to reduce the carbon footprint."
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(f"{GEMINI_API_URL}?key={API_KEY}", headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"Error: {response.status_code} - {response.text}"

# Streamlit app
def main():
    st.set_page_config(page_title="AgroInsight", layout="wide")
    st.title("AgroInsight")
    
    menu = ["Home", "Crop Yield Prediction", "Plant Disease Detection", "Data Analysis", 
            "Crop Recommendations", "Pest Control Advice", "Weather Forecast", 
            "Soil Analysis", "Crop Rotation Planner", "Market Price Predictor", 
            "Carbon Footprint Calculator", "Data Upload"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.write("Welcome to AgroInsight. Choose an option from the sidebar to get started.")
        st.write("This application offers the following features:")
        st.write("1. Crop Yield Prediction: Predict crop yields based on various factors.")
        st.write("2. Plant Disease Detection: Identify plant diseases from images.")
        st.write("3. Data Analysis: Analyze agricultural data and visualize insights.")
        st.write("4. Crop Recommendations: Get AI-powered crop recommendations based on soil and climate.")
        st.write("5. Pest Control Advice: Get AI-powered organic pest control methods.")
        st.write("6. Weather Forecast: Get weather forecasts for your location.")
        st.write("7. Soil Analysis: Analyze soil health and get recommendations.")
        st.write("8. Crop Rotation Planner: Get suggestions for crop rotation.")
        st.write("9. Market Price Predictor: Predict future crop prices.")
        st.write("10. Carbon Footprint Calculator: Estimate your farm's carbon footprint.")
        st.write("11. Data Upload: Upload your own dataset for analysis and prediction.")
    
    elif choice == "Crop Yield Prediction":
        st.header("Crop Yield Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_samples = st.number_input("Number of synthetic data samples", min_value=100, max_value=10000, value=1000, step=100)
            if st.button("Generate Synthetic Data"):
                data = generate_synthetic_data(num_samples)
                st.session_state['data'] = data
                st.write("Synthetic data generated successfully!")
                st.write(data.head())
                
                csv = data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="synthetic_data.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            if 'data' in st.session_state and st.button("Train Model"):
                model, scaler, mse, r2, feature_names = train_model(st.session_state['data'])
                st.session_state['model'] = model
                st.session_state['scaler'] = scaler
                st.session_state['feature_names'] = feature_names
                st.write(f"Model trained successfully!")
                st.write(f"Mean Squared Error: {mse:.2f}")
                st.write(f"R-squared Score: {r2:.2f}")
        
        if 'model' in st.session_state:
            st.subheader("Make a Prediction")
            col1, col2 = st.columns(2)
            
            with col1:
                soil_type = st.selectbox("Soil Type", ['Clay', 'Loam', 'Sandy', 'Silt'])
                temperature = st.slider("Temperature (°C)", 10.0, 35.0, 25.0)
                rainfall = st.slider("Rainfall (mm)", 0.0, 1000.0, 500.0)
                humidity = st.slider("Humidity (%)", 30.0, 90.0, 60.0)
            
            with col2:
                weather_condition = st.selectbox("Weather Condition", ['Sunny', 'Rainy', 'Cloudy', 'Windy'])
                crop_type = st.selectbox("Crop Type", ['Wheat', 'Rice', 'Corn', 'Soybeans'])
                fertilizer_used = st.slider("Fertilizer Used (kg/hectare)", 50.0, 200.0, 100.0)
            
            if st.button("Predict Yield"):
                input_data = {
                    'soil_type': soil_type,
                    'temperature': temperature,
                    'rainfall': rainfall,
                    'humidity': humidity,
                    'weather_condition': weather_condition,
                    'crop_type': crop_type,
                    'fertilizer_used': fertilizer_used
                }
                prediction = predict_yield(st.session_state['model'], st.session_state['scaler'], input_data, st.session_state['feature_names'])
                st.write(f"Predicted Crop Yield: {prediction:.2f} kg/hectare")
            
            st.subheader("Feature Importance")
            fig = plot_feature_importance(st.session_state['model'], st.session_state['feature_names'])
            st.pyplot(fig)
    
    elif choice == "Plant Disease Detection":
        st.header("Plant Disease Detection")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            if st.button("Detect Disease"):
                with st.spinner("Analyzing image..."):
                    result = detect_plant_disease(uploaded_file)
                st.write(result)
    
    elif choice == "Data Analysis":
        st.header("Data Analysis")
        
        if 'data' not in st.session_state:
            st.warning("No data available. Please generate synthetic data or upload a file first.")
        else:
            data = st.session_state['data']
            
            st.subheader("Data Overview")
            st.write(data.head())
            
            st.subheader("Data Statistics")
            st.write(data.describe())
            
            st.subheader("Correlation Heatmap")
            fig = plot_correlation_heatmap(data)
            st.pyplot(fig)
            
            st.subheader("Pairplot")
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            pairplot = sns.pairplot(data[numeric_cols])
            st.pyplot(pairplot.fig)
    
    elif choice == "Crop Recommendations":
        st.header("AI-Powered Crop Recommendations")
        soil_type = st.selectbox("Soil Type", ['Clay', 'Loam', 'Sandy', 'Silt'])
        climate = st.selectbox("Climate", ['Tropical', 'Subtropical', 'Temperate', 'Arid', 'Mediterranean'])
        
        if st.button("Get Recommendations"):
            with st.spinner("Generating recommendations..."):
                recommendations = get_crop_recommendations(soil_type, climate)
            st.write(recommendations)
    
    elif choice == "Pest Control Advice":
        st.header("AI-Powered Pest Control Advice")
        crop = st.text_input("Crop")
        pest = st.text_input("Pest")
        
        if st.button("Get Advice"):
            with st.spinner("Generating advice..."):
                advice = get_pest_control_advice(crop, pest)
            st.write(advice)
    
    elif choice == "Weather Forecast":
        st.header("Weather Forecast")
        city = st.text_input("Enter city name")
        if st.button("Get Forecast"):
            with st.spinner("Fetching weather data..."):
                forecast = get_weather_forecast(city)
            if forecast:
                for f in forecast:
                    st.write(f"Date: {f['date']}, Temperature: {f['temp']}°C, Condition: {f['description']}")
            else:
                st.error("Unable to fetch weather data. Please check the city name and try again.")
    
    elif choice == "Soil Analysis":
        st.header("Soil Analysis")
        ph = st.slider("pH Level", 0.0, 14.0, 7.0, 0.1)
        nitrogen = st.number_input("Nitrogen (ppm)", 0.0, 1000.0, 100.0)
        phosphorus = st.number_input("Phosphorus (ppm)", 0.0, 1000.0, 100.0)
        potassium = st.number_input("Potassium (ppm)", 0.0, 1000.0, 100.0)
        
        if st.button("Analyze Soil"):
            with st.spinner("Analyzing soil..."):
                analysis = analyze_soil(ph, nitrogen, phosphorus, potassium)
            st.write(analysis)
    
    elif choice == "Crop Rotation Planner":
        st.header("Crop Rotation Planner")
        current_crop = st.text_input("Current Crop")
        soil_type = st.selectbox("Soil Type", ['Clay', 'Loam', 'Sandy', 'Silt'])
        climate = st.selectbox("Climate", ['Tropical', 'Subtropical', 'Temperate', 'Arid', 'Mediterranean'])
        
        if st.button("Plan Rotation"):
            with st.spinner("Planning crop rotation..."):
                plan = plan_crop_rotation(current_crop, soil_type, climate)
            st.write(plan)
    
    elif choice == "Market Price Predictor":
        st.header("Market Price Predictor")
        crop = st.text_input("Crop")
        
        if st.button("Predict Price"):
            with st.spinner("Predicting market price..."):
                prediction = predict_market_price(crop)
            st.write(prediction)
    
    elif choice == "Carbon Footprint Calculator":
        st.header("Carbon Footprint Calculator")
        farm_size = st.number_input("Farm Size (hectares)", 1.0, 10000.0, 100.0)
        crop_type = st.text_input("Main Crop Type")
        fertilizer_use = st.number_input("Annual Fertilizer Use (kg/ha)", 0.0, 1000.0, 100.0)
        machinery_hours = st.number_input("Weekly Machinery Operation Hours", 0.0, 168.0, 40.0)
        
        if st.button("Calculate Carbon Footprint"):
            with st.spinner("Calculating carbon footprint..."):
                footprint = calculate_carbon_footprint(farm_size, crop_type, fertilizer_use, machinery_hours)
            st.write(footprint)
    
    elif choice == "Data Upload":
        st.header("Data Upload")
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                
                st.session_state['data'] = data
                st.write("Data uploaded successfully!")
                st.write(data.head())
                
                st.subheader("Data Statistics")
                st.write(data.describe())
                
                st.subheader("Correlation Heatmap")
                fig = plot_correlation_heatmap(data)
                st.pyplot(fig)
                
                # Offer option to use this data for prediction
                if st.button("Use This Data for Prediction"):
                    try:
                        model, scaler, mse, r2, feature_names = train_model(data)
                        st.session_state['model'] = model
                        st.session_state['scaler'] = scaler
                        st.session_state['feature_names'] = feature_names
                        st.write(f"Model trained successfully on uploaded data!")
                        st.write(f"Mean Squared Error: {mse:.2f}")
                        st.write(f"R-squared Score: {r2:.2f}")
                    except Exception as e:
                        st.error(f"Error training model on uploaded data: {str(e)}")
                        st.write("Please ensure your data has the required columns for prediction.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.write("Please ensure your file is a valid CSV or Excel file.")

if __name__ == "__main__":
    main()
