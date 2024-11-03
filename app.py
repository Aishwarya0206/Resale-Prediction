import streamlit as st
import pickle
import numpy as np

# Load the model
with open("singapore_resale_price_model.pkl", 'rb') as f:
    model = pickle.load(f)

# Feature names 
features = ['Town', 'Flat Type', 'Storey Range', 'Floor Area (Sqm)', 'Flat Model', 
            'Transaction Year', 'Transaction Month', 'Remaining lease (months)']

# Categorical variable mappings
categorical_mappings = {
    'Town': {'ANG MO KIO': 0, 'BEDOK': 1, 'BISHAN': 2, 'BUKIT BATOK': 3, 'BUKIT MERAH': 4, 
             'BUKIT PANJANG': 5, 'BUKIT TIMAH': 6, 'CENTRAL AREA': 7, 'CHOA CHU KANG': 8, 
             'CLEMENTI': 9, 'GEYLANG': 10, 'HOUGANG': 11, 'JURONG EAST': 12, 'JURONG WEST': 13, 
             'KALLANG/WHAMPOA': 14, 'MARINE PARADE': 15, 'PASIR RIS': 16, 'PUNGGOL': 17, 
             'QUEENSTOWN': 18, 'SEMBAWANG': 19, 'SENGKANG': 20, 'SERANGOON': 21,
             'TAMPINES': 22, 'TOA PAYOH': 23, 'WOODLANDS': 24, 'YISHUN': 25,
            },
    
    'Flat Type': {'1 ROOM': 0, '2 ROOM': 1, '3 ROOM': 2, '4 ROOM': 3, 
                  '5 ROOM': 4, 'EXECUTIVE': 5, 'MULTI-GENERATION': 6,
                 },
    
    'Storey Range': {'01 TO 03': 0, '04 TO 06': 1, '07 TO 09': 2, '10 TO 12': 3, 
                     '13 TO 15': 4, '16 TO 18': 5, '19 TO 21': 6, '22 TO 24': 7,
                     '25 TO 27': 8, '28 TO 30': 9, '31 TO 33': 10, '34 TO 36': 11,
                     '37 TO 39': 12, '40 TO 42': 13, '43 TO 45': 14, '46 TO 48': 15,
                     '49 TO 51': 16},
    
    'Flat Model': {'2-room': 0, '3Gen': 1, 'Adjoined flat': 2, 'Apartment': 3, 'DBSS': 4, 
                   'Improved': 5, 'Improved-Maisonette': 6, 'Maisonette': 7, 'Model A': 8, 
                   'Model A-Maisonette': 9, 'Model A2': 10, 'Multi Generation': 11, 
                   'New Generation': 12, 'Premium Apartment': 13, 'Premium Apartment Loft': 14,
                   'Premium Maisonette': 15, 'Simplified': 16, 'Standard': 17, 'Terrace': 18,
                   'Type S1': 19, 'Type S2': 20,  
                  },
    'Transaction Month': {'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'Jun': 5, 
                      'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11}
}
range_mappings = {
    'Transaction Year': {'min': 2015, 'max': 2024}
}

# Input widgets for user interaction
st.title("House Price Prediction App")
st.markdown("<b>Problem Statement:</b>", unsafe_allow_html=True)
st.write("The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.")
st.markdown("<b>Deliverables:</b>", unsafe_allow_html=True)
st.write("The project will deliver the following deliverables: \n1. A well-trained machine learning model for resale price prediction. \n2. A user-friendly web application (built with Streamlit/ Flask / Django) deployed on the Render platform/ Any Cloud Platform \n3. Documentation and instructions for using the application. \n4. A project report summarizing the data analysis, model development, and deployment process.")

input_data = {}
for feature in features:
    if feature in categorical_mappings:
        selected_option = st.sidebar.selectbox(f"Select {feature.capitalize()}:", options=list(categorical_mappings[feature].keys()))
        input_data[feature] = categorical_mappings[feature][selected_option]

        # input_data[feature] = st.sidebar.number_input(f"{feature.capitalize()}:")
    elif 'Remaining lease (months)' == feature or 'Transaction Year' == feature:
        input_data[feature] = st.sidebar.number_input(f"{feature.capitalize()}:", step=1, format="%d")
    else:
        input_data[feature] = st.sidebar.number_input(f"{feature.capitalize()}:")
    
# Make predictions using the loaded model
if st.sidebar.button("Predict resale price"):
    input_array = np.array([input_data[feature] for feature in features]).reshape(1, -1)
    prediction = model.predict(input_array)

    # Display the prediction result
    prediction_scale = np.exp(prediction[0])
    st.subheader("Prediction Result:")
    st.write(f"The predicted house price is: {prediction_scale:,.2f} SGD")