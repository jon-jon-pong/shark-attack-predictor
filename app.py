import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# -----------------------------
# Model Definition (must match training)
# -----------------------------
class MLPClassifier(nn.Module):
    """
    Multi-layer perceptron for binary classification.
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


@st.cache_resource
def load_model_and_data():
    """Load the trained model and get feature information from training data"""
    # Load model checkpoint
    checkpoint = torch.load("pytorch_mlp_model.pt", map_location="cpu", weights_only=False)
    n_features = checkpoint["n_features"]
    feature_columns = checkpoint["feature_columns"]
    
    # Initialize model
    model = MLPClassifier(in_features=n_features)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Load original data to get encoding mappings
    # Try multiple paths for flexibility (local dev vs deployment)
    csv_paths = [
        r"C:\Users\roark\Downloads\attacks_encoded_ml.csv",  # Local path
        "attacks_encoded_ml.csv",  # Same directory
        "data/attacks_encoded_ml.csv"  # Data folder
    ]
    
    df = None
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            break
        except FileNotFoundError:
            continue
    
    if df is None:
        raise FileNotFoundError("Could not find attacks_encoded_ml.csv. Please ensure the data file is available.")
    
    df.columns = [c.strip().replace(" ", "_").replace(".", "_").lower() for c in df.columns]
    
    # Get unique values for dropdowns
    unique_values = {
        'type': sorted(df['type'].dropna().unique()),
        'country': sorted(df['country'].dropna().unique()),
        'area': sorted(df['area'].dropna().unique()),
        'activity': sorted(df['activity'].dropna().unique()),
        'sex': sorted(df['sex'].dropna().unique())
    }
    
    # Get feature statistics for normalization (year and injury)
    year_mean = df['year'].mean()
    year_std = df['year'].std()
    injury_mean = df['injury'].mean()
    injury_std = df['injury'].std()
    
    stats = {
        'year_mean': year_mean,
        'year_std': year_std,
        'injury_mean': injury_mean,
        'injury_std': injury_std
    }
    
    return model, feature_columns, unique_values, stats


def prepare_input(year, type_val, country, area, activity, sex, injury, 
                   feature_columns, unique_values, stats):
    """Prepare input features matching the training format"""
    
    # Create a dataframe with the input
    input_data = pd.DataFrame({
        'year': [year],
        'type': [str(type_val)],
        'country': [str(country)],
        'area': [str(area)],
        'activity': [str(activity)],
        'sex': [str(sex)],
        'injury': [injury]
    })
    
    # Normalize numeric features
    input_data['year'] = (input_data['year'] - stats['year_mean']) / (stats['year_std'] + 1e-9)
    input_data['injury'] = (input_data['injury'] - stats['injury_mean']) / (stats['injury_std'] + 1e-9)
    
    # One-hot encode categorical features
    X_num = input_data[['year', 'injury']].astype(float)
    X_cat = pd.get_dummies(input_data[['type', 'country', 'area', 'activity', 'sex']])
    
    # Combine
    X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    
    # Ensure all training features are present (add missing columns as 0)
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    
    # Reorder to match training
    X = X[feature_columns]
    
    return torch.from_numpy(X.values.astype(np.float32))


def predict_fatality(model, input_tensor):
    """Make prediction and return probability"""
    with torch.no_grad():
        logit = model(input_tensor)
        prob = torch.sigmoid(logit).item()
    return prob


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="Shark Attack Predictor", page_icon="🦈", layout="wide")
    
    st.title("🦈 Shark Attack Fatality Predictor")
    st.markdown("""
    This machine learning model predicts the probability of a shark attack being fatal 
    based on various factors. The model was trained on historical shark attack data 
    and achieved **88.7% accuracy**.
    """)
    
    # Load model
    try:
        model, feature_columns, unique_values, stats = load_model_and_data()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Make sure 'pytorch_mlp_model.pt' and the CSV file are in the correct location.")
        return
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Attack Details")
        
        # Input fields
        year = st.number_input("Year", min_value=1900, max_value=2030, value=2023, step=1)
        
        type_val = st.selectbox("Type", options=unique_values['type'], 
                                help="Type code representing attack circumstances")
        
        country = st.selectbox("Country", options=unique_values['country'],
                              help="Country code where attack occurred")
        
        area = st.selectbox("Area", options=unique_values['area'],
                           help="Specific area/region code")
        
        activity = st.selectbox("Activity", options=unique_values['activity'],
                               help="Activity victim was engaged in")
        
        sex = st.selectbox("Sex", options=unique_values['sex'],
                          help="0=Female, 1=Male, 2=Unknown")
        
        injury = st.number_input("Injury Code", min_value=0, max_value=5000, value=1000, step=10,
                                help="Numeric code representing injury severity")
        
        predict_button = st.button("🔮 Predict Fatality Risk", type="primary")
    
    with col2:
        st.subheader("📊 Prediction")
        
        if predict_button:
            try:
                # Prepare input
                input_tensor = prepare_input(year, type_val, country, area, activity, sex, injury,
                                            feature_columns, unique_values, stats)
                
                # Make prediction
                probability = predict_fatality(model, input_tensor)
                
                # Display result
                st.metric("Fatality Probability", f"{probability*100:.1f}%")
                
                # Risk level indicator
                if probability < 0.3:
                    st.success("🟢 LOW RISK")
                    st.write("This attack profile suggests a lower probability of fatality.")
                elif probability < 0.7:
                    st.warning("🟡 MODERATE RISK")
                    st.write("This attack profile suggests moderate risk. Medical attention crucial.")
                else:
                    st.error("🔴 HIGH RISK")
                    st.write("This attack profile suggests high fatality risk. Immediate emergency response needed!")
                
                # Progress bar visualization
                st.progress(probability)
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
    
    # Add information section
    st.markdown("---")
    st.subheader("ℹ️ About This Model")
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.metric("Model Accuracy", "88.7%")
    
    with info_col2:
        st.metric("Features Used", "7")
        st.caption("Year, Type, Country, Area, Activity, Sex, Injury")
    
    with info_col3:
        st.metric("Training Samples", "~5,500")
    
    with st.expander("🔍 How It Works"):
        st.markdown("""
        This model uses a **Multi-Layer Perceptron (MLP)** neural network trained on historical 
        shark attack data. It considers:
        
        - **Temporal factors**: Year of attack
        - **Geographic factors**: Country and specific area
        - **Circumstantial factors**: Type of attack and victim activity
        - **Victim factors**: Sex
        - **Injury severity**: Coded injury information
        
        The model learns complex patterns and interactions between these features to estimate 
        fatality risk.
        """)


if __name__ == "__main__":
    main()
