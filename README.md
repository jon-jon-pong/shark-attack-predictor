# 🦈 Shark Attack Fatality Predictor

A machine learning model that predicts the probability of a shark attack being fatal based on various factors including location, activity, and injury severity.

## 📊 Model Performance

- **Accuracy**: 88.7%
- **Architecture**: Multi-Layer Perceptron (MLP) Neural Network
- **Features Used**: 7 (Year, Type, Country, Area, Activity, Sex, Injury)
- **Training Samples**: ~5,500 historical shark attacks

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
shark-attack-predictor/
├── app.py                      # Streamlit web application
├── encoded.py                  # Model training script
├── model.py                    # Original model experiment
├── pytorch_mlp_model.pt        # Trained model weights (not in repo)
├── attacks_encoded_ml.csv      # Training data (not in repo)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 Training Your Own Model

1. Prepare your dataset in CSV format with columns: Year, Type, Country, Area, Activity, Sex, Injury, Fatal
2. Run the training script:

```bash
python encoded.py --csv "path/to/your/data.csv"
```

3. The trained model will be saved as `pytorch_mlp_model.pt`

## 📊 Features

The model considers the following factors:

- **Year**: Temporal trends in shark attacks
- **Type**: Type code of the attack
- **Country**: Geographic location (country code)
- **Area**: Specific area/region code
- **Activity**: What the victim was doing
- **Sex**: Gender of the victim (0=Female, 1=Male, 2=Unknown)
- **Injury**: Numeric code representing injury severity

## 🎯 How It Works

1. **Data Preprocessing**: Features are normalized (numeric) or one-hot encoded (categorical)
2. **Neural Network**: 3-layer MLP with BatchNorm and Dropout for regularization
3. **Prediction**: Outputs probability (0-1) of fatality
4. **Risk Classification**: 
   - 🟢 Low Risk: < 30%
   - 🟡 Moderate Risk: 30-70%
   - 🔴 High Risk: > 70%

## 📝 Note on Data

The training data (`attacks_encoded_ml.csv`) and model weights (`pytorch_mlp_model.pt`) are not included in this repository due to size and privacy considerations. 

To use this project:
1. Obtain or prepare your own shark attack dataset
2. Train the model using `encoded.py`
3. Run the Streamlit app

## 🌐 Deployment

This app can be deployed for free on [Streamlit Cloud](https://streamlit.io/cloud):

1. Push this repo to GitHub
2. Sign up for Streamlit Cloud
3. Connect your GitHub repo
4. Deploy!

## 📄 License

This project is for educational purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## ⚠️ Disclaimer

This model is for educational and research purposes only. It should not be used as the sole basis for medical or emergency response decisions.
