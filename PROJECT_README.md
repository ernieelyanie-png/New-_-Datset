# Course Completion Prediction Project

This project provides machine learning models and dashboards for analyzing and predicting course completion rates.

## 📁 Project Files

- **app.py** - Main data visualization dashboard
- **train_model_v1.py** - ML model training script
- **monitor_dashboard.py** - ML model monitoring dashboard
- **Course_Completion_Prediction.csv** - Dataset
- **requirements.txt** - Python dependencies

## 🚀 Getting Started

### 1. Install Dependencies

```bash
# Using the virtual environment
& ".\venv\Scripts\python.exe" -m pip install -r requirements.txt
```

### 2. Train the Machine Learning Model

```bash
& "..\venv\Scripts\python.exe" train_model_v1.py
```

This will:
- Load and preprocess the data
- Train multiple models (Random Forest, Gradient Boosting, Logistic Regression)
- Select the best performing model
- Save the model and metrics to files

### 3. Run the Data Visualization Dashboard

```bash
& "..\venv\Scripts\streamlit.exe" run app.py
```

Features:
- 📊 Demographics analysis
- 🎓 Course analytics
- ⏱️ Engagement metrics
- 💰 Payment analysis
- 🔍 Interactive filters and data explorer

### 4. Run the Model Monitoring Dashboard

```bash
& "..\venv\Scripts\streamlit.exe" run monitor_dashboard.py
```

Features:
- 📊 Model performance overview
- 📈 Detailed metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- 🎯 Feature importance analysis
- 🔮 Confusion matrix visualization
- Model prediction interface

## 📊 Model Training Details

The training script (`train_model_v1.py`) performs:

1. **Data Preprocessing**
   - Removes unnecessary columns (IDs, names, etc.)
   - Encodes categorical variables
   - Handles missing values
   - Scales numerical features

2. **Model Training**
   - Random Forest Classifier
   - Gradient Boosting Classifier
   - Logistic Regression
   - Selects best model based on F1-score

3. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - ROC-AUC score
   - Confusion matrix
   - Feature importance ranking

4. **Model Artifacts**
   - `course_completion_model_v1.pkl` - Trained model
   - `course_completion_model_v1_scaler.pkl` - Feature scaler
   - `course_completion_model_v1_encoders.pkl` - Label encoders
   - `course_completion_model_v1_features.json` - Feature names
   - `course_completion_model_v1_metrics.json` - Performance metrics

## 🔍 Using the Trained Model

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('course_completion_model_v1.pkl')
scaler = joblib.load('course_completion_model_v1_scaler.pkl')
encoders = joblib.load('course_completion_model_v1_encoders.pkl')

# Prepare new data
new_data = pd.DataFrame({...})  # Your input data

# Encode categorical features
for col, encoder in encoders.items():
    if col in new_data.columns:
        new_data[col] = encoder.transform(new_data[col])

# Scale features
new_data_scaled = scaler.transform(new_data)

# Make predictions
prediction = model.predict(new_data_scaled)
probability = model.predict_proba(new_data_scaled)

print(f"Prediction: {'Completed' if prediction[0] == 1 else 'Not Completed'}")
print(f"Confidence: {probability[0][prediction[0]] * 100:.2f}%")
```

## 📦 Dependencies

- streamlit==1.53.0
- pandas==2.3.3
- plotly==6.5.2
- numpy==2.4.1
- scikit-learn==1.8.0
- joblib==1.5.3
- scipy==1.17.0

## 🤝 Contributing

Feel free to fork this repository and submit pull requests!

## 📄 License

MIT License
