import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CourseCompletionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.metrics = {}
        
    def load_data(self, filepath='Course_Completion_Prediction.csv'):
        """Load and prepare the dataset"""
        print("Loading data...")
        df = pd.read_csv(filepath)
        print(f"Dataset shape: {df.shape}")
        print(f"Target distribution:\n{df['Completed'].value_counts()}")
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for modeling"""
        print("\nPreprocessing data...")
        
        # Create a copy
        data = df.copy()
        
        # Drop unnecessary columns
        drop_cols = ['Student_ID', 'Name', 'Course_ID', 'Course_Name', 
                     'Enrolment_Date', 'City']
        data = data.drop(columns=drop_cols, errors='ignore')
        
        # Separate features and target
        X = data.drop('Completed', axis=1)
        y = data['Completed']
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        
        # Encode categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"Features: {len(self.feature_names)}")
        print(f"Samples: {len(X)}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print("\nSplitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train, model_type='random_forest'):
        """Train the model"""
        print(f"\nTraining {model_type} model...")
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.fit(X_train, y_train)
        print("Model training completed!")
        
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        print("\nEvaluating model...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Print results
        print("\n" + "="*50)
        print("MODEL PERFORMANCE METRICS")
        print("="*50)
        print(f"Accuracy:  {self.metrics['accuracy']:.4f}")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall:    {self.metrics['recall']:.4f}")
        print(f"F1-Score:  {self.metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {self.metrics['roc_auc']:.4f}")
        print("\nConfusion Matrix:")
        print(self.metrics['confusion_matrix'])
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Important Features:")
            print(feature_importance.head(10).to_string(index=False))
            
            self.metrics['feature_importance'] = feature_importance.to_dict('records')
        
        return self.metrics
    
    def save_model(self, model_name='course_completion_model_v1'):
        """Save the trained model and artifacts"""
        print(f"\nSaving model as {model_name}...")
        
        # Save model
        joblib.dump(self.model, f'{model_name}.pkl')
        
        # Save scaler
        joblib.dump(self.scaler, f'{model_name}_scaler.pkl')
        
        # Save label encoders
        joblib.dump(self.label_encoders, f'{model_name}_encoders.pkl')
        
        # Save feature names
        with open(f'{model_name}_features.json', 'w') as f:
            json.dump(self.feature_names, f)
        
        # Save metrics
        with open(f'{model_name}_metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        print("Model saved successfully!")
        print(f"Files created:")
        print(f"  - {model_name}.pkl")
        print(f"  - {model_name}_scaler.pkl")
        print(f"  - {model_name}_encoders.pkl")
        print(f"  - {model_name}_features.json")
        print(f"  - {model_name}_metrics.json")
    
    def load_trained_model(self, model_name='course_completion_model_v1'):
        """Load a previously trained model"""
        print(f"\nLoading model {model_name}...")
        
        self.model = joblib.load(f'{model_name}.pkl')
        self.scaler = joblib.load(f'{model_name}_scaler.pkl')
        self.label_encoders = joblib.load(f'{model_name}_encoders.pkl')
        
        with open(f'{model_name}_features.json', 'r') as f:
            self.feature_names = json.load(f)
        
        with open(f'{model_name}_metrics.json', 'r') as f:
            self.metrics = json.load(f)
        
        print("Model loaded successfully!")
        return self.metrics


def main():
    """Main training pipeline"""
    print("="*60)
    print("COURSE COMPLETION PREDICTION MODEL TRAINING")
    print("="*60)
    
    # Initialize model
    model = CourseCompletionModel()
    
    # Load data
    df = model.load_data()
    
    # Preprocess
    X, y = model.preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = model.split_data(X, y)
    
    # Train model (try different models)
    model_types = ['random_forest', 'gradient_boosting', 'logistic_regression']
    best_model = None
    best_f1 = 0
    best_type = None
    
    for model_type in model_types:
        print("\n" + "="*60)
        temp_model = CourseCompletionModel()
        temp_model.feature_names = model.feature_names
        temp_model.label_encoders = model.label_encoders
        temp_model.scaler = model.scaler
        
        temp_model.train_model(X_train, y_train, model_type=model_type)
        metrics = temp_model.evaluate_model(X_test, y_test)
        
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_model = temp_model
            best_type = model_type
    
    print("\n" + "="*60)
    print(f"BEST MODEL: {best_type.upper()}")
    print(f"F1-Score: {best_f1:.4f}")
    print("="*60)
    
    # Save the best model
    best_model.save_model()
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    main()
