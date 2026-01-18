import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="ML Model Monitoring Dashboard",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stAlert {
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_model_metrics(model_name='course_completion_model_v1'):
    """Load model metrics from JSON file"""
    try:
        with open(f'{model_name}_metrics.json', 'r') as f:
            metrics = json.load(f)
        return metrics
    except FileNotFoundError:
        return None

@st.cache_data
def load_feature_importance(model_name='course_completion_model_v1'):
    """Load feature importance data"""
    try:
        with open(f'{model_name}_metrics.json', 'r') as f:
            metrics = json.load(f)
        if 'feature_importance' in metrics:
            return pd.DataFrame(metrics['feature_importance'])
        return None
    except:
        return None

@st.cache_resource
def load_model(model_name='course_completion_model_v1'):
    """Load the trained model"""
    try:
        model = joblib.load(f'{model_name}.pkl')
        scaler = joblib.load(f'{model_name}_scaler.pkl')
        encoders = joblib.load(f'{model_name}_encoders.pkl')
        
        with open(f'{model_name}_features.json', 'r') as f:
            features = json.load(f)
        
        return model, scaler, encoders, features
    except:
        return None, None, None, None

def display_confusion_matrix(cm):
    """Display confusion matrix as a heatmap"""
    cm_array = np.array(cm)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_array,
        x=['Predicted: Not Completed', 'Predicted: Completed'],
        y=['Actual: Not Completed', 'Actual: Completed'],
        text=cm_array,
        texttemplate='%{text}',
        textfont={"size": 20},
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        height=400
    )
    
    return fig

def create_metrics_gauge(value, title, max_value=1.0):
    """Create a gauge chart for metrics"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': 0.8},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.5], 'color': '#ffcccc'},
                {'range': [0.5, 0.7], 'color': '#ffffcc'},
                {'range': [0.7, max_value], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.9
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def main():
    st.title("🤖 ML Model Monitoring Dashboard")
    st.markdown("### Course Completion Prediction Model - Performance Monitoring")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("⚙️ Dashboard Settings")
    model_name = st.sidebar.text_input("Model Name", value="course_completion_model_v1")
    
    refresh_btn = st.sidebar.button("🔄 Refresh Metrics")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dashboard Sections")
    st.sidebar.markdown("""
    - Model Performance Overview
    - Detailed Metrics Analysis
    - Feature Importance
    - Confusion Matrix
    - Model Predictions
    """)
    
    # Check if model exists
    if not os.path.exists(f'{model_name}.pkl'):
        st.error(f"❌ Model '{model_name}' not found! Please train the model first using train_model_v1.py")
        st.info("Run the following command to train the model:")
        st.code('python train_model_v1.py', language='bash')
        return
    
    # Load model and metrics
    metrics = load_model_metrics(model_name)
    
    if metrics is None:
        st.error("❌ Unable to load model metrics!")
        return
    
    st.success(f"✅ Model loaded successfully! Last updated: {metrics.get('timestamp', 'N/A')}")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", 
        "📈 Performance Metrics", 
        "🎯 Feature Analysis",
        "🔮 Make Predictions"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.header("Model Performance Overview")
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Accuracy",
                value=f"{metrics['accuracy']:.4f}",
                delta=f"{(metrics['accuracy'] - 0.8):.4f}"
            )
        
        with col2:
            st.metric(
                label="Precision",
                value=f"{metrics['precision']:.4f}",
                delta=f"{(metrics['precision'] - 0.8):.4f}"
            )
        
        with col3:
            st.metric(
                label="Recall",
                value=f"{metrics['recall']:.4f}",
                delta=f"{(metrics['recall'] - 0.8):.4f}"
            )
        
        with col4:
            st.metric(
                label="F1-Score",
                value=f"{metrics['f1_score']:.4f}",
                delta=f"{(metrics['f1_score'] - 0.8):.4f}"
            )
        
        st.markdown("---")
        
        # ROC-AUC metric
        st.subheader("ROC-AUC Score")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
            
            if metrics['roc_auc'] >= 0.9:
                st.success("🌟 Excellent Performance!")
            elif metrics['roc_auc'] >= 0.8:
                st.info("👍 Good Performance")
            elif metrics['roc_auc'] >= 0.7:
                st.warning("⚠️ Fair Performance")
            else:
                st.error("❌ Poor Performance")
        
        with col2:
            # Metrics comparison bar chart
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                'Score': [
                    metrics['accuracy'],
                    metrics['precision'],
                    metrics['recall'],
                    metrics['f1_score'],
                    metrics['roc_auc']
                ]
            })
            
            fig = px.bar(
                metrics_df,
                x='Score',
                y='Metric',
                orientation='h',
                title='Model Metrics Comparison',
                color='Score',
                color_continuous_scale='Viridis',
                range_x=[0, 1]
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Performance Metrics
    with tab2:
        st.header("Detailed Performance Analysis")
        
        # Gauge charts
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_acc = create_metrics_gauge(metrics['accuracy'], 'Accuracy')
            st.plotly_chart(fig_acc, use_container_width=True)
        
        with col2:
            fig_prec = create_metrics_gauge(metrics['precision'], 'Precision')
            st.plotly_chart(fig_prec, use_container_width=True)
        
        with col3:
            fig_rec = create_metrics_gauge(metrics['recall'], 'Recall')
            st.plotly_chart(fig_rec, use_container_width=True)
        
        st.markdown("---")
        
        # Confusion Matrix
        st.subheader("Confusion Matrix Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            cm_fig = display_confusion_matrix(metrics['confusion_matrix'])
            st.plotly_chart(cm_fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Matrix Interpretation")
            cm = np.array(metrics['confusion_matrix'])
            
            tn, fp, fn, tp = cm.ravel()
            
            st.markdown(f"""
            - **True Negatives (TN):** {tn}
            - **False Positives (FP):** {fp}
            - **False Negatives (FN):** {fn}
            - **True Positives (TP):** {tp}
            
            ---
            
            - **Total Predictions:** {tn + fp + fn + tp}
            - **Correct Predictions:** {tn + tp}
            - **Incorrect Predictions:** {fp + fn}
            """)
    
    # Tab 3: Feature Analysis
    with tab3:
        st.header("Feature Importance Analysis")
        
        feature_importance = load_feature_importance(model_name)
        
        if feature_importance is not None:
            # Top features chart
            top_n = st.slider("Number of top features to display", 5, 30, 15)
            
            top_features = feature_importance.head(top_n)
            
            fig = px.bar(
                top_features,
                x='importance',
                y='feature',
                orientation='h',
                title=f'Top {top_n} Most Important Features',
                labels={'importance': 'Importance Score', 'feature': 'Feature'},
                color='importance',
                color_continuous_scale='Plasma'
            )
            fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Feature importance table
            st.subheader("All Features Importance")
            st.dataframe(
                feature_importance.style.background_gradient(subset=['importance'], cmap='YlOrRd'),
                use_container_width=True,
                height=400
            )
        else:
            st.warning("⚠️ Feature importance data not available for this model.")
    
    # Tab 4: Make Predictions
    with tab4:
        st.header("Make Predictions")
        st.markdown("Load the model and make predictions on new data")
        
        model, scaler, encoders, features = load_model(model_name)
        
        if model is not None:
            st.success("✅ Model loaded and ready for predictions!")
            
            st.info("📝 To make predictions, prepare your input data with the following features and use the loaded model.")
            
            # Display feature list
            with st.expander("View Required Features"):
                st.write(features)
            
            # Sample prediction code
            st.subheader("Sample Prediction Code")
            st.code("""
import joblib
import pandas as pd

# Load model
model = joblib.load('course_completion_model_v1.pkl')
scaler = joblib.load('course_completion_model_v1_scaler.pkl')

# Prepare your data
new_data = pd.DataFrame({...})  # Your input data

# Scale and predict
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
probability = model.predict_proba(new_data_scaled)

print(f"Prediction: {prediction[0]}")
print(f"Probability: {probability[0]}")
            """, language='python')
            
        else:
            st.error("❌ Unable to load model for predictions.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🤖 ML Model Monitoring Dashboard | Built with Streamlit</p>
        <p>Model Version: v1.0 | Last Updated: """ + metrics.get('timestamp', 'N/A') + """</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
