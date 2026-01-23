import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Course Completion Dashboard V2 - ML Risk Prediction",
    page_icon="🤖",
    layout="wide"
)

# Load data with caching
@st.cache_data
def load_data():
    df = pd.read_csv('Course_Completion_Prediction.csv')
    # Convert Completed column to binary (1 for Completed, 0 for Not Completed)
    df['Completed'] = (df['Completed'] == 'Completed').astype(int)
    return df

# Feature Engineering
@st.cache_data
def engineer_features(df):
    """Create advanced features for ML models"""
    df_ml = df.copy()
    
    # Engagement Score (composite metric)
    df_ml['engagement_score'] = (
        df_ml['Login_Frequency'] * 0.2 +
        df_ml['Video_Completion_Rate'] * 0.3 +
        (df_ml['Time_Spent_Hours'] / df_ml['Time_Spent_Hours'].max()) * 100 * 0.2 +
        (df_ml['Average_Session_Duration_Min'] / df_ml['Average_Session_Duration_Min'].max()) * 100 * 0.15 +
        df_ml['Discussion_Participation'] * 0.15
    )
    
    # Performance Score
    df_ml['performance_score'] = (
        df_ml['Quiz_Score_Avg'] * 0.5 +
        df_ml['Progress_Percentage'] * 0.3 +
        (10 - df_ml['Assignments_Missed']) * 5 * 0.2
    )
    
    # Activity Recency (inverse of days since last login)
    df_ml['activity_recency'] = 1 / (df_ml['Days_Since_Last_Login'] + 1)
    
    # Video-to-Time Ratio (efficiency metric)
    df_ml['video_time_ratio'] = df_ml['Video_Completion_Rate'] / (df_ml['Time_Spent_Hours'] + 1)
    
    # Assignment Completion Rate
    total_assignments = 10  # Assuming 10 total assignments
    df_ml['assignment_completion_rate'] = ((total_assignments - df_ml['Assignments_Missed']) / total_assignments) * 100
    
    # Session Effectiveness
    df_ml['session_effectiveness'] = (
        (df_ml['Time_Spent_Hours'] / (df_ml['Login_Frequency'] + 1)) * 
        (df_ml['Quiz_Score_Avg'] / 100)
    )
    
    # Risk indicators (binary flags)
    df_ml['low_engagement_flag'] = (df_ml['Login_Frequency'] < df_ml['Login_Frequency'].median()).astype(int)
    df_ml['poor_performance_flag'] = (df_ml['Quiz_Score_Avg'] < 60).astype(int)
    df_ml['inactive_flag'] = (df_ml['Days_Since_Last_Login'] > 7).astype(int)
    df_ml['low_progress_flag'] = (df_ml['Progress_Percentage'] < 50).astype(int)
    
    return df_ml

# Prepare data for ML models
@st.cache_data
def prepare_ml_data(df):
    """Prepare features for machine learning"""
    df_ml = engineer_features(df)
    
    # Select features for modeling
    numeric_features = [
        'Age', 'Login_Frequency', 'Video_Completion_Rate', 
        'Time_Spent_Hours', 'Average_Session_Duration_Min',
        'Quiz_Score_Avg', 'Assignments_Missed', 'Discussion_Participation',
        'Days_Since_Last_Login', 'Progress_Percentage', 'Fee_Paid',
        'Payment_Amount', 'Instructor_Rating', 'Course_Duration_Weeks',
        'engagement_score', 'performance_score', 'activity_recency',
        'video_time_ratio', 'assignment_completion_rate', 'session_effectiveness',
        'low_engagement_flag', 'poor_performance_flag', 'inactive_flag', 'low_progress_flag'
    ]
    
    categorical_features = [
        'Gender', 'Education_Level', 'Employment_Status',
        'Course_Level', 'Payment_Mode', 'Discount_Used'
    ]
    
    # Create feature matrix
    X = df_ml[numeric_features].copy()
    
    # Encode categorical features
    le_dict = {}
    for cat in categorical_features:
        le = LabelEncoder()
        X[cat] = le.fit_transform(df_ml[cat].astype(str))
        le_dict[cat] = le
    
    # Target variable (inverse of Completed - we predict risk of NOT completing)
    y = 1 - df_ml['Completed']  # 1 = At Risk (Not Completed), 0 = Safe (Completed)
    
    return X, y, df_ml, le_dict

# Train ML models
@st.cache_resource
def train_models(X, y):
    """Train multiple ML models for risk prediction"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    predictions = {}
    probabilities = {}
    metrics = {}
    
    # 1. Logistic Regression
    st.sidebar.text("Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    lr_model.fit(X_train_scaled, y_train)
    models['Logistic Regression'] = (lr_model, scaler)
    predictions['Logistic Regression'] = lr_model.predict(X_test_scaled)
    probabilities['Logistic Regression'] = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    # 2. Random Forest
    st.sidebar.text("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100, random_state=42, max_depth=10,
        min_samples_split=5, class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = (rf_model, None)
    predictions['Random Forest'] = rf_model.predict(X_test)
    probabilities['Random Forest'] = rf_model.predict_proba(X_test)[:, 1]
    
    # 3. Gradient Boosting
    st.sidebar.text("Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=100, random_state=42, max_depth=5,
        learning_rate=0.1
    )
    gb_model.fit(X_train, y_train)
    models['Gradient Boosting'] = (gb_model, None)
    predictions['Gradient Boosting'] = gb_model.predict(X_test)
    probabilities['Gradient Boosting'] = gb_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics for each model
    for model_name in models.keys():
        y_pred = predictions[model_name]
        y_prob = probabilities[model_name]
        
        metrics[model_name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
    
    return models, X_test, y_test, predictions, probabilities, metrics, X_train, y_train

# Main app
def main():
    st.title("🤖 Course Completion Dashboard V2 - ML-Powered Risk Prediction")
    st.markdown("### Advanced Machine Learning Models for Student Risk Assessment")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Gender filter
    gender_options = ['All'] + df['Gender'].unique().tolist()
    selected_gender = st.sidebar.selectbox("Gender", gender_options)
    
    # Education Level filter
    education_options = ['All'] + df['Education_Level'].unique().tolist()
    selected_education = st.sidebar.selectbox("Education Level", education_options)
    
    # Course Level filter
    course_level_options = ['All'] + df['Course_Level'].unique().tolist()
    selected_course_level = st.sidebar.selectbox("Course Level", course_level_options)
    
    # Apply filters
    filtered_df = df.copy()
    if selected_gender != 'All':
        filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]
    if selected_education != 'All':
        filtered_df = filtered_df[filtered_df['Education_Level'] == selected_education]
    if selected_course_level != 'All':
        filtered_df = filtered_df[filtered_df['Course_Level'] == selected_course_level]
    
    # Display dataset overview
    st.header("📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Students", len(filtered_df))
    with col2:
        completion_rate = (filtered_df['Completed'].sum() / len(filtered_df) * 100)
        st.metric("Completion Rate", f"{completion_rate:.1f}%")
    with col3:
        avg_progress = filtered_df['Progress_Percentage'].mean()
        st.metric("Avg Progress", f"{avg_progress:.1f}%")
    with col4:
        total_courses = filtered_df['Course_ID'].nunique()
        st.metric("Total Courses", total_courses)
    
    st.markdown("---")
    
    # Prepare ML data
    with st.spinner("Preparing machine learning models..."):
        X, y, df_ml, le_dict = prepare_ml_data(df)
        models, X_test, y_test, predictions, probabilities, metrics, X_train, y_train = train_models(X, y)
    
    st.sidebar.success("✅ Models trained successfully!")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🤖 ML Model Performance",
        "🎯 Risk Prediction Dashboard", 
        "📊 Feature Importance",
        "🔍 Individual Prediction",
        "📈 Advanced Analytics"
    ])
    
    # Tab 1: ML Model Performance
    with tab1:
        st.subheader("🤖 Machine Learning Model Performance Comparison")
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model for Detailed Analysis",
            list(models.keys())
        )
        
        # Display metrics comparison
        st.markdown("### 📊 Model Performance Metrics")
        
        metrics_df = pd.DataFrame(metrics).T
        metrics_df = metrics_df.round(4)
        
        # Display metrics table
        st.dataframe(
            metrics_df.style.highlight_max(axis=0, color='lightgreen')
            .format("{:.4f}"),
            use_container_width=True
        )
        
        # Visualize metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart of metrics
            fig_metrics = go.Figure()
            
            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
                fig_metrics.add_trace(go.Bar(
                    name=metric.replace('_', ' ').title(),
                    x=list(metrics.keys()),
                    y=[metrics[model][metric] for model in metrics.keys()]
                ))
            
            fig_metrics.update_layout(
                title="Model Performance Comparison",
                xaxis_title="Model",
                yaxis_title="Score",
                barmode='group',
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        with col2:
            # Radar chart for selected model
            categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
            values = [
                metrics[selected_model]['accuracy'],
                metrics[selected_model]['precision'],
                metrics[selected_model]['recall'],
                metrics[selected_model]['f1_score'],
                metrics[selected_model]['roc_auc']
            ]
            
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=selected_model
            ))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title=f"{selected_model} - Performance Metrics",
                showlegend=False
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Confusion Matrix
        st.markdown("### 📈 Confusion Matrix")
        
        col1, col2, col3 = st.columns(3)
        
        for i, (model_name, model_tuple) in enumerate(models.items()):
            with [col1, col2, col3][i]:
                cm = confusion_matrix(y_test, predictions[model_name])
                
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Safe', 'At Risk'],
                    y=['Safe', 'At Risk'],
                    title=f"{model_name}",
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_cm, use_container_width=True)
        
        # ROC Curves
        st.markdown("### 📉 ROC Curves Comparison")
        
        fig_roc = go.Figure()
        
        for model_name in models.keys():
            fpr, tpr, _ = roc_curve(y_test, probabilities[model_name])
            auc_score = auc(fpr, tpr)
            
            fig_roc.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                name=f'{model_name} (AUC = {auc_score:.3f})',
                mode='lines'
            ))
        
        # Add diagonal line
        fig_roc.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))
        
        fig_roc.update_layout(
            title='ROC Curves - All Models',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=800,
            height=500
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        
        # Classification Report
        st.markdown(f"### 📋 Detailed Classification Report - {selected_model}")
        
        report = classification_report(
            y_test,
            predictions[selected_model],
            target_names=['Safe (Will Complete)', 'At Risk (Won\'t Complete)'],
            output_dict=True
        )
        
        report_df = pd.DataFrame(report).T
        st.dataframe(
            report_df.style.format("{:.3f}"),
            use_container_width=True
        )
    
    # Tab 2: Risk Prediction Dashboard
    with tab2:
        st.subheader("🎯 ML-Powered Risk Prediction Dashboard")
        
        # Select model for predictions
        prediction_model = st.selectbox(
            "Select Model for Risk Predictions",
            list(models.keys()),
            key='prediction_model'
        )
        
        # Generate predictions for all students
        model, scaler = models[prediction_model]
        
        # Apply same filters to X
        filtered_indices = filtered_df.index
        X_filtered = X.loc[filtered_indices]
        
        # Make predictions
        if scaler is not None:
            X_filtered_scaled = scaler.transform(X_filtered)
            risk_predictions = model.predict(X_filtered_scaled)
            risk_probabilities = model.predict_proba(X_filtered_scaled)[:, 1]
        else:
            risk_predictions = model.predict(X_filtered)
            risk_probabilities = model.predict_proba(X_filtered)[:, 1]
        
        # Add predictions to dataframe
        filtered_df_pred = filtered_df.copy()
        filtered_df_pred['ml_risk_prediction'] = risk_predictions
        filtered_df_pred['ml_risk_probability'] = risk_probabilities
        
        # Categorize risk levels based on probability
        filtered_df_pred['ml_risk_level'] = pd.cut(
            filtered_df_pred['ml_risk_probability'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        # Risk overview metrics
        st.markdown("### 🎯 ML Risk Prediction Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        high_risk_ml = len(filtered_df_pred[filtered_df_pred['ml_risk_level'] == 'High Risk'])
        medium_risk_ml = len(filtered_df_pred[filtered_df_pred['ml_risk_level'] == 'Medium Risk'])
        low_risk_ml = len(filtered_df_pred[filtered_df_pred['ml_risk_level'] == 'Low Risk'])
        avg_risk_prob = filtered_df_pred['ml_risk_probability'].mean()
        
        with col1:
            st.metric("🔴 High Risk (ML)", high_risk_ml,
                     delta=f"{high_risk_ml/len(filtered_df_pred)*100:.1f}%")
        with col2:
            st.metric("🟡 Medium Risk (ML)", medium_risk_ml,
                     delta=f"{medium_risk_ml/len(filtered_df_pred)*100:.1f}%")
        with col3:
            st.metric("🟢 Low Risk (ML)", low_risk_ml,
                     delta=f"{low_risk_ml/len(filtered_df_pred)*100:.1f}%")
        with col4:
            st.metric("📊 Avg Risk Probability", f"{avg_risk_prob:.2%}")
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # ML Risk distribution
            risk_dist_ml = filtered_df_pred['ml_risk_level'].value_counts()
            fig_ml_pie = px.pie(
                values=risk_dist_ml.values,
                names=risk_dist_ml.index,
                title="ML-Predicted Risk Distribution",
                color=risk_dist_ml.index,
                color_discrete_map={
                    'Low Risk': '#00CC96',
                    'Medium Risk': '#FFA500',
                    'High Risk': '#EF553B'
                }
            )
            st.plotly_chart(fig_ml_pie, use_container_width=True)
            
            # Risk probability distribution
            fig_prob_dist = px.histogram(
                filtered_df_pred,
                x='ml_risk_probability',
                nbins=30,
                title="Risk Probability Distribution",
                labels={'ml_risk_probability': 'Risk Probability'},
                color='ml_risk_level',
                color_discrete_map={
                    'Low Risk': '#00CC96',
                    'Medium Risk': '#FFA500',
                    'High Risk': '#EF553B'
                }
            )
            st.plotly_chart(fig_prob_dist, use_container_width=True)
        
        with col2:
            # Actual vs Predicted
            actual_completed = filtered_df_pred['Completed'].map({1: 'Completed', 0: 'Not Completed'})
            predicted_risk = filtered_df_pred['ml_risk_prediction'].map({0: 'Safe', 1: 'At Risk'})
            
            confusion_actual_pred = pd.crosstab(
                actual_completed,
                predicted_risk,
                normalize='all'
            ) * 100
            
            fig_actual_pred = px.imshow(
                confusion_actual_pred,
                text_auto='.1f',
                labels=dict(x="ML Prediction", y="Actual Status", color="Percentage (%)"),
                title="Actual Status vs ML Prediction",
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig_actual_pred, use_container_width=True)
            
            # Risk by course level
            risk_by_level_ml = pd.crosstab(
                filtered_df_pred['Course_Level'],
                filtered_df_pred['ml_risk_level'],
                normalize='index'
            ) * 100
            
            fig_level_ml = px.bar(
                risk_by_level_ml,
                title="ML Risk Distribution by Course Level",
                labels={'value': 'Percentage (%)', 'Course_Level': 'Course Level'},
                color_discrete_map={
                    'Low Risk': '#00CC96',
                    'Medium Risk': '#FFA500',
                    'High Risk': '#EF553B'
                },
                barmode='stack'
            )
            st.plotly_chart(fig_level_ml, use_container_width=True)
        
        # High-risk students table
        st.markdown("### 🚨 High-Risk Students - ML Prediction (Action Required)")
        
        high_risk_ml_students = filtered_df_pred[
            filtered_df_pred['ml_risk_level'] == 'High Risk'
        ].copy()
        
        if len(high_risk_ml_students) > 0:
            # Select columns for display
            display_cols = [
                'Student_ID', 'Name', 'Course_Name', 'Progress_Percentage',
                'Video_Completion_Rate', 'Quiz_Score_Avg', 'Assignments_Missed',
                'Days_Since_Last_Login', 'ml_risk_probability', 'ml_risk_level'
            ]
            
            # Filter existing columns
            display_cols = [col for col in display_cols if col in high_risk_ml_students.columns]
            
            high_risk_display = high_risk_ml_students[display_cols].sort_values(
                'ml_risk_probability', ascending=False
            )
            
            # Format the dataframe
            st.dataframe(
                high_risk_display.head(20).style.format({
                    'ml_risk_probability': '{:.2%}',
                    'Progress_Percentage': '{:.1f}%',
                    'Video_Completion_Rate': '{:.2f}',
                    'Quiz_Score_Avg': '{:.1f}'
                }).background_gradient(
                    subset=['ml_risk_probability'], cmap='Reds'
                ),
                use_container_width=True
            )
            
            # Download button
            csv_high_risk_ml = high_risk_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download ML High-Risk Students List",
                data=csv_high_risk_ml,
                file_name=f"ml_high_risk_students_{prediction_model.replace(' ', '_')}.csv",
                mime="text/csv"
            )
        else:
            st.success("✅ No high-risk students predicted by ML model!")
        
        # Model confidence analysis
        st.markdown("### 🎲 Prediction Confidence Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence distribution
            filtered_df_pred['confidence'] = filtered_df_pred['ml_risk_probability'].apply(
                lambda x: max(x, 1-x)
            )
            
            fig_confidence = px.histogram(
                filtered_df_pred,
                x='confidence',
                nbins=30,
                title="Model Prediction Confidence Distribution",
                labels={'confidence': 'Confidence Score'},
                color='ml_risk_level',
                color_discrete_map={
                    'Low Risk': '#00CC96',
                    'Medium Risk': '#FFA500',
                    'High Risk': '#EF553B'
                }
            )
            st.plotly_chart(fig_confidence, use_container_width=True)
        
        with col2:
            # High confidence predictions
            high_confidence = filtered_df_pred[filtered_df_pred['confidence'] > 0.8]
            low_confidence = filtered_df_pred[filtered_df_pred['confidence'] < 0.6]
            
            st.markdown("#### Confidence Statistics")
            st.metric("High Confidence Predictions (>80%)", len(high_confidence),
                     delta=f"{len(high_confidence)/len(filtered_df_pred)*100:.1f}%")
            st.metric("Low Confidence Predictions (<60%)", len(low_confidence),
                     delta=f"{len(low_confidence)/len(filtered_df_pred)*100:.1f}%")
            
            st.markdown("#### Model Reliability")
            st.write(f"**Average Confidence:** {filtered_df_pred['confidence'].mean():.2%}")
            st.write(f"**Median Confidence:** {filtered_df_pred['confidence'].median():.2%}")
            
            # Show students requiring manual review (low confidence)
            if len(low_confidence) > 0:
                st.warning(f"⚠️ {len(low_confidence)} students require manual review due to low prediction confidence")
    
    # Tab 3: Feature Importance
    with tab3:
        st.subheader("📊 Feature Importance Analysis")
        
        # Select model for feature importance
        fi_model_name = st.selectbox(
            "Select Model for Feature Importance",
            ['Random Forest', 'Gradient Boosting'],  # Only tree-based models
            key='fi_model'
        )
        
        model, _ = models[fi_model_name]
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_names = X.columns
            
            # Create dataframe
            fi_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            # Display top features
            st.markdown("### 🏆 Top 20 Most Important Features")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Bar chart
                fig_fi = px.bar(
                    fi_df.head(20),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f"Feature Importance - {fi_model_name}",
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                fig_fi.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_fi, use_container_width=True)
            
            with col2:
                st.dataframe(
                    fi_df.head(20).style.format({'Importance': '{:.4f}'}),
                    use_container_width=True,
                    height=600
                )
            
            # Feature importance categories
            st.markdown("### 📂 Feature Importance by Category")
            
            # Categorize features
            engagement_features = [f for f in feature_names if any(
                word in f.lower() for word in ['login', 'video', 'time', 'session', 'discussion', 'engagement']
            )]
            performance_features = [f for f in feature_names if any(
                word in f.lower() for word in ['quiz', 'assignment', 'progress', 'performance', 'score']
            )]
            demographic_features = [f for f in feature_names if any(
                word in f.lower() for word in ['age', 'gender', 'education', 'employment']
            )]
            financial_features = [f for f in feature_names if any(
                word in f.lower() for word in ['fee', 'payment', 'discount', 'amount']
            )]
            
            category_importance = {
                'Engagement': fi_df[fi_df['Feature'].isin(engagement_features)]['Importance'].sum(),
                'Performance': fi_df[fi_df['Feature'].isin(performance_features)]['Importance'].sum(),
                'Demographics': fi_df[fi_df['Feature'].isin(demographic_features)]['Importance'].sum(),
                'Financial': fi_df[fi_df['Feature'].isin(financial_features)]['Importance'].sum(),
                'Other': fi_df[~fi_df['Feature'].isin(
                    engagement_features + performance_features + demographic_features + financial_features
                )]['Importance'].sum()
            }
            
            fig_category = px.pie(
                values=list(category_importance.values()),
                names=list(category_importance.keys()),
                title="Feature Importance by Category",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_category, use_container_width=True)
            
            # Feature correlation with risk
            st.markdown("### 🔗 Feature Correlation with Risk")
            
            # Calculate correlations
            correlations = []
            for feature in fi_df.head(15)['Feature']:
                corr = X[feature].corr(y)
                correlations.append({
                    'Feature': feature,
                    'Correlation': abs(corr),
                    'Direction': 'Positive' if corr > 0 else 'Negative'
                })
            
            corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
            
            fig_corr = px.bar(
                corr_df,
                x='Correlation',
                y='Feature',
                orientation='h',
                color='Direction',
                title="Feature Correlation with Risk (Top 15 Important Features)",
                color_discrete_map={'Positive': '#EF553B', 'Negative': '#00CC96'}
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    
    # Tab 4: Individual Prediction
    with tab4:
        st.subheader("🔍 Individual Student Risk Prediction")
        
        st.markdown("""
        Enter student information below to get a real-time risk prediction from the ML model.
        This tool can be used for:
        - Assessing new student enrollment risk
        - Monitoring current student progress
        - Testing intervention scenarios
        """)
        
        # Select model
        individual_model = st.selectbox(
            "Select ML Model",
            list(models.keys()),
            key='individual_model'
        )
        
        st.markdown("---")
        
        # Create input form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 👤 Demographics")
            age = st.number_input("Age", min_value=18, max_value=70, value=25)
            gender = st.selectbox("Gender", df['Gender'].unique())
            education = st.selectbox("Education Level", df['Education_Level'].unique())
            employment = st.selectbox("Employment Status", df['Employment_Status'].unique())
        
        with col2:
            st.markdown("#### 📚 Course Information")
            course_level = st.selectbox("Course Level", df['Course_Level'].unique())
            instructor_rating = st.slider("Instructor Rating", 0.0, 5.0, 4.0, 0.1)
            course_duration = st.number_input("Course Duration (weeks)", 1, 52, 12)
            
        with col3:
            st.markdown("#### 💰 Payment")
            payment_mode = st.selectbox("Payment Mode", df['Payment_Mode'].unique())
            fee_paid = st.number_input("Fee Paid ($)", 0, 5000, 500)
            payment_amount = st.number_input("Payment Amount ($)", 0, 5000, 500)
            discount_used = st.selectbox("Discount Used", [0, 1])
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📊 Engagement Metrics")
            login_freq = st.slider("Login Frequency (per week)", 0, 50, 10)
            video_completion = st.slider("Video Completion Rate", 0.0, 1.0, 0.7, 0.01)
            time_spent = st.number_input("Time Spent (hours)", 0.0, 200.0, 30.0, 1.0)
            session_duration = st.slider("Avg Session Duration (min)", 0, 180, 45)
            discussion_participation = st.slider("Discussion Participation", 0, 50, 5)
        
        with col2:
            st.markdown("#### 🎯 Performance Metrics")
            progress = st.slider("Progress Percentage", 0, 100, 50)
            quiz_score = st.slider("Quiz Score Average", 0, 100, 70)
            assignments_missed = st.slider("Assignments Missed", 0, 10, 2)
        
        with col3:
            st.markdown("#### ⏰ Activity")
            days_since_login = st.slider("Days Since Last Login", 0, 30, 3)
        
        # Predict button
        if st.button("🔮 Predict Risk", type="primary", use_container_width=True):
            # Create input dataframe
            input_data = pd.DataFrame({
                'Age': [age],
                'Gender': [gender],
                'Education_Level': [education],
                'Employment_Status': [employment],
                'Course_Level': [course_level],
                'Instructor_Rating': [instructor_rating],
                'Course_Duration_Weeks': [course_duration],
                'Payment_Mode': [payment_mode],
                'Fee_Paid': [fee_paid],
                'Payment_Amount': [payment_amount],
                'Discount_Used': [discount_used],
                'Login_Frequency': [login_freq],
                'Video_Completion_Rate': [video_completion],
                'Time_Spent_Hours': [time_spent],
                'Average_Session_Duration_Min': [session_duration],
                'Discussion_Participation': [discussion_participation],
                'Progress_Percentage': [progress],
                'Quiz_Score_Avg': [quiz_score],
                'Assignments_Missed': [assignments_missed],
                'Days_Since_Last_Login': [days_since_login]
            })
            
            # Engineer features for input
            # Engagement score
            input_data['engagement_score'] = (
                input_data['Login_Frequency'] * 0.2 +
                input_data['Video_Completion_Rate'] * 30 +
                (input_data['Time_Spent_Hours'] / df['Time_Spent_Hours'].max()) * 100 * 0.2 +
                (input_data['Average_Session_Duration_Min'] / df['Average_Session_Duration_Min'].max()) * 100 * 0.15 +
                input_data['Discussion_Participation'] * 0.15
            )
            
            # Performance score
            input_data['performance_score'] = (
                input_data['Quiz_Score_Avg'] * 0.5 +
                input_data['Progress_Percentage'] * 0.3 +
                (10 - input_data['Assignments_Missed']) * 5 * 0.2
            )
            
            # Other engineered features
            input_data['activity_recency'] = 1 / (input_data['Days_Since_Last_Login'] + 1)
            input_data['video_time_ratio'] = input_data['Video_Completion_Rate'] / (input_data['Time_Spent_Hours'] + 1)
            input_data['assignment_completion_rate'] = ((10 - input_data['Assignments_Missed']) / 10) * 100
            input_data['session_effectiveness'] = (
                (input_data['Time_Spent_Hours'] / (input_data['Login_Frequency'] + 1)) *
                (input_data['Quiz_Score_Avg'] / 100)
            )
            
            # Binary flags
            input_data['low_engagement_flag'] = (input_data['Login_Frequency'] < df['Login_Frequency'].median()).astype(int)
            input_data['poor_performance_flag'] = (input_data['Quiz_Score_Avg'] < 60).astype(int)
            input_data['inactive_flag'] = (input_data['Days_Since_Last_Login'] > 7).astype(int)
            input_data['low_progress_flag'] = (input_data['Progress_Percentage'] < 50).astype(int)
            
            # Encode categorical features
            for cat in ['Gender', 'Education_Level', 'Employment_Status', 'Course_Level', 'Payment_Mode']:
                if cat in le_dict:
                    try:
                        input_data[cat] = le_dict[cat].transform(input_data[cat])
                    except:
                        # Handle unseen categories
                        input_data[cat] = 0
            
            # Ensure all features are present
            for col in X.columns:
                if col not in input_data.columns:
                    input_data[col] = 0
            
            # Reorder columns to match training data
            input_data = input_data[X.columns]
            
            # Make prediction
            model, scaler = models[individual_model]
            
            if scaler is not None:
                input_scaled = scaler.transform(input_data)
                risk_pred = model.predict(input_scaled)[0]
                risk_prob = model.predict_proba(input_scaled)[0, 1]
            else:
                risk_pred = model.predict(input_data)[0]
                risk_prob = model.predict_proba(input_data)[0, 1]
            
            # Determine risk level
            if risk_prob < 0.3:
                risk_level = "🟢 Low Risk"
                risk_color = "green"
            elif risk_prob < 0.7:
                risk_level = "🟡 Medium Risk"
                risk_color = "orange"
            else:
                risk_level = "🔴 High Risk"
                risk_color = "red"
            
            # Display results
            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Risk Level", risk_level)
            with col2:
                st.metric("Risk Probability", f"{risk_prob:.1%}")
            with col3:
                completion_prob = 1 - risk_prob
                st.metric("Completion Probability", f"{completion_prob:.1%}")
            
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Score", 'font': {'size': 24}},
                delta={'reference': 50, 'increasing': {'color': "red"}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': risk_color},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': '#d4edda'},
                        {'range': [30, 70], 'color': '#fff3cd'},
                        {'range': [70, 100], 'color': '#f8d7da'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            
            if risk_prob >= 0.7:
                st.error("""
                **🚨 IMMEDIATE ACTION REQUIRED**
                - Schedule urgent 1-on-1 mentoring session
                - Send personalized intervention email/SMS
                - Provide additional learning resources
                - Consider deadline extension
                - Connect with academic support services
                - Implement daily check-ins
                """)
            elif risk_prob >= 0.3:
                st.warning("""
                **⚠️ MONITORING RECOMMENDED**
                - Send reminder notifications
                - Encourage increased engagement
                - Provide study guides and tips
                - Suggest peer study groups
                - Schedule optional check-in
                - Monitor progress weekly
                """)
            else:
                st.success("""
                **✅ STUDENT ON TRACK**
                - Continue current support level
                - Send positive progress updates
                - Recognize achievements
                - Offer advanced challenges
                - Consider as peer mentor
                - Request feedback on course
                """)
    
    # Tab 5: Advanced Analytics
    with tab5:
        st.subheader("📈 Advanced Risk Analytics")
        
        # Add predictions to df_ml
        df_ml_full = df_ml.copy()
        
        # Generate predictions for all students
        best_model_name = max(metrics.items(), key=lambda x: x[1]['f1_score'])[0]
        model, scaler = models[best_model_name]
        
        if scaler is not None:
            X_scaled = scaler.transform(X)
            all_predictions = model.predict(X_scaled)
            all_probabilities = model.predict_proba(X_scaled)[:, 1]
        else:
            all_predictions = model.predict(X)
            all_probabilities = model.predict_proba(X)[:, 1]
        
        df_ml_full['ml_risk_prediction'] = all_predictions
        df_ml_full['ml_risk_probability'] = all_probabilities
        
        # Risk trends over time
        st.markdown(f"### 📊 Risk Analytics Dashboard (Using {best_model_name})")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk by age groups
            df_ml_full['age_group'] = pd.cut(
                df_ml_full['Age'],
                bins=[0, 25, 35, 45, 100],
                labels=['18-25', '26-35', '36-45', '46+']
            )
            
            risk_by_age = df_ml_full.groupby('age_group')['ml_risk_probability'].mean() * 100
            
            fig_age_risk = px.bar(
                x=risk_by_age.index,
                y=risk_by_age.values,
                title="Average Risk Probability by Age Group",
                labels={'x': 'Age Group', 'y': 'Avg Risk Probability (%)'},
                color=risk_by_age.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_age_risk, use_container_width=True)
            
            # Risk by education level
            risk_by_edu = df_ml_full.groupby('Education_Level')['ml_risk_probability'].mean().sort_values(ascending=False) * 100
            
            fig_edu_risk = px.bar(
                x=risk_by_edu.index,
                y=risk_by_edu.values,
                title="Average Risk Probability by Education Level",
                labels={'x': 'Education Level', 'y': 'Avg Risk Probability (%)'},
                color=risk_by_edu.values,
                color_continuous_scale='Oranges'
            )
            st.plotly_chart(fig_edu_risk, use_container_width=True)
        
        with col2:
            # Scatter: Engagement vs Performance colored by risk
            fig_scatter = px.scatter(
                df_ml_full.sample(500),  # Sample for performance
                x='engagement_score',
                y='performance_score',
                color='ml_risk_probability',
                size='ml_risk_probability',
                title="Engagement vs Performance (colored by ML Risk)",
                labels={
                    'engagement_score': 'Engagement Score',
                    'performance_score': 'Performance Score',
                    'ml_risk_probability': 'Risk Probability'
                },
                color_continuous_scale='RdYlGn_r',
                opacity=0.6
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Risk distribution by payment mode
            risk_by_payment = df_ml_full.groupby('Payment_Mode')['ml_risk_probability'].mean().sort_values(ascending=False) * 100
            
            fig_payment_risk = px.bar(
                x=risk_by_payment.index,
                y=risk_by_payment.values,
                title="Average Risk Probability by Payment Mode",
                labels={'x': 'Payment Mode', 'y': 'Avg Risk Probability (%)'},
                color=risk_by_payment.values,
                color_continuous_scale='Purples'
            )
            st.plotly_chart(fig_payment_risk, use_container_width=True)
        
        # Risk distribution heatmap
        st.markdown("### 🗺️ Risk Heatmap by Demographics")
        
        # Create pivot table
        risk_pivot = df_ml_full.pivot_table(
            values='ml_risk_probability',
            index='Education_Level',
            columns='Employment_Status',
            aggfunc='mean'
        ) * 100
        
        fig_heatmap = px.imshow(
            risk_pivot,
            text_auto='.1f',
            aspect='auto',
            title="Average Risk Probability (%) by Education & Employment",
            labels=dict(x="Employment Status", y="Education Level", color="Risk %"),
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Summary statistics
        st.markdown("### 📊 Risk Prediction Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Students Analyzed", len(df_ml_full))
        with col2:
            high_risk_total = len(df_ml_full[df_ml_full['ml_risk_probability'] > 0.7])
            st.metric("Total High Risk", high_risk_total,
                     delta=f"{high_risk_total/len(df_ml_full)*100:.1f}%")
        with col3:
            avg_risk = df_ml_full['ml_risk_probability'].mean()
            st.metric("Average Risk Score", f"{avg_risk:.2%}")
        with col4:
            model_accuracy = metrics[best_model_name]['accuracy']
            st.metric(f"{best_model_name} Accuracy", f"{model_accuracy:.1%}")

if __name__ == "__main__":
    main()
