import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Course Completion Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load data with caching
@st.cache_data
def load_data():
    df = pd.read_csv('Course_Completion_Prediction.csv')
    # Convert Completed column to binary (1 for Completed, 0 for Not Completed)
    df['Completed'] = (df['Completed'] == 'Completed').astype(int)
    return df

# Main app
def main():
    st.title("📊 Course Completion Prediction - Data Visualization")
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
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Demographics", 
        "🎓 Course Analytics", 
        "⏱️ Engagement Metrics",
        "💰 Payment Analysis",
        "⚠️ Risk Prediction",
        "🔍 Data Explorer"
    ])
    
    # Tab 1: Demographics
    with tab1:
        st.subheader("Student Demographics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender distribution
            gender_dist = filtered_df['Gender'].value_counts()
            fig_gender = px.pie(
                values=gender_dist.values,
                names=gender_dist.index,
                title="Gender Distribution",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_gender, use_container_width=True)
            
            # Age distribution
            fig_age = px.histogram(
                filtered_df,
                x='Age',
                nbins=30,
                title="Age Distribution",
                color_discrete_sequence=['#636EFA']
            )
            fig_age.update_layout(bargap=0.1)
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            # Education Level
            education_dist = filtered_df['Education_Level'].value_counts()
            fig_education = px.bar(
                x=education_dist.index,
                y=education_dist.values,
                title="Education Level Distribution",
                labels={'x': 'Education Level', 'y': 'Count'},
                color_discrete_sequence=['#EF553B']
            )
            st.plotly_chart(fig_education, use_container_width=True)
            
            # Employment Status
            employment_dist = filtered_df['Employment_Status'].value_counts()
            fig_employment = px.bar(
                x=employment_dist.index,
                y=employment_dist.values,
                title="Employment Status Distribution",
                labels={'x': 'Employment Status', 'y': 'Count'},
                color_discrete_sequence=['#00CC96']
            )
            st.plotly_chart(fig_employment, use_container_width=True)
    
    # Tab 2: Course Analytics
    with tab2:
        st.subheader("Course Performance Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Completion rate by course level
            completion_by_level = filtered_df.groupby('Course_Level')['Completed'].mean() * 100
            fig_level = px.bar(
                x=completion_by_level.index,
                y=completion_by_level.values,
                title="Completion Rate by Course Level",
                labels={'x': 'Course Level', 'y': 'Completion Rate (%)'},
                color_discrete_sequence=['#AB63FA']
            )
            st.plotly_chart(fig_level, use_container_width=True)
            
            # Top categories by enrollment
            category_dist = filtered_df['Category'].value_counts().head(10)
            fig_category = px.bar(
                x=category_dist.values,
                y=category_dist.index,
                orientation='h',
                title="Top 10 Course Categories",
                labels={'x': 'Number of Students', 'y': 'Category'},
                color_discrete_sequence=['#FFA15A']
            )
            st.plotly_chart(fig_category, use_container_width=True)
        
        with col2:
            # Progress percentage distribution
            fig_progress = px.histogram(
                filtered_df,
                x='Progress_Percentage',
                nbins=20,
                title="Progress Percentage Distribution",
                color_discrete_sequence=['#19D3F3']
            )
            st.plotly_chart(fig_progress, use_container_width=True)
            
            # Instructor Rating vs Completion
            fig_instructor = px.box(
                filtered_df,
                x='Completed',
                y='Instructor_Rating',
                title="Instructor Rating vs Course Completion",
                labels={'Completed': 'Course Completed', 'Instructor_Rating': 'Instructor Rating'},
                color='Completed',
                color_discrete_map={0: '#EF553B', 1: '#00CC96'}
            )
            st.plotly_chart(fig_instructor, use_container_width=True)
    
    # Tab 3: Engagement Metrics
    with tab3:
        st.subheader("Student Engagement Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Login frequency vs completion
            fig_login = px.box(
                filtered_df,
                x='Completed',
                y='Login_Frequency',
                title="Login Frequency vs Completion",
                labels={'Completed': 'Course Completed', 'Login_Frequency': 'Login Frequency'},
                color='Completed',
                color_discrete_map={0: '#EF553B', 1: '#00CC96'}
            )
            st.plotly_chart(fig_login, use_container_width=True)
            
            # Time spent hours vs completion
            fig_time = px.box(
                filtered_df,
                x='Completed',
                y='Time_Spent_Hours',
                title="Time Spent (Hours) vs Completion",
                labels={'Completed': 'Course Completed', 'Time_Spent_Hours': 'Time Spent (Hours)'},
                color='Completed',
                color_discrete_map={0: '#EF553B', 1: '#00CC96'}
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col2:
            # Video completion rate
            fig_video = px.histogram(
                filtered_df,
                x='Video_Completion_Rate',
                nbins=20,
                title="Video Completion Rate Distribution",
                color='Completed',
                color_discrete_map={0: '#EF553B', 1: '#00CC96'},
                barmode='overlay'
            )
            fig_video.update_traces(opacity=0.7)
            st.plotly_chart(fig_video, use_container_width=True)
            
            # Quiz score average
            fig_quiz = px.box(
                filtered_df,
                x='Completed',
                y='Quiz_Score_Avg',
                title="Average Quiz Score vs Completion",
                labels={'Completed': 'Course Completed', 'Quiz_Score_Avg': 'Average Quiz Score'},
                color='Completed',
                color_discrete_map={0: '#EF553B', 1: '#00CC96'}
            )
            st.plotly_chart(fig_quiz, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Engagement Metrics Correlation")
        engagement_cols = [
            'Login_Frequency', 'Average_Session_Duration_Min', 
            'Video_Completion_Rate', 'Time_Spent_Hours',
            'Quiz_Score_Avg', 'Progress_Percentage', 'Completed'
        ]
        corr_matrix = filtered_df[engagement_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect='auto',
            title="Correlation Heatmap - Engagement Metrics",
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Tab 4: Payment Analysis
    with tab4:
        st.subheader("Payment and Financial Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Payment mode distribution
            payment_dist = filtered_df['Payment_Mode'].value_counts()
            fig_payment = px.pie(
                values=payment_dist.values,
                names=payment_dist.index,
                title="Payment Mode Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_payment, use_container_width=True)
            
            # Fee paid vs completion
            fig_fee = px.box(
                filtered_df,
                x='Completed',
                y='Fee_Paid',
                title="Fee Paid vs Course Completion",
                labels={'Completed': 'Course Completed', 'Fee_Paid': 'Fee Paid'},
                color='Completed',
                color_discrete_map={0: '#EF553B', 1: '#00CC96'}
            )
            st.plotly_chart(fig_fee, use_container_width=True)
        
        with col2:
            # Discount usage
            discount_completion = filtered_df.groupby('Discount_Used')['Completed'].mean() * 100
            fig_discount = px.bar(
                x=discount_completion.index,
                y=discount_completion.values,
                title="Completion Rate by Discount Usage",
                labels={'x': 'Discount Used', 'y': 'Completion Rate (%)'},
                color_discrete_sequence=['#B6E880']
            )
            st.plotly_chart(fig_discount, use_container_width=True)
            
            # Payment amount distribution
            fig_amount = px.histogram(
                filtered_df,
                x='Payment_Amount',
                nbins=30,
                title="Payment Amount Distribution",
                color='Completed',
                color_discrete_map={0: '#EF553B', 1: '#00CC96'},
                barmode='overlay'
            )
            fig_amount.update_traces(opacity=0.7)
            st.plotly_chart(fig_amount, use_container_width=True)
    
    # Tab 5: Risk Prediction Panel
    with tab5:
        st.subheader("⚠️ Student At-Risk Analysis & Prediction")
        
        # Calculate risk scores
        filtered_df['risk_score'] = 0
        
        # Risk factors calculation
        # Low engagement indicators
        filtered_df.loc[filtered_df['Login_Frequency'] < filtered_df['Login_Frequency'].median(), 'risk_score'] += 1
        filtered_df.loc[filtered_df['Video_Completion_Rate'] < 0.5, 'risk_score'] += 2
        filtered_df.loc[filtered_df['Time_Spent_Hours'] < filtered_df['Time_Spent_Hours'].median(), 'risk_score'] += 1
        filtered_df.loc[filtered_df['Average_Session_Duration_Min'] < 30, 'risk_score'] += 1
        
        # Academic performance indicators
        filtered_df.loc[filtered_df['Quiz_Score_Avg'] < 60, 'risk_score'] += 2
        filtered_df.loc[filtered_df['Assignments_Missed'] > 2, 'risk_score'] += 2
        filtered_df.loc[filtered_df['Progress_Percentage'] < 50, 'risk_score'] += 2
        
        # Participation indicators
        filtered_df.loc[filtered_df['Discussion_Participation'] == 0, 'risk_score'] += 1
        filtered_df.loc[filtered_df['Days_Since_Last_Login'] > 7, 'risk_score'] += 2
        
        # Categorize risk levels
        filtered_df['risk_level'] = pd.cut(
            filtered_df['risk_score'], 
            bins=[-1, 3, 6, 15], 
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        # Risk overview metrics
        st.markdown("### 🎯 Risk Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        high_risk_count = len(filtered_df[filtered_df['risk_level'] == 'High Risk'])
        medium_risk_count = len(filtered_df[filtered_df['risk_level'] == 'Medium Risk'])
        low_risk_count = len(filtered_df[filtered_df['risk_level'] == 'Low Risk'])
        at_risk_rate = ((high_risk_count + medium_risk_count) / len(filtered_df) * 100)
        
        with col1:
            st.metric("🔴 High Risk", high_risk_count, 
                     delta=f"{high_risk_count/len(filtered_df)*100:.1f}%")
        with col2:
            st.metric("🟡 Medium Risk", medium_risk_count,
                     delta=f"{medium_risk_count/len(filtered_df)*100:.1f}%")
        with col3:
            st.metric("🟢 Low Risk", low_risk_count,
                     delta=f"{low_risk_count/len(filtered_df)*100:.1f}%")
        with col4:
            st.metric("⚠️ At-Risk Rate", f"{at_risk_rate:.1f}%")
        
        st.markdown("---")
        
        # Risk distribution visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk level distribution pie chart
            risk_dist = filtered_df['risk_level'].value_counts()
            fig_risk_pie = px.pie(
                values=risk_dist.values,
                names=risk_dist.index,
                title="Student Risk Distribution",
                color=risk_dist.index,
                color_discrete_map={
                    'Low Risk': '#00CC96',
                    'Medium Risk': '#FFA500',
                    'High Risk': '#EF553B'
                }
            )
            st.plotly_chart(fig_risk_pie, use_container_width=True)
            
            # Risk by course level
            risk_by_level = pd.crosstab(
                filtered_df['Course_Level'], 
                filtered_df['risk_level'],
                normalize='index'
            ) * 100
            
            fig_level = px.bar(
                risk_by_level,
                title="Risk Distribution by Course Level (%)",
                labels={'value': 'Percentage (%)', 'Course_Level': 'Course Level'},
                color_discrete_map={
                    'Low Risk': '#00CC96',
                    'Medium Risk': '#FFA500',
                    'High Risk': '#EF553B'
                },
                barmode='stack'
            )
            st.plotly_chart(fig_level, use_container_width=True)
        
        with col2:
            # Risk score distribution
            fig_score_dist = px.histogram(
                filtered_df,
                x='risk_score',
                title="Risk Score Distribution",
                labels={'risk_score': 'Risk Score', 'count': 'Number of Students'},
                color='risk_level',
                color_discrete_map={
                    'Low Risk': '#00CC96',
                    'Medium Risk': '#FFA500',
                    'High Risk': '#EF553B'
                }
            )
            st.plotly_chart(fig_score_dist, use_container_width=True)
            
            # Key risk factors
            st.markdown("#### 🔍 Key Risk Factors")
            risk_factors = {
                '🔴 Critical (2 points)': [
                    'Video Completion < 50%',
                    'Quiz Average < 60',
                    'Missed > 2 Assignments',
                    'Progress < 50%',
                    'Last Login > 7 days ago'
                ],
                '🟡 Moderate (1 point)': [
                    'Low Login Frequency',
                    'Low Time Spent',
                    'Short Session Duration',
                    'No Discussion Participation'
                ]
            }
            
            for severity, factors in risk_factors.items():
                with st.expander(severity):
                    for factor in factors:
                        st.write(f"• {factor}")
        
        st.markdown("---")
        
        # High-risk students table
        st.markdown("### 🚨 High-Risk Students (Action Required)")
        
        high_risk_students = filtered_df[filtered_df['risk_level'] == 'High Risk'].copy()
        
        if len(high_risk_students) > 0:
            # Select relevant columns for display
            display_cols = [
                'Student_ID', 'Name', 'Course_Name', 'Progress_Percentage',
                'Video_Completion_Rate', 'Quiz_Score_Avg', 'Assignments_Missed',
                'Days_Since_Last_Login', 'risk_score', 'risk_level'
            ]
            
            # Filter only existing columns
            display_cols = [col for col in display_cols if col in high_risk_students.columns]
            
            high_risk_display = high_risk_students[display_cols].sort_values(
                'risk_score', ascending=False
            )
            
            st.dataframe(
                high_risk_display.head(20).style.background_gradient(
                    subset=['risk_score'], cmap='Reds'
                ),
                use_container_width=True
            )
            
            # Download high-risk students
            csv_high_risk = high_risk_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download High-Risk Students List",
                data=csv_high_risk,
                file_name="high_risk_students.csv",
                mime="text/csv"
            )
        else:
            st.success("✅ No high-risk students found!")
        
        st.markdown("---")
        
        # Intervention recommendations
        st.markdown("### 💡 Recommended Interventions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🔴 High Risk")
            st.markdown("""
            - **Immediate Action Required**
            - Send personalized email/SMS
            - Schedule 1-on-1 mentoring session
            - Provide additional resources
            - Offer extension on deadlines
            - Connect with academic support
            """)
        
        with col2:
            st.markdown("#### 🟡 Medium Risk")
            st.markdown("""
            - **Monitor Closely**
            - Send reminder notifications
            - Encourage discussion participation
            - Provide study guides
            - Suggest study groups
            - Share success tips
            """)
        
        with col3:
            st.markdown("#### 🟢 Low Risk")
            st.markdown("""
            - **Keep Engaged**
            - Send progress updates
            - Recognize achievements
            - Offer advanced challenges
            - Request peer mentoring
            - Gather feedback
            """)
    
    # Tab 6: Data Explorer
    with tab6:
        st.subheader("Raw Data Explorer")
        
        # Display sample data
        st.write(f"Showing {len(filtered_df)} records")
        st.dataframe(filtered_df.head(100), use_container_width=True)
        
        # Download button
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name="filtered_course_data.csv",
            mime="text/csv"
        )
        
        # Summary statistics
        st.subheader("Summary Statistics")
        st.dataframe(filtered_df.describe(), use_container_width=True)

if __name__ == "__main__":
    main()
