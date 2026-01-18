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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Demographics", 
        "🎓 Course Analytics", 
        "⏱️ Engagement Metrics",
        "💰 Payment Analysis",
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
    
    # Tab 5: Data Explorer
    with tab5:
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
