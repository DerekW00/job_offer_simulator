"""Enhanced Streamlit application with visualization."""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import sys
import json
from datetime import datetime

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulation import SimulationParameters, JobOfferSimulator, load_historical_data

# Set page config
st.set_page_config(
    page_title="Job Offer Simulator",
    page_icon="🏹",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("Job Offer Probability Simulator 🏹")
    st.markdown("""
    This tool estimates your probability of receiving job offers using Monte Carlo simulation.
    Adjust the parameters below to see how different factors affect your job search outcomes.
    """)
    
    # Sidebar with parameters
    st.sidebar.header("Simulation Parameters")
    
    # Time horizon settings
    time_horizon = st.sidebar.slider(
        "Time Horizon (days)", 
        min_value=30, 
        max_value=365, 
        value=90, 
        step=30,
        help="How many days into the future to simulate"
    )
    
    apps_per_week = st.sidebar.slider(
        "Applications per Week", 
        min_value=1.0, 
        max_value=20.0, 
        value=5.0, 
        step=1.0,
        help="Average number of job applications you send per week"
    )
    
    # Job seeker profile
    st.sidebar.subheader("Your Profile")
    
    resume_quality = st.sidebar.slider(
        "Resume Quality", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.7, 
        step=0.1,
        help="Quality of your resume (higher is better)"
    )
    
    interview_prep = st.sidebar.slider(
        "Interview Preparation", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.7, 
        step=0.1,
        help="How well-prepared you are for interviews"
    )
    
    referral_ratio = st.sidebar.slider(
        "Referral Percentage", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.3, 
        step=0.1,
        help="Percentage of applications that have an employee referral"
    )
    
    # Market conditions
    st.sidebar.subheader("Market Factors")
    
    industry_exp = st.sidebar.slider(
        "Industry Experience", 
        min_value=-0.5, 
        max_value=0.5, 
        value=0.0, 
        step=0.1,
        help="Your experience level compared to job requirements (-0.5: underqualified, 0.5: overqualified)"
    )
    
    skill_match = st.sidebar.slider(
        "Skill Match", 
        min_value=-0.5, 
        max_value=0.5, 
        value=0.0, 
        step=0.1,
        help="How well your skills match job requirements (-0.5: poor match, 0.5: excellent match)"
    )
    
    market_conditions = st.sidebar.slider(
        "Market Conditions", 
        min_value=-0.5, 
        max_value=0.5, 
        value=0.0, 
        step=0.1,
        help="Current job market conditions (-0.5: recession, 0.5: hot market)"
    )
    
    num_simulations = st.sidebar.selectbox(
        "Number of Simulations",
        options=[100, 500, 1000, 5000, 10000],
        index=2,
        help="Higher numbers give more accurate results but take longer to compute"
    )
    
    # Optional historical data
    st.sidebar.subheader("Data Options")
    
    use_sample_data = st.sidebar.checkbox(
        "Use Sample Data", 
        value=True,
        help="Use built-in sample data instead of uploading your own"
    )
    
    data_file = None
    if not use_sample_data:
        uploaded_file = st.sidebar.file_uploader(
            "Upload Job Application CSV", 
            type="csv",
            help="Upload your job application history in CSV format"
        )
        if uploaded_file is not None:
            # Save the uploaded file temporarily
            with open("temp_data.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            data_file = "temp_data.csv"
    else:
        # Use sample data from the generator
        data_file = "data/sample_job_applications.csv"
        if not os.path.exists(data_file):
            st.warning("Sample data file not found. Please run scripts/seed_fake_data.py first.")
    
    # Load data for analysis tab
    historical_data = None
    if data_file and os.path.exists(data_file):
        historical_data = load_historical_data(data_file)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Simulation Results", "Historical Data", "About"])
    
    with tab1:
        # Run simulation button
        if st.button("Run Simulation", type="primary"):
            with st.spinner("Running simulation..."):
                # Set up parameters
                params = SimulationParameters(
                    num_simulations=num_simulations,
                    time_horizon_days=time_horizon,
                    applications_per_week=apps_per_week,
                    resume_quality=resume_quality,
                    interview_prep_level=interview_prep,
                    referral_ratio=referral_ratio,
                    industry_experience_modifier=industry_exp,
                    skill_match_modifier=skill_match,
                    market_condition_modifier=market_conditions
                )
                
                # Run simulation
                simulator = JobOfferSimulator(params, historical_data)
                results = simulator.run_simulation()
                
                # Display results
                display_simulation_results(results, time_horizon)
    
    with tab2:
        if historical_data is not None:
            display_historical_data(historical_data)
        else:
            st.info("No historical data available. Please upload a CSV file or use sample data.")
    
    with tab3:
        display_about_info()

def display_simulation_results(results, time_horizon):
    """Display simulation results with visualizations."""
    
    # Key metrics in a nice formatted way
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Probability of Getting an Offer",
            f"{results['probability_at_least_one_offer']:.1%}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Expected Number of Offers",
            f"{results['expected_num_offers']:.2f}",
            delta=None
        )
    
    with col3:
        if results['days_to_first_offer_stats']['median'] is not None:
            median_days = results['days_to_first_offer_stats']['median']
            st.metric(
                "Median Days to First Offer",
                f"{median_days:.0f}",
                delta=None
            )
        else:
            st.metric(
                "Median Days to First Offer",
                "N/A",
                delta=None
            )
    
    # Timeline chart - Probability of receiving an offer by day
    st.subheader("Offer Probability Timeline")
    
    timeline_data = pd.DataFrame({
        'Day': [int(k.split('_')[1]) for k in results['offers_by_day'].keys()],
        'Probability': list(results['offers_by_day'].values())
    })
    
    # Sort by day
    timeline_data = timeline_data.sort_values('Day')
    
    # Create Altair chart
    timeline_chart = alt.Chart(timeline_data).mark_line(point=True).encode(
        x=alt.X('Day:Q', title='Days from Now'),
        y=alt.Y('Probability:Q', title='Probability of Receiving an Offer', scale=alt.Scale(domain=[0, 1])),
        tooltip=['Day:Q', alt.Tooltip('Probability:Q', format='.1%')]
    ).properties(
        height=300
    ).interactive()
    
    st.altair_chart(timeline_chart, use_container_width=True)
    
    # Distribution of number of offers
    st.subheader("Offer Count Distribution")
    
    # Convert distribution to DataFrame
    dist_data = pd.DataFrame({
        'Number of Offers': list(results['offer_count_distribution'].keys()),
        'Probability': list(results['offer_count_distribution'].values())
    })
    
    # Convert 'Number of Offers' to integer
    dist_data['Number of Offers'] = dist_data['Number of Offers'].astype(int)
    
    # Sort by number of offers
    dist_data = dist_data.sort_values('Number of Offers')
    
    # Create Altair chart for distribution
    dist_chart = alt.Chart(dist_data).mark_bar().encode(
        x=alt.X('Number of Offers:O', title='Number of Offers'),
        y=alt.Y('Probability:Q', title='Probability', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('Number of Offers:O', legend=None),
        tooltip=['Number of Offers:O', alt.Tooltip('Probability:Q', format='.1%')]
    ).properties(
        height=300
    ).interactive()
    
    st.altair_chart(dist_chart, use_container_width=True)
    
    # Add explanatory text
    st.markdown(f"""
    **Interpretation:** 
    
    Based on the simulation of your job search over {time_horizon} days:
    
    - You have a **{results['probability_at_least_one_offer']:.1%} chance** of receiving at least one job offer
    - You can expect to receive **{results['expected_num_offers']:.1f} offers** on average
    - The first offer typically arrives after **{results['days_to_first_offer_stats']['median'] or 'N/A'} days** (median)
    
    The timeline chart shows how your chances of receiving at least one offer increase over time.
    """)

def display_historical_data(data):
    """Display and analyze historical job application data."""
    
    st.header("Historical Data Analysis")
    
    # Show the data table
    with st.expander("View Raw Data"):
        st.dataframe(data)
    
    # Basic statistics
    total_applications = len(data)
    
    # Current stage distribution
    st.subheader("Application Status Distribution")
    
    # Count by current stage
    stage_counts = data['current_stage'].value_counts().reset_index()
    stage_counts.columns = ['Stage', 'Count']
    
    # Calculate percentages
    stage_counts['Percentage'] = stage_counts['Count'] / total_applications
    
    # Create a bar chart
    stage_chart = alt.Chart(stage_counts).mark_bar().encode(
        x=alt.X('Stage:N', sort='-y', title=None),
        y=alt.Y('Count:Q', title='Number of Applications'),
        color=alt.Color('Stage:N', legend=None),
        tooltip=['Stage:N', 'Count:Q', alt.Tooltip('Percentage:Q', format='.1%')]
    ).properties(
        height=300
    )
    
    st.altair_chart(stage_chart, use_container_width=True)
    
    # Application source analysis
    if 'application_source' in data.columns:
        st.subheader("Application Source Analysis")
        
        # Count by source
        source_counts = data['application_source'].value_counts().reset_index()
        source_counts.columns = ['Source', 'Count']
        
        # Create a pie chart
        source_chart = alt.Chart(source_counts).mark_arc().encode(
            theta=alt.Theta('Count:Q'),
            color=alt.Color('Source:N'),
            tooltip=['Source:N', 'Count:Q']
        ).properties(
            height=300
        )
        
        st.altair_chart(source_chart, use_container_width=True)
    
    # Response time analysis
    if 'days_to_response' in data.columns:
        st.subheader("Response Time Analysis")
        
        # Filter out rows where days_to_response is null
        response_data = data[data['days_to_response'].notna()]
        
        if not response_data.empty:
            # Create a histogram
            response_chart = alt.Chart(response_data).mark_bar().encode(
                x=alt.X('days_to_response:Q', bin=alt.Bin(maxbins=20), title='Days to Response'),
                y=alt.Y('count()', title='Number of Applications'),
                tooltip=[alt.Tooltip('count()', title='Applications'), 
                         alt.Tooltip('days_to_response:Q', title='Days')]
            ).properties(
                height=300
            )
            
            st.altair_chart(response_chart, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Response Time", f"{response_data['days_to_response'].mean():.1f} days")
            with col2:
                st.metric("Median Response Time", f"{response_data['days_to_response'].median():.1f} days")
            with col3:
                st.metric("Maximum Response Time", f"{response_data['days_to_response'].max():.0f} days")

def display_about_info():
    """Display information about the simulator."""
    
    st.header("About the Job Offer Simulator")
    
    st.markdown("""
    ### How It Works
    
    This simulator uses Monte Carlo methods to model the job application process:
    
    1. **Monte Carlo Simulation**: We run thousands of simulated job searches based on your parameters
    2. **Probability Chain**: Each application goes through a series of stages (application → screen → interview → etc.)
    3. **Influencing Factors**: Your resume, interview skills, referrals, and market conditions affect the probabilities
    
    ### Key Factors That Impact Results
    
    - **Applications per Week**: More applications generally improve your odds
    - **Resume Quality**: A strong resume increases your callback rate
    - **Interview Preparation**: Better preparation improves your conversion to offers
    - **Referrals**: Applications with referrals have significantly higher success rates
    - **Skill Match**: How well your skills match job requirements
    - **Market Conditions**: The overall job market affects all transition probabilities
    
    ### Limitations
    
    - This is a probabilistic model, not a guarantee of outcomes
    - The simulation assumes a stationary process (probabilities don't change over time)
    - Real-world job searches involve many intangible factors not captured in the model
    
    ### About the Project
    
    This project was created as an open-source tool to help job seekers understand the probabilistic nature of the job search process. The code is available on GitHub under the MIT license.
    """)

    st.markdown("---")
    st.markdown(f"© {datetime.now().year} | Version 0.1.0-alpha | Last updated: April 19, 2025")

if __name__ == "__main__":
    main()
