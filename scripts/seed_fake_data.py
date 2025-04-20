#!/usr/bin/env python3
"""
Generate synthetic job application and outcome data for demonstration purposes.
This script creates a CSV file with realistic job search scenarios.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_job_data(num_records=100):
    """Generate synthetic job application data."""
    
    # Seed for reproducibility
    np.random.seed(42)
    
    # Company names
    companies = [
        "TechGiant Inc.", "DataVerse Systems", "Cloud Nine Solutions", 
        "Algorithm Associates", "Neural Networks Ltd.", "Quantum Computing Co.",
        "ByteWise Technologies", "SiliconValley Startups", "AImagination Tech",
        "CodeCrafters International", "ML Innovations", "FutureSoft",
        "DevOps Dynamics", "CyberSafe Solutions", "BlockChain Ventures"
    ]
    
    # Job titles
    titles = [
        "Software Engineer", "Data Scientist", "ML Engineer",
        "DevOps Engineer", "Data Engineer", "Backend Developer",
        "Frontend Developer", "Full Stack Developer", "Product Manager",
        "QA Engineer", "SRE", "Cloud Architect"
    ]
    
    # Application sources
    sources = ["LinkedIn", "Indeed", "Company Website", "Referral", "Job Fair", "Recruiter"]
    
    # Levels of experience required
    experience_levels = ["Entry Level", "Mid-Level", "Senior", "Lead", "Principal"]
    
    # Now generate the data
    today = datetime.now()
    
    data = {
        "company": np.random.choice(companies, num_records),
        "job_title": np.random.choice(titles, num_records),
        "experience_level": np.random.choice(experience_levels, num_records),
        "application_source": np.random.choice(sources, num_records),
        "date_applied": [(today - timedelta(days=np.random.randint(1, 180))).strftime("%Y-%m-%d") 
                          for _ in range(num_records)],
        "resume_tailored": np.random.choice([True, False], num_records),
        "cover_letter": np.random.choice([True, False], num_records),
        "referral": np.random.choice([True, False], num_records, p=[0.3, 0.7]),
        "remote": np.random.choice([True, False], num_records),
        "interview_rounds": np.random.randint(0, 6, num_records),
        "days_to_response": np.random.choice([None, *range(1, 31)], num_records, 
                                            p=[0.2, *[0.8/30 for _ in range(30)]]),
    }
    
    # Outcome depends partly on referral, tailored resume, etc.
    outcome_probs = []
    stages = ["Rejected", "Phone Screen", "Technical Interview", "Onsite", "Offer", "Accepted"]
    
    for i in range(num_records):
        # Base probability
        base_p = [0.5, 0.2, 0.15, 0.08, 0.05, 0.02]
        
        # Adjust for referral
        if data["referral"][i]:
            base_p = [0.3, 0.25, 0.2, 0.12, 0.08, 0.05]
            
        # Adjust for tailored resume
        if data["resume_tailored"][i]:
            base_p = [max(0.25, base_p[0]-0.1), *[min(p+0.02, 1.0) for p in base_p[1:]]]
            
        # Adjust for cover letter
        if data["cover_letter"][i]:
            base_p = [max(0.3, base_p[0]-0.05), *[min(p+0.01, 1.0) for p in base_p[1:]]]
            
        # If no response yet (days_to_response is None), mark as pending
        if data["days_to_response"][i] is None:
            base_p = [0, 0, 0, 0, 0, 0]  # Will be converted to "Pending"
            
        outcome_probs.append(base_p)
    
    # Assign outcomes based on the calculated probabilities
    data["current_stage"] = [
        "Pending" if all(p == 0 for p in probs) else
        np.random.choice(stages, p=probs)
        for probs in outcome_probs
    ]
    
    # Create a dataframe and return
    df = pd.DataFrame(data)
    
    # Add salary where offers exist
    df["offered_salary"] = np.where(
        df["current_stage"].isin(["Offer", "Accepted"]),
        np.random.randint(80000, 200000, num_records),
        np.nan
    )
    
    return df

def main():
    """Generate and save synthetic job application data."""
    # Create scripts directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Generate data
    df = generate_job_data(150)
    
    # Save to CSV
    output_path = "data/sample_job_applications.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} job application records in {output_path}")
    
    # Print sample statistics for verification
    print("\nSample statistics:")
    print(f"Application success rate: {(df['current_stage'] == 'Offer').mean():.1%}")
    print(f"Applications with referrals: {df['referral'].mean():.1%}")
    print(f"Median days to response: {df['days_to_response'].median()}")
    
    return output_path

if __name__ == "__main__":
    main() 