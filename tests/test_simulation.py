"""Tests for the simulation module."""

import pytest
import numpy as np
import pandas as pd
from src.simulation import SimulationParameters, JobOfferSimulator

def test_simulation_parameters_defaults():
    """Test that SimulationParameters has the expected defaults."""
    params = SimulationParameters()
    
    assert params.num_simulations == 1000
    assert params.time_horizon_days == 90
    assert params.applications_per_week == 5.0
    assert params.resume_quality == 0.7
    assert params.interview_prep_level == 0.7
    assert params.referral_ratio == 0.3
    
    # Check that the application sources sum to 1.0
    assert abs(sum(params.application_sources.values()) - 1.0) < 0.001

def test_simulation_results_shape():
    """Test that simulation results have the expected structure."""
    np.random.seed(42)  # For reproducibility
    
    simulator = JobOfferSimulator(SimulationParameters(num_simulations=100))
    results = simulator.run_simulation()
    
    # Check that the results contain the expected keys
    assert "probability_at_least_one_offer" in results
    assert "expected_num_offers" in results
    assert "days_to_first_offer_stats" in results
    assert "offer_count_distribution" in results
    assert "offers_by_day" in results
    
    # Check that the probability is between 0 and 1
    assert 0 <= results["probability_at_least_one_offer"] <= 1
    
    # Check that the expected number of offers is non-negative
    assert results["expected_num_offers"] >= 0

def test_probability_adjustments():
    """Test that modifying parameters affects the outcome probabilities."""
    np.random.seed(42)  # For reproducibility
    
    # Base case
    base_params = SimulationParameters(
        num_simulations=500,
        applications_per_week=5.0,
        resume_quality=0.5,
        interview_prep_level=0.5,
        referral_ratio=0.3
    )
    base_simulator = JobOfferSimulator(base_params)
    base_results = base_simulator.run_simulation()
    
    # Improved case
    improved_params = SimulationParameters(
        num_simulations=500,
        applications_per_week=5.0,
        resume_quality=0.9,  # Better resume
        interview_prep_level=0.9,  # Better interview prep
        referral_ratio=0.7    # More referrals
    )
    improved_simulator = JobOfferSimulator(improved_params)
    improved_results = improved_simulator.run_simulation()
    
    # The probability should be higher with better parameters
    assert improved_results["probability_at_least_one_offer"] > base_results["probability_at_least_one_offer"]
    assert improved_results["expected_num_offers"] > base_results["expected_num_offers"]

def test_deterministic_with_seed():
    """Test that simulations with the same seed produce the same results."""
    # First simulation
    np.random.seed(12345)
    sim1 = JobOfferSimulator(SimulationParameters(num_simulations=100))
    results1 = sim1.run_simulation()
    
    # Second simulation with same seed
    np.random.seed(12345)
    sim2 = JobOfferSimulator(SimulationParameters(num_simulations=100))
    results2 = sim2.run_simulation()
    
    # Results should be identical
    assert results1["probability_at_least_one_offer"] == results2["probability_at_least_one_offer"]
    assert results1["expected_num_offers"] == results2["expected_num_offers"]
    
    # Different seed should produce different results
    np.random.seed(54321)
    sim3 = JobOfferSimulator(SimulationParameters(num_simulations=100))
    results3 = sim3.run_simulation()
    
    # Results should be different (this might rarely fail due to chance)
    assert results1["probability_at_least_one_offer"] != results3["probability_at_least_one_offer"] or \
           results1["expected_num_offers"] != results3["expected_num_offers"]

def test_historical_data_extraction():
    """Test that historical data is correctly processed."""
    # Create a simple synthetic dataset
    data = pd.DataFrame({
        'current_stage': ['Rejected', 'Phone Screen', 'Technical Interview', 
                         'Onsite', 'Offer', 'Accepted', 'Rejected', 'Rejected']
    })
    
    simulator = JobOfferSimulator(historical_data=data)
    
    # Check the extracted probabilities
    assert simulator.baseline_probs["application_to_screen"] == 5/8  # 5 out of 8 passed screening
    assert simulator.baseline_probs["screen_to_interview"] == 3/5  # 3 out of 5 got to interview
    assert simulator.baseline_probs["interview_to_technical"] == 3/3  # All interviews went to technical
    assert simulator.baseline_probs["technical_to_onsite"] == 3/3  # All technical went to onsite
    assert simulator.baseline_probs["onsite_to_offer"] == 2/3  # 2 of 3 onsites got offers
    assert simulator.baseline_probs["offer_to_accept"] == 1/2  # 1 of 2 offers accepted
