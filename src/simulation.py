"""Simulation engine for job offer forecasting."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SimulationParameters:
    """Parameters for job offer simulation."""
    num_simulations: int = 1000
    time_horizon_days: int = 90
    applications_per_week: float = 5.0  # Average number of applications sent per week
    application_sources: Dict[str, float] = None  # Source probabilities
    
    # Factors affecting callbacks/interviews
    resume_quality: float = 0.7  # 0-1 scale
    interview_prep_level: float = 0.7  # 0-1 scale
    referral_ratio: float = 0.3  # Ratio of applications with referrals
    
    # Progression probability adjustments
    industry_experience_modifier: float = 0.0  # -0.5 to 0.5 adjustment
    skill_match_modifier: float = 0.0  # -0.5 to 0.5 adjustment
    market_condition_modifier: float = 0.0  # -0.5 to 0.5 adjustment
    
    def __post_init__(self):
        """Set default application sources if not provided."""
        if self.application_sources is None:
            self.application_sources = {
                "LinkedIn": 0.4,
                "Company Website": 0.2,
                "Referral": 0.3,
                "Job Board": 0.1
            }
            
        # Validate probabilities sum to 1
        total = sum(self.application_sources.values())
        if abs(total - 1.0) > 0.001:
            logger.warning(f"Application source probabilities sum to {total}, normalizing to 1.0")
            for k in self.application_sources:
                self.application_sources[k] /= total


class JobOfferSimulator:
    """Monte Carlo simulation engine for job offer forecasting."""
    
    def __init__(self, params: SimulationParameters = None, historical_data: pd.DataFrame = None):
        """Initialize simulator with parameters and optional historical data."""
        self.params = params or SimulationParameters()
        self.historical_data = historical_data
        
        # If historical data is provided, extract baseline probabilities
        if historical_data is not None:
            self._extract_baseline_stats()
        else:
            # Default baseline progression probabilities
            self.baseline_probs = {
                "application_to_screen": 0.25,  # 25% of applications get a screening call
                "screen_to_interview": 0.60,    # 60% of screens lead to an interview
                "interview_to_technical": 0.70, # 70% of first interviews lead to technical rounds
                "technical_to_onsite": 0.60,    # 60% of technical interviews lead to onsites
                "onsite_to_offer": 0.40,        # 40% of onsites lead to offers
                "offer_to_accept": 0.90         # 90% of offers are accepted (by the candidate)
            }
        
    def _extract_baseline_stats(self):
        """Extract baseline progression probabilities from historical data."""
        # This is a simplified version - in a real implementation, 
        # this would involve more sophisticated data analysis
        df = self.historical_data
        
        # Count applications at each stage
        total_apps = len(df)
        screens = df[df['current_stage'] != 'Rejected'].count()['current_stage'] 
        interviews = df[df['current_stage'].isin(['Technical Interview', 'Onsite', 'Offer', 'Accepted'])].count()['current_stage']
        technical = df[df['current_stage'].isin(['Onsite', 'Offer', 'Accepted'])].count()['current_stage']
        onsites = df[df['current_stage'].isin(['Offer', 'Accepted'])].count()['current_stage']
        offers = df[df['current_stage'].isin(['Offer', 'Accepted'])].count()['current_stage']
        accepts = df[df['current_stage'] == 'Accepted'].count()['current_stage']
        
        # Calculate transition probabilities
        self.baseline_probs = {
            "application_to_screen": screens / max(1, total_apps),
            "screen_to_interview": interviews / max(1, screens),
            "interview_to_technical": technical / max(1, interviews),
            "technical_to_onsite": onsites / max(1, technical),
            "onsite_to_offer": offers / max(1, onsites),
            "offer_to_accept": accepts / max(1, offers) 
        }
        
        logger.info(f"Extracted baseline probabilities: {self.baseline_probs}")
    
    def _adjust_probability(self, base_prob: float, factors: Dict[str, float]) -> float:
        """
        Adjust a base probability based on influencing factors.
        
        Args:
            base_prob: The baseline probability
            factors: Dictionary of {factor_name: modifier} where modifier is typically -0.5 to 0.5
            
        Returns:
            Adjusted probability (clamped between 0 and 1)
        """
        adj_prob = base_prob
        
        for factor, modifier in factors.items():
            if factor in ('resume_quality', 'interview_prep'):
                # These affect the base probability multiplicatively
                adj_prob *= (1 + modifier)
            else:
                # Others are additive adjustments
                adj_prob += modifier
                
        # Ensure result is between 0 and 1
        return max(0.0, min(1.0, adj_prob))
    
    def _simulate_application_process(self) -> Dict[str, int]:
        """
        Simulate a single job application through the entire process.
        
        Returns:
            Dictionary with results of this application simulation
        """
        # Apply modifiers based on simulation parameters
        modifiers = {
            'resume_quality': (self.params.resume_quality - 0.5) * 0.5,  # Convert 0-1 scale to -0.25 to 0.25
            'industry_exp': self.params.industry_experience_modifier,
            'skill_match': self.params.skill_match_modifier,
            'market': self.params.market_condition_modifier
        }
        
        # Determine if this application has a referral
        has_referral = np.random.random() < self.params.referral_ratio
        
        # Referrals boost initial screening probability significantly
        if has_referral:
            modifiers['referral_boost'] = 0.2
            
        # Simulate progression through stages
        
        # Application to screen
        p_screen = self._adjust_probability(
            self.baseline_probs["application_to_screen"], 
            {k: v for k, v in modifiers.items()}
        )
        passed_screen = np.random.random() < p_screen
        
        if not passed_screen:
            return {"final_stage": "application_rejected", "days_to_result": np.random.randint(1, 21)}
            
        # Screen to interview    
        p_interview = self._adjust_probability(
            self.baseline_probs["screen_to_interview"],
            {'interview_prep': (self.params.interview_prep_level - 0.5) * 0.5}
        )
        passed_interview = np.random.random() < p_interview
        
        if not passed_interview:
            return {"final_stage": "screen_rejected", "days_to_result": np.random.randint(5, 30)}
            
        # First interview to technical
        p_technical = self._adjust_probability(
            self.baseline_probs["interview_to_technical"],
            {'interview_prep': (self.params.interview_prep_level - 0.5) * 0.5, 
             'skill_match': modifiers['skill_match']}
        )
        passed_technical = np.random.random() < p_technical
        
        if not passed_technical:
            return {"final_stage": "interview_rejected", "days_to_result": np.random.randint(10, 45)}
            
        # Technical to onsite
        p_onsite = self._adjust_probability(
            self.baseline_probs["technical_to_onsite"],
            {'interview_prep': (self.params.interview_prep_level - 0.5) * 0.5,
             'skill_match': modifiers['skill_match']}
        )
        passed_onsite = np.random.random() < p_onsite
        
        if not passed_onsite:
            return {"final_stage": "technical_rejected", "days_to_result": np.random.randint(15, 60)}
            
        # Onsite to offer
        p_offer = self._adjust_probability(
            self.baseline_probs["onsite_to_offer"],
            {'interview_prep': (self.params.interview_prep_level - 0.5) * 0.5,
             'skill_match': modifiers['skill_match'],
             'industry_exp': modifiers['industry_exp']}
        )
        received_offer = np.random.random() < p_offer
        
        if not received_offer:
            return {"final_stage": "onsite_rejected", "days_to_result": np.random.randint(20, 75)}
            
        # Offer acceptance (mostly up to candidate)
        p_accept = self._adjust_probability(
            self.baseline_probs["offer_to_accept"],
            {'market': modifiers['market']}  # Market affects how likely you are to accept
        )
        accepted_offer = np.random.random() < p_accept
        
        if accepted_offer:
            return {"final_stage": "offer_accepted", "days_to_result": np.random.randint(30, 90)}
        else:
            return {"final_stage": "offer_declined", "days_to_result": np.random.randint(30, 90)}
    
    def run_simulation(self) -> Dict:
        """
        Run a Monte Carlo simulation of the job search process.
        
        Returns:
            Dictionary containing simulation results
        """
        logger.info(f"Starting simulation with {self.params.num_simulations} iterations")
        
        # How many applications will be sent in the time horizon
        apps_in_horizon = int(self.params.applications_per_week * (self.params.time_horizon_days / 7))
        
        # Arrays to store results
        offers_received = np.zeros(self.params.num_simulations, dtype=int)
        offers_accepted = np.zeros(self.params.num_simulations, dtype=int)
        days_to_first_offer = np.full(self.params.num_simulations, self.params.time_horizon_days + 1)
        
        # Run the simulations
        for sim in range(self.params.num_simulations):
            # For each simulation, process each application
            offers_in_sim = 0
            accepts_in_sim = 0
            
            for app in range(apps_in_horizon):
                # Calculate the day this application is sent
                app_day = int((app / apps_in_horizon) * self.params.time_horizon_days)
                
                # Simulate this application
                result = self._simulate_application_process()
                
                # Update counters based on result
                if result["final_stage"] in ("offer_accepted", "offer_declined"):
                    offers_in_sim += 1
                    # Check if this is the earliest offer
                    offer_day = app_day + result["days_to_result"]
                    if offer_day <= self.params.time_horizon_days:
                        days_to_first_offer[sim] = min(days_to_first_offer[sim], offer_day)
                
                if result["final_stage"] == "offer_accepted":
                    accepts_in_sim += 1
                    # Once we've accepted an offer, we stop applying
                    break
            
            offers_received[sim] = offers_in_sim
            offers_accepted[sim] = accepts_in_sim
        
        # Process results
        results = {
            "probability_at_least_one_offer": (offers_received > 0).mean(),
            "probability_accepted_offer": (offers_accepted > 0).mean(),
            "expected_num_offers": offers_received.mean(),
            "days_to_first_offer_stats": {
                "mean": days_to_first_offer[days_to_first_offer <= self.params.time_horizon_days].mean() 
                        if any(days_to_first_offer <= self.params.time_horizon_days) else None,
                "median": np.median(days_to_first_offer[days_to_first_offer <= self.params.time_horizon_days])
                          if any(days_to_first_offer <= self.params.time_horizon_days) else None,
                "p25": np.percentile(days_to_first_offer[days_to_first_offer <= self.params.time_horizon_days], 25)
                       if any(days_to_first_offer <= self.params.time_horizon_days) else None,
                "p75": np.percentile(days_to_first_offer[days_to_first_offer <= self.params.time_horizon_days], 75)
                       if any(days_to_first_offer <= self.params.time_horizon_days) else None,
            },
            "offer_count_distribution": {
                str(i): (offers_received == i).mean() for i in range(max(offers_received) + 1)
            },
            "offers_by_day": {
                # Probability of at least one offer by each week
                f"day_{d}": (days_to_first_offer <= d).mean() 
                for d in range(0, self.params.time_horizon_days + 1, 7)
            }
        }
        
        logger.info(f"Simulation complete. Probability of at least one offer: {results['probability_at_least_one_offer']:.2f}")
        return results

def load_historical_data(filepath: str) -> pd.DataFrame:
    """Load historical job application data from CSV."""
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        return None

def main():
    """Command-line interface for the simulator."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Job Offer Probability Simulator")
    parser.add_argument("--data", type=str, help="Path to historical data CSV file")
    parser.add_argument("--output", type=str, default="simulation_results.json", 
                        help="Output file for simulation results")
    parser.add_argument("--num-sims", type=int, default=1000, 
                        help="Number of Monte Carlo simulations to run")
    parser.add_argument("--days", type=int, default=90, 
                        help="Time horizon in days")
    parser.add_argument("--apps-per-week", type=float, default=5.0, 
                        help="Average applications per week")
    parser.add_argument("--resume-quality", type=float, default=0.7, 
                        help="Resume quality score (0-1)")
    parser.add_argument("--interview-prep", type=float, default=0.7, 
                        help="Interview preparation level (0-1)")
    parser.add_argument("--referral-ratio", type=float, default=0.3, 
                        help="Ratio of applications with referrals (0-1)")
    
    args = parser.parse_args()
    
    # Load historical data if provided
    historical_data = None
    if args.data:
        historical_data = load_historical_data(args.data)
        if historical_data is None:
            logger.warning("Could not load historical data, using default baseline probabilities")
    
    # Set up simulation parameters
    params = SimulationParameters(
        num_simulations=args.num_sims,
        time_horizon_days=args.days,
        applications_per_week=args.apps_per_week,
        resume_quality=args.resume_quality,
        interview_prep_level=args.interview_prep,
        referral_ratio=args.referral_ratio
    )
    
    # Run simulation
    simulator = JobOfferSimulator(params, historical_data)
    results = simulator.run_simulation()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {args.output}")
    
    # Print summary
    print("\n=== Simulation Results ===")
    print(f"Probability of receiving at least one offer: {results['probability_at_least_one_offer']:.2%}")
    print(f"Expected number of offers: {results['expected_num_offers']:.2f}")
    if results['days_to_first_offer_stats']['median'] is not None:
        print(f"Median days to first offer: {results['days_to_first_offer_stats']['median']:.1f}")
    
    print("\nOffer probability by day:")
    for day, prob in sorted(results['offers_by_day'].items()):
        day_num = int(day.split('_')[1])
        print(f"  Day {day_num}: {prob:.2%}")

if __name__ == "__main__":
    main()
