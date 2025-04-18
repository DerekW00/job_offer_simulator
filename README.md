# Job‑Offer Probability Simulator 🏹

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/yourusername/job-offer-simulator/actions/workflows/python-tests.yml/badge.svg)](https://github.com/yourusername/job-offer-simulator/actions/workflows/python-tests.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Last Commit](https://img.shields.io/github/last-commit/yourusername/job-offer-simulator)](https://github.com/yourusername/job-offer-simulator/commits/main)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A Monte Carlo simulation engine that estimates your probability of receiving a job offer within a given timeframe based on your application strategy, resume quality, interview preparation, and market conditions.

![Simulator Screenshot](docs/images/screenshot.png)

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/job-offer-simulator.git
cd job-offer-simulator

# Install dependencies
poetry install

# Generate sample data
poetry run python scripts/seed_fake_data.py

# Run the Streamlit app
poetry run streamlit run app/streamlit_app.py
```

Then visit http://localhost:8501 in your browser.

## 🎯 What Does It Do?

The Job Offer Simulator helps you:

1. **Estimate Probabilities**: Calculate your chances of receiving a job offer within a specific timeframe
2. **Run What-If Scenarios**: See how changes to your application strategy affect outcomes
3. **Visualize Outcomes**: Explore interactive charts of offer probability over time
4. **Optimize Strategy**: Find the most effective ways to improve your job search

## 💡 Why Is It Useful?

Job searching is inherently probabilistic, but most people approach it with a deterministic mindset. This simulator helps you:

- **Set Realistic Expectations**: Understand the likely timeline and outcomes of your search
- **Make Data-Driven Decisions**: Identify which factors most improve your chances
- **Reduce Anxiety**: Replace uncertainty with probability-based planning
- **Allocate Resources**: Focus your efforts where they have the highest impact

## 🛠️ How Does It Work?

The simulator uses Monte Carlo methods to model the job application process as a series of probability-weighted transitions:

1. **Application → Phone Screen**: Affected by resume quality, referrals
2. **Phone Screen → Interview**: Affected by interview preparation
3. **Interview → Technical Rounds**: Affected by technical skills, preparation
4. **Technical → Onsite**: Affected by technical performance, experience
5. **Onsite → Offer**: Affected by overall performance, cultural fit
6. **Offer → Acceptance**: Affected by market conditions, competing offers

Each simulation runs thousands of virtual job searches with your parameters to estimate probabilities.

## 📊 Example Output

The simulator produces:

- Probability of receiving at least one offer within your timeframe
- Expected number of offers
- Median time to first offer
- Probability timeline showing offer likelihood by day
- Offer count distribution

## 🔧 Configuration

You can adjust various parameters:

- Time horizon (days to simulate)
- Applications per week
- Resume quality (0-1 scale)
- Interview preparation level (0-1 scale)
- Percentage of applications with referrals
- Industry experience modifier (-0.5 to 0.5)
- Skill match modifier (-0.5 to 0.5)
- Market condition modifier (-0.5 to 0.5)

## 💻 Development

### Prerequisites

- Python 3.9+
- Poetry

### Installation

```bash
# Install dependencies
poetry install

# Install pre-commit hooks
poetry run pre-commit install
```

### Testing

```bash
# Run tests
poetry run pytest

# Run linting
poetry run ruff check .
poetry run black --check .
```

## 🗺️ Roadmap

- [ ] Add salary negotiation simulation
- [ ] Incorporate geographic factors
- [ ] Add industry-specific calibration
- [ ] Implement profile saving/loading
- [ ] Create Docker container
- [ ] Add CI/CD pipeline for testing

## 🤝 Contributing

Contributions are welcome! Please check out our [Contributing Guidelines](CONTRIBUTING.md) for details on how to get started.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Getting Help

If you have questions or need help using the simulator, please [open an issue](https://github.com/yourusername/job-offer-simulator/issues/new) on GitHub.
