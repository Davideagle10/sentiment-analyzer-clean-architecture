# 🧠 Sentiment Analyzer – Clean Architecture with OpenAI API

A modular and extensible sentiment analysis system built with **Python**, designed under **SOLID principles** and leveraging the **OpenAI API** for natural language understanding.

This project demonstrates professional-grade architecture, dependency injection, configuration validation with **Pydantic**, and a robust testing approach — all structured for scalability and maintainability.

---

## Key Features

- **Clean Architecture:** Clear separation between configuration, logic, and testing layers.
- **SOLID Principles:** Ensures extensibility and easy maintenance.
- **Dependency Injection:** Reduces coupling and promotes testability.
- **Config Validation:** Uses `Pydantic` for secure and structured environment management.
- **Error Handling:** Implements layered exception management for stability.
- **Unit Testing:** Includes tests to ensure consistency and reliability.
- **OpenAI Integration:** Abstracted GPT-powered sentiment analysis.


---

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Davideagle10/sentiment-analyzer-clean-architecture.git
   cd sentiment-analyzer
Create and activate a virtual environment

bash
Copiar código
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
Install dependencies

bash
Copiar código
pip install -r requirements.txt
Set environment variables

bash
Copiar código
export OPENAI_API_KEY="your_api_key"
Run the analyzer

bash
Copiar código
python sentiment_analyzer.py
Running Tests
bash
Copiar código
pytest
All core components are covered by unit tests to ensure stability and correctness.

# Design Highlights
Principle	Application
Single Responsibility	Each module handles one concern (config, logic, tests).
Open/Closed	Easily extendable to support new models or providers.
Dependency Inversion	Abstracts external APIs via interface-style design.
Validation Layer	Pydantic ensures consistent and secure configuration.
Error Abstraction	Clear exception hierarchy for debugging and observability.

# Tech Stack
Python 3.10+

OpenAI API (GPT-3.5 / GPT-4)

Pydantic

Pytest

SOLID / Clean Architecture Principles



# Author
David Aguilar
DevOps & Software Engineer | Cloud | Python | Clean Architecture


# License
This project is released under the MIT License.
You’re free to use, modify, and distribute it with proper attribution.




