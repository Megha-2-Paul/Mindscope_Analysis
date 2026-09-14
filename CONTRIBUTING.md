# Contributing to MindScope

MindScope is currently maintained as a portfolio/research project. Contributions that improve correctness, reproducibility, testing, documentation, or responsible ML practices are welcome.

## Development setup

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies with `pip install -r requirements.txt`.
4. Run the test suite with `pytest` before submitting changes.

## Code guidelines

- Keep functions focused and reusable.
- Prefer clear names over clever implementations.
- Avoid hard-coded paths and thresholds where configuration is more appropriate.
- Keep model training and evaluation reproducible with fixed random seeds where practical.
- Do not introduce target leakage.
- Do not commit secrets, credentials, private employee data, or generated sensitive records.

## Testing

New functionality should include tests where practical. Changes to the ML pipeline should verify both model behavior and leakage prevention.

## Pull requests

A useful pull request should explain:

- What changed
- Why it changed
- How it was tested
- Any limitations or follow-up work

For changes affecting model performance, include the relevant evaluation results and explain any change in methodology.

## Responsible use

MindScope should not be used as a clinical diagnostic system or as the sole basis for consequential decisions about employees. Contributions should preserve appropriate privacy, security, and responsible-ML safeguards.
