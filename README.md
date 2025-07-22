# Premier League Score Predictor

This project uses a Poisson regression model to predict football match results based on historical Premier League data. It estimates expected goals for each team in a matchup and visualizes scoreline probabilities.

## What It Does

- Trains a Poisson model using real match data.
- Predicts expected goals for both home and away teams.
- Simulates full scoreline probabilities (e.g. 2–1, 0–0, etc.).
- Visualizes the result as a heatmap.

## How It Works

1. **Data Loading**: Reads a CSV file of past match results.
2. **Data Formatting**: Converts results into a long format suitable for modeling.
3. **Model Training**: Uses statsmodels to fit a Poisson regression.
4. **Simulation**: Predicts average goals for any given fixture.
5. **Visualization**: Uses matplotlib to display a heatmap of scoreline probabilities.

## 📁 File Structure

- `predictions.py` — Main Python script (data prep, modeling, prediction, visualization)
- `README.md` — This file
- `premier_league_results.csv` — Historical match data (must include `HomeTeam`, `AwayTeam`, `HomeGoals`, `AwayGoals`)

