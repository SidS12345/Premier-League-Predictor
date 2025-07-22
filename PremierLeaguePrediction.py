
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson,skellam

epl_2122 = pd.read_csv("/Users/sid/Desktop/Python/Predicting_premier_league/premier_league_results.csv")
epl_2122 = epl_2122[:-423]
import statsmodels.api as sm
import statsmodels.formula.api as smf

goal_model_data = pd.concat([epl_2122[['HomeTeam','AwayTeam','HomeGoals']].assign(home=1).rename(
            columns={'HomeTeam':'team', 'AwayTeam':'opponent','HomeGoals':'goals'}),
           epl_2122[['AwayTeam','HomeTeam','AwayGoals']].assign(home=0).rename(
            columns={'AwayTeam':'team', 'HomeTeam':'opponent','AwayGoals':'goals'})])

poisson_model = smf.glm(formula="goals ~ home + team + opponent", data=goal_model_data, 
                        family=sm.families.Poisson()).fit()

# Simulate match gives us probabilities of each team scoring a certain number of goals
def simulate_match(foot_model, homeTeam, awayTeam, max_goals=5):
    home_goals_avg = foot_model.predict(pd.DataFrame(data={'team': [homeTeam], 
                                                            'opponent': [awayTeam], 
                                                            'home': [1]}))[0]
    away_goals_avg = foot_model.predict(pd.DataFrame(data={'team': [awayTeam], 
                                                            'opponent': [homeTeam], 
                                                            'home': [0]}))[0]

    home_probs = [poisson.pmf(i, home_goals_avg) for i in range(max_goals+1)]
    away_probs = [poisson.pmf(i, away_goals_avg) for i in range(max_goals+1)]

    match_probs = np.outer(home_probs, away_probs)

    return match_probs

home_team = input("enter home team: ") 
away_team = input("enter away team: ")

home_team = input("Enter home team: ")
away_team = input("Enter away team: ")

# 
match_matrix = simulate_match(poisson_model, home_team, away_team, max_goals=5)

# Create heatmap
plt.figure(figsize=(8, 6))
plt.imshow(match_matrix, cmap='viridis', origin='lower')

plt.colorbar(label='Probability')
plt.xticks(np.arange(6), labels=[str(i) for i in range(6)])
plt.yticks(np.arange(6), labels=[str(i) for i in range(6)])

plt.xlabel(f'{away_team} goals')
plt.ylabel(f'{home_team} goals')
plt.title(f'Scoreline probabilities: {home_team} vs {away_team}')

# Add probability annotations
for i in range(6):
    for j in range(6):
        prob = match_matrix[i, j]
        plt.text(j, i, f"{prob:.2%}", ha='center', va='center', color='white' if prob < 0.08 else 'black')

plt.tight_layout()
plt.show()

# Print expected score of the game
print(poisson_model.predict(pd.DataFrame(data={'team': home_team, 'opponent': away_team,
                                       'home':1},index=[1])))
print(poisson_model.predict(pd.DataFrame(data={'team': away_team, 'opponent': home_team,
                                       'home':0},index=[1])))