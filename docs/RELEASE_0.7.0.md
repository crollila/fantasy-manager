# Fantasy Manager 0.7.0

Look back at any week's forecasts, see power rankings, compare the model with Kalshi and
Polymarket, check that the model is learning, and switch to dark mode.

## Week picker on NFL games

A **Week** menu above the game cards. The current week shows the live forecasts as before. Any
earlier week shows, for each game, the last forecast saved before kickoff next to the final
score, a ✓ or ✗ for the pick, and the Kalshi and Polymarket odds at kickoff. Each week has a
summary: pick record, probability error (Brier score and log loss) and average score miss.
Switch between cards and a table.

## Power rankings

A new **Power rankings** tab. Each team's rating is the number of points the forecasting model
expects it to beat an average NFL team by on a neutral field. Every team's current data
(ratings, recent form, quarterback, injuries) is paired against every other team's, home and
away, so home-field advantage and schedule cancel out. Also shown: offense and defense split,
record, rank change since last week, a rank trend by week and the next opponent. Click a team
for points for and against, Elo and recent results.

## Model vs prediction markets

A new **Market backtest** tab.

- **Weekly scoreboard.** After each week finishes, the app records Kalshi and Polymarket prices
  at kickoff and shows whether the model's probabilities beat each market's (Brier score) that
  week.
- **Betting backtest.** If you had bought a team's "wins" contract whenever the model's win
  probability beat its kickoff price by a chosen edge, what would have happened? Choose the
  market, the minimum edge, flat $100 stakes or quarter Kelly, and which games. Shows profit,
  return, record, worst drawdown, a profit chart and every bet. Kalshi and Polymarket taker
  fees are deducted.
- Covers 2024–2025 using the engine's walk-forward predictions (each season predicted by a
  model trained only on earlier seasons) plus every live 2026 forecast. Kalshi game markets
  start in 2025.

Results so far, flat $100 at a 5-point edge: 2024–25 walk-forward, Kalshi −2.4% and Polymarket
−5.0% return, with the markets' probabilities more accurate than the model's. 2026 weeks 1–4
live, Kalshi +49% and Polymarket +39% on 24 and 33 bets, beating both markets' probabilities.
The live sample is small. Hypothetical only; no orders are placed.

## Is the model learning?

The **Learning** tab now re-scores every finished week three ways, each using only models
trained before that week: what the app did, the same without the weekly refit, and a version
that never learns from new results (ratings, form and quarterback stats frozen at week 1).

On 2022–2025 (1,084 games, out of sample), learning from results gives 65.7% correct picks
versus 60.0% frozen. The gap grows from nothing in weeks 1–4 to 66.9% versus 57.9% in weeks
10–18. The weekly refit itself changes probabilities by under one point on average; most of the
learning comes from updated team data.

## Dark mode

A theme button in the top bar cycles between System, Dark and Light. The choice is remembered.

Install `Fantasy-Manager-Setup-0.7.0.exe` over the previous version. Saved teams, the ESPN
profile and prediction history are kept. Kalshi and Polymarket prices for finished 2026 games
are collected on the first refresh after the update.
