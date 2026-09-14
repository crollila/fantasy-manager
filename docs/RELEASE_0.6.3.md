# Fantasy Manager 0.6.3

Makes the prediction loop fully automatic and adds a validated player projection model.

## The loop now runs itself

Once every regular-season game in a week has a final score, the NFL forecasting engine refits
on all completed games and promotes itself inside the ordinary refresh — no command, release
or restart. Every player with an upcoming game is projected and archived before kickoff, not
just the players on a connected roster. Completed games are graded automatically and become
training data for the next week. State lives on disk, so a restart resumes without repeating
or losing work.

## Player projections use role, red-zone and passing volume

Weekly projections now go through a model selected by walk-forward validation over 17,110
held-out player-weeks. It uses snap share and recent role change, red-zone and goal-line
opportunity, and expected team passing volume.

| | Before | After |
|---|---|---|
| Mean absolute error | 4.5917 | **4.4181** |
| Root mean squared error | 6.3772 | **6.1557** |
| Bias | +0.2518 | **+0.0393** |

A 0.1736-point improvement (3.8%), bootstrap 95% interval [+0.1409, +0.2087]. By position:
QB +0.31, RB +0.11, WR +0.17, TE +0.18. Every position improves; none degrades.

The model was chosen as the *smallest* feature set clearing a promotion rule fixed in advance,
out of 43 combinations tested. Market, generic usage and opponent-defence families were
evaluated and left out: they added 149 inputs for a further 0.042 points.

Grading uses the last projection saved strictly before each player's kickoff, falling back to
the most recent valid pregame version if the app was not running at kickoff.

## What did not change

Prediction ranges (p10/p90), boom and bust probabilities, and participation handling are
unchanged. A replacement interval system was tested and rejected for inadequate calibration —
71% coverage against an 80% target.

Also in this release: `/api/health` reports the real application version instead of a
hardcoded 0.5.0.

The research behind the projection model, including the rejected experiments, is published at
https://github.com/crollila/nfl-adaptive-forecasting

Install `Fantasy-Manager-Setup-0.6.3.exe` over the previous version. Saved teams, the ESPN
profile and existing prediction history are retained.
