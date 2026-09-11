# Fantasy Manager 0.6.1

Fixes the win probability displayed beside an NFL game pick.

The schedule feed and the scoreboard spell one team differently: the Los Angeles Rams are `LA` in nflverse data and `LAR` on ESPN's scoreboard. The game card decided which probability to show by matching the pick's name against the home team's name, so for a Rams home game the two strings never matched and the card showed the **opponent's** win probability next to the Rams pick. A card reading "LAR 25.2, SF 24.4, Pick: LA, 46.1% win probability" was really a 50.9% pick for the Rams; 46.1% was San Francisco's number.

- Forecasts now publish `pick_win_probability` explicitly, so the number beside a pick always belongs to the picked team. The card reads that field instead of matching names.
- Grading and postgame reviews compare canonical team codes, so a pick saved as `LA` is graded a win when `LAR` wins. No existing record was affected: nothing had been graded yet.
- Two regression tests cover the relabelling and the grading path.

No model, feature or accuracy change: the champion, its training data and its holdout results are identical to 0.6.0.

Install `Fantasy-Manager-Setup-0.6.1.exe` over the previous version. Saved teams, the ESPN profile and existing prediction history are retained.
