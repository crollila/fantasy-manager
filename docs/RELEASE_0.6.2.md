# Fantasy Manager 0.6.2

Adds team logos to the NFL game cards.

The cards identified each team by abbreviation alone, so reading a full week of games meant decoding thirty-two two- and three-letter codes. Every card now carries the team's logo beside its name, on both score rows, next to the pick, and in the matchup heading when a game is opened.

- All 32 logos are bundled inside the application, downscaled to 128px (364 KB in total). Nothing is fetched at runtime, so the cards look the same with no internet connection.
- The schedule feed and the scoreboard spell some franchises differently (`LA` and `LAR`, `WAS` and `WSH`). The logo lookup maps those onto one image each, including the historical `STL`, `SD` and `OAK` codes, and falls back to a team-coloured initials badge for any code it does not recognise rather than leaving a gap.

No forecast, feature or grading change. The champion's configuration, features and holdout results are unchanged from 0.6.0; its learners are refit on every game completed to date (`nfl-forecast-fv1-20260914T005641Z`, 7,290 games through 2026 week 1), which is the routine refresh described in the model card and does not alter how a forecast is produced.

Install `Fantasy-Manager-Setup-0.6.2.exe` over the previous version. Saved teams, the ESPN profile and existing prediction history are retained.
