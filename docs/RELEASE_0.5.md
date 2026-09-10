# Fantasy Manager 0.5.0

Game predictions now have a postgame review and a prospective learning process. Open **Accuracy → Review** to see score errors, what matched, statistical surprises, missing evidence and the injury/participation assumptions saved before kickoff. Open **Learning** to see when a correction is training, being tested, promoted or rolled back.

- Saves expectations for up to 14 metrics using team statistics, play-by-play and newly integrated Next Gen Stats passing, rushing and receiving feeds.
- Preserves original predictions, exact model artifacts and versioned result corrections. Each game has a full local audit export.
- Fits score corrections from at least 80 eligible archived games, tests frozen candidates on at least 32 later games across three weeks, and requires a paired accuracy test before promotion. Active corrections receive later rollback tests.
- Adds same-game market margin, total and probability comparisons, probability calibration and condition error breakdowns.
- Accepts licensed game-score forecast JSON exports as local benchmarks. Past games cannot be imported into the live record.
- Retains remembered ESPN sign-in and the professional desktop theme.

Install `Fantasy-Manager-Setup-0.5.0.exe` over the previous version. Saved teams, the ESPN profile and existing prediction history are retained. The Windows x64 installer includes the analysis engine; Python and Node are not required. It is currently unsigned, as prior releases were.

The foundation model updates completed history on refresh. The correction layer starts collecting new-format archived evidence and will not claim it has learned an improvement before its future test passes. Detailed source statistics can arrive after the final score and are revisited automatically. No advantage over Vegas is established or guaranteed.

See [the complete data, review and learning methodology](GAME_LEARNING.md) and [installation instructions](DESKTOP.md).
