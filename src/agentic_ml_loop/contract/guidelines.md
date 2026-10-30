# Guidelines

Rules that code cannot check. The `## Enforced rules` table in this repository's `AGENTS.md` lists the rules that code enforces.
Method lives in [program.md](program.md).

## Evidence

1. Write the objective, success criterion, expected direction, and falsifier in the journal before running code.
2. Point every material claim at a command, script, output file, source, or leaderboard entry.
3. Record a failure before you retry it. After two failures with the same cause, simplify the approach or change the hypothesis.

## Splits

4. Train fits parameters and transforms. Validation selects features, models, thresholds, calibration, and stopping.
5. Fit every learned preprocessing step on train only. Fit supervised transforms on train only or out of fold.
6. Use temporal splits for time-ordered data and group-aware splits for repeated entities.
7. Keep tuning CV inside train. The validation score, not the CV score, enters `objective_score`.
8. Ask of each feature family: does this value exist, with this definition and freshness, at prediction time?
9. When a score looks too good, run a shuffled-target baseline, an ID check, a duplicate audit, or adversarial validation.

## Uncertainty

10. Before you claim a winner, report uncertainty: paired bootstrap with at least 1000 resamples, repeated CV, or a permutation test. On a validation set under 5K rows, a 0.01 AUC difference is noise.
11. After every 5 candidates on one validation split, report the best candidate's 95% CI. If it overlaps the runner-up, the ranking is not trustworthy. After 20 candidates, treat the leaderboard with increasing skepticism.
12. `experiment validate` warns when a gap exceeds 0.05 AUC or 10% relative. Investigate before you accept that candidate.
13. Declare the tuning budget in the journal before the search (for example `n_trials=50`). Tune on train-only data. Persist the study under `work/`.

## Reporting

14. Report a ranking metric and a metric that reflects the deployment decision. `experiment.md` names both, their direction, and the cost asymmetry.
15. Report cohort metrics for top candidates when the domain has meaningful cohorts or time periods.
16. Rebalance on train only. Prefer class weights to resampling.
17. Record source files, row counts, target definition, split policy, exclusions, and collection bias before modeling decisions depend on them.

## Candidates

18. Change one variable per candidate. Keep the preprocessing identical within a comparison group.
19. Tune decision thresholds on validation, after model selection, and keep them out of `objective_score`.
20. Seed every stochastic component. Record the command and the output paths in the journal.
