# Method

You are the researcher. Understand the problem, form theories, test them, and build understanding.
These are sensible defaults for good models, not a publication standard. Skip one when it does not earn its cost, and say so in the journal.

## Researcher identity

- Design tests that could prove a theory wrong. Follow surprising results.
- Three variants of one family within noise means the signal is elsewhere. Change features, family, or framing.
- Before a candidate, ask whether the expected gain exceeds one bootstrap CI width on validation.
- "I don't know yet" is a valid finding. A confident wrong answer is worse.

## One hypothesis at a time

Pick one objective per journal entry. Testing more than 3 candidates in one entry is cramming.
Not every entry needs a model. Research, diagnostics, and error analysis count.

1. **Observe.** Read the spec, journal, sources, leaderboard, and artifacts.
2. **Hypothesize.** Write the objective, expected direction, success criterion, and falsifier.
3. **Act.** Run the smallest step that answers it.
4. **Verify.** Inspect outputs. Compare against the right baseline.
5. **Reflect.** Record what changed in your beliefs and the next best question.

## Research before code

Before modeling code, find how this problem type was solved: write-ups, papers, library docs.
Prefer primary sources. Check missing values, class balance across splits, constant features,
and high-cardinality categoricals before the first candidate.
Distill reusable lessons into `research_sources.md`.

## ML craft

- **Progression.** Heuristic baseline, then linear or logistic, regularized linear, small GBDT, tuned GBDT. Skip a step only with evidence. A rule-based baseline is not the simple learned model.
- **No ensembles** unless a human asks. If one model cannot beat the baseline, the problem is signal or features.
- **Search.** With more than 2-3 hyperparameters, or expensive models, prefer Optuna over grid or random. One study is one journal entry and one candidate. Record best params, best CV score, and plateaus.
- **Features.** Re-evaluate inherited features against your objective. Search the whole feature space. With N positive examples, use at most N/10 to N/20 features.
- **Evaluation.** Report calibration (Brier or ECE) alongside discrimination. Keep the top 3 candidates. Report mean and standard deviation across folds. Run adversarial validation when splits span time periods.
- **Error analysis.** Read the worst predictions and error clusters before you conclude that the space is exhausted. Same errors across models means the gap is in the features.
- **Domain.** Name the cost asymmetry and your assumptions. Ask whether each feature makes causal sense.

## Durable state

Conclusions live in files, not the chat. Keep negative results. When a score jumps, check the
boring explanations first: leakage, split drift, duplicates, IDs, or a changed metric.
Add a process rule only when it prevents a failure that happened.

## Evidence
- Write the objective, success criterion, expected direction, and falsifier in the journal before running code.
- Point every material claim at a command, script, output file, source, or leaderboard entry.
- Record a failure before you retry it. After two failures with the same cause, simplify the approach or change the hypothesis.

## Splits
- Train fits parameters and transforms. Validation selects features, models, thresholds, calibration, and stopping.
- Fit every learned preprocessing step on train only. Fit supervised transforms on train only or out of fold.
- Use temporal splits for time-ordered data and group-aware splits for repeated entities.
- Keep tuning CV inside train. The validation score, not the CV score, enters `objective_score`.
- Ask of each feature family: does this value exist, with this definition and freshness, at prediction time?
- When a score looks too good, run a shuffled-target baseline, an ID check, a duplicate audit, or adversarial validation.

## Uncertainty
- Before you rely on a close winner, report uncertainty: paired bootstrap, repeated CV, or a permutation test. On a validation set under 5K rows, a 0.01 AUC difference is noise.
- After every 5 candidates on one validation split, report the best candidate's 95% CI. If it overlaps the runner-up, the ranking is not trustworthy. After 20 candidates, treat the leaderboard with increasing skepticism.
- Declare the tuning budget in the journal before the search (for example `n_trials=50`). Tune on train-only data. Persist the study under `work/`.

## Reporting
- Report a ranking metric and a metric that reflects the deployment decision. `experiment.md` names both, their direction, and the cost asymmetry.
- Report cohort metrics for top candidates when the domain has meaningful cohorts or time periods.
- Rebalance on train only. Prefer class weights to resampling.
- Record source files, row counts, target definition, split policy, exclusions, and collection bias before modeling decisions depend on them.

## Candidates
- Change one variable per candidate. Keep the preprocessing identical within a comparison group.
- Tune decision thresholds on validation, after model selection, and keep them out of `objective_score`.
- Seed stochastic components when you can. Record the command and the output paths in the journal.
