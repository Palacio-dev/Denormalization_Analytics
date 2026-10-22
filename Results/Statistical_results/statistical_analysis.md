# Statistical Analysis of the Schema Denormalization Experiment

This document explains what the `analyze_llm_schema_experiment.py` script does.

## 1. Experimental Design

The experiment is a **complete 6 × 3 × 16 factorial design**, with repeated measures within schema:

- **6 prompt templates**: RISEN, RISEN-few-shot, JUNIOR, MID-LEVEL, SENIOR, OBT
- **3 LLM models**: Gemini 3.1-Pro, qwen3-coder:480b-cloud, qwen3.5:4b
- **16 reference relational schemas**

Each of the 16 schemas is denormalized under all 18 prompt × model combinations, generating **288 observations**. Since the same schema appears 18 times, these observations **are not independent** — this is what motivates the use of a mixed-effects model instead of, for example, a standard ANOVA.

The script checks this automatically at the start (`validate_design`): it counts how many schemas/prompts/models were detected in the CSV, warns if this does not match the expected (16/6/3), and reports whether any of the 288 combinations are missing.

## 2. The Six Dependent Variables

| Outcome | What it measures | How it is calculated | Range | Direction |
|---|---|---|---|---|
| `completeness` | Attribute coverage | `generated_attributes / reference_attributes` | ≥ 0, no fixed ceiling (can exceed 1 if the LLM generates extra attributes) | higher is better |
| `type_correctness` | Type correctness | proportion of **paired** attributes with the correct type | [0, 1] | higher is better |
| `field_correctness` | Attribute name correctness | **average** Levenshtein distance between paired names | ≥ 0, no fixed ceiling | **lower is better** (the only inverted metric) |
| `bleu` | Textual similarity | standard BLEU | [0, 1] | higher is better |
| `rouge` | Textual similarity | standard ROUGE | [0, 1] | higher is better |
| `meteor` | Textual similarity | standard METEOR | [0, 1] | higher is better |

None of these six is a raw count bounded by `total_attributes` — they all arrive from the CSV already as ratios, proportions, or average distances. For this reason, all of them are modeled as continuous variables (see Section 3), rather than as binomial proportions.

## 3. Primary Statistical Model

For each of the 6 outcomes, the script fits the same model:

```
outcome ~ Prompt * Model + (1 | Schema)
```

- **Fixed effects**: Prompt (6 levels), Model (3 levels), and the Prompt × Model interaction — testing whether the prompt type and the model (and their combination) affect the outcome.
- **Random effect**: random intercept by Schema — absorbs the fact that some schemas are intrinsically "easier" or "harder" to denormalize, and correctly models the dependence among the 18 repeated measures within each schema.
- It is a **Linear Mixed Model (LMM)**, fitted via maximum likelihood (`reml=False`), using `statsmodels`. ML (instead of REML) was chosen to keep the fits comparable across outcomes; if at some point you want more accurate variance estimates to report in the paper, REML is the preferred alternative (simply change `reml=False` to `reml=True`).

**Numerical adjustment:** the `lbfgs` optimizer converges well, but for some fits it produces an approximate Hessian that is not invertible, causing `statsmodels` to fail when computing standard errors (`Singular matrix`), even though the model has already converged. For this reason, the script tries a chain of optimizers (`lbfgs` → `bfgs` → `cg` → `powell` → `nm`) and uses the first one that works — this is purely a numerical detail and does not change the model specification.

## 4. Residual Diagnostics

For each LMM, the script saves two plots (`residuals_<outcome>.png` and `qq_<outcome>.png`):

- **Residuals vs. fitted values**: to check for non-linearity or heteroscedasticity.
- **Normal QQ-plot**: to check whether the residuals follow a normal distribution, an assumption of the LMM.

It is worth examining these plots before reporting the results — if any outcome shows a strong deviation from normality (for instance, `field_correctness`, which is a distance and tends to be right-skewed), it may be worth transforming the variable (e.g., `log1p`) before reporting it in the paper.

## 5. Multiple Comparisons Correction (FDR)

Each LMM tests several coefficients (Prompt, Model, and their interactions) — this constitutes test multiplicity within a single outcome, and there are also 6 outcomes being tested. To control the false positive rate, the script groups the outcomes into two **families** and applies the **Benjamini-Hochberg (FDR)** correction separately within each one:

- **Structural family**: `completeness`, `field_correctness`, `type_correctness` → `structural_fdr_results.csv`
- **NLP family**: `bleu`, `rouge`, `meteor` → `lmm_fdr_results.csv`

Each file has one row per coefficient tested (per outcome), with `p_value` (raw) and `p_fdr` (corrected). The random-effect variance terms (`Group Var`) are excluded from this correction — they are not hypothesis tests about Prompt/Model.

## 6. Exploratory Pairwise Comparisons (Post-hoc)

In addition to the LMM (which is the primary analysis), the script runs simple pairwise comparisons via the **Wilcoxon test** (non-parametric), paired by Schema: for each level of the other factor (e.g., fixing Model), it compares each pair of levels of the factor of interest (e.g., RISEN vs. JUNIOR), using only the schemas present in both conditions. The p-values are FDR-corrected within each outcome × factor combination, and saved in `posthoc_<outcome>_<factor>.csv`.

**Important**: these comparisons are labeled as exploratory in the script itself — the study's main inference should come from the LMM (Section 3), which already correctly models the dependence structure. The paired Wilcoxon test is useful for a quick, intuitive reading of "which prompt beats which," but it does not control for the interaction with the other factor.

## 7. Descriptive Statistics and Correlations

- `descriptive_statistics.csv`: mean, standard deviation, median, minimum, and maximum of each outcome, by prompt × model combination.
- `correlations_structural.csv` / `correlations_nlp.csv`: Spearman correlation between the outcomes within each family (e.g., is `completeness` associated with `bleu`?).
- `interaction_<outcome>.png`: interaction plot (means by prompt, one line per model) — a good complementary visualization to the LMM.

## 8. Data Quality Checks

Before any modeling, the script now performs three automatic checks:

1. **Decimal separator**: automatically converts comma → period if it detects cells with a decimal comma (preventing the entire column from being read as text).
2. **Extra whitespace**: removes extra spaces in categorical levels (schema/prompt/model), preventing, for example, `"RISEN"` and `"RISEN "` from being counted as different levels.
3. **Value range**: warns if `bleu`, `rouge`, `meteor`, or `type_correctness` fall outside the `[0, 1]` range, which indicates a corrupted/shifted cell in the data source (this is not something the script should correct on its own — it is a signal to check the data source).

## 9. Generated Files (`results/`)

| File | Content |
|---|---|
| `descriptive_statistics.csv` | Means/SD/median by prompt×model, all outcomes |
| `correlations_structural.csv`, `correlations_nlp.csv` | Spearman correlations between outcomes |
| `lmm_<outcome>.csv` | Full summary of the fitted LMM |
| `lmm_fdr_results.csv` | FDR-corrected p-values, NLP family |
| `structural_fdr_results.csv` | FDR-corrected p-values, structural family |
| `posthoc_<outcome>_<factor>.csv` | Exploratory pairwise comparisons (Wilcoxon) |
| `interaction_<outcome>.png` | Prompt × model interaction plot |
| `residuals_<outcome>.png`, `qq_<outcome>.png` | LMM residual diagnostics |
| `analysis_summary.txt` | Textual summary of the run |