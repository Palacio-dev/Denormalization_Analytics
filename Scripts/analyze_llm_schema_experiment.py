#!/usr/bin/env python3

"""
Analysis of a 6 x 3 factorial experiment with repeated schemas.

Experimental design
-------------------
Fixed factors:
    Prompt: 6 levels
    Model: 3 levels

Experimental/repeated unit:
    Schema: 16 schemas

Expected observations:
    6 prompts x 3 models x 16 schemas = 288

Primary statistical structure:
    Outcome ~ Prompt * Model + (1 | Schema)

Models:
    Completeness (already a ready-made ratio: generated attribute
    count / reference attribute count -- no fixed upper bound,
    NOT a count out of total_attributes):
        Linear Mixed Model (LMM)

    Data Type Correctness (already a ready-made [0,1] proportion
    of paired attributes -- not a count out of total_attributes):
        Linear Mixed Model (LMM)

    Data Field Correctness (mean Levenshtein distance between
    matched attribute names/types -- non-negative continuous,
    lower is better, NOT a bounded proportion):
        Linear Mixed Model (LMM)

    BLEU / ROUGE / METEOR:
        Linear Mixed Model (LMM)

Multiple outcomes:
    Results are grouped into:
        1. Structural correctness:
           Completeness, Field Correctness, Type Correctness
           (FDR-corrected together in structural_fdr_results.csv,
           even though Field Correctness is fit with an LMM and
           the other two with GLMMs)

        2. NLP similarity:
           BLEU, ROUGE, METEOR (lmm_fdr_results.csv)

    Benjamini-Hochberg FDR correction is applied within each
    outcome family.

The script produces:
    results/
        descriptive_statistics.csv
        lmm_<metric>.csv
        lmm_<metric>_fixed_effects.csv
        glmm_<metric>.csv
        glmm_<metric>_fixed_effects.csv
        lmm_fdr_results.csv
        structural_fdr_results.csv
        posthoc_<metric>.csv
        correlations_<family>.csv
        interaction_<metric>.png
        residuals_<metric>.png
        qq_<metric>.png
        analysis_summary.txt
"""

# ============================================================
# Imports
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.formula.api as smf

from scipy import stats
from statsmodels.stats.multitest import multipletests


warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "schema_denormalization_results.csv"
OUTPUT_DIR = "results"

# Expected factor names
SCHEMA = "schema"
PROMPT = "prompt"
MODEL = "model"

# Count variables
TOTAL_ATTRIBUTES = "total_attributes"
COMPLETE_ATTRIBUTES = "complete_attributes"
CORRECT_FIELDS = "correct_fields"
CORRECT_TYPES = "correct_types"

# Continuous metrics
NLP_METRICS = [
    "bleu",
    "rouge",
    "meteor",
]

# Correctness/count outcomes (used for descriptive stats,
# interaction plots, and exploratory paired comparisons — these
# are model-agnostic and apply to all three structural outcomes)
COUNT_OUTCOMES = [
    "completeness",
    "field_correctness",
    "type_correctness",
]

# Outcomes fit with the binomial GLMM (bounded successes/trials
# proportions, where total_attributes is a valid trial count).
# None of the three structural outcomes qualify, given their
# confirmed definitions:
#   - completeness is already a ready-made ratio (generated
#     attribute count / reference attribute count) with no fixed
#     upper bound (can exceed 1 if the LLM over-generates) -- not
#     a raw count out of total_attributes.
#   - field_correctness is the mean Levenshtein distance between
#     matched attribute names -- a non-negative continuous
#     distance (no natural "trials" count), not a proportion.
#   - type_correctness is already a ready-made proportion (share
#     of PAIRED attributes with correct types) computed over the
#     number of paired attributes, not total_attributes -- and
#     that pairing count isn't in this CSV, so it can't be turned
#     back into a valid successes/trials pair.
# Kept as an empty list (rather than deleted) so a future outcome
# that IS a genuine bounded count can be added here directly, with
# its successes/trials columns declared in GLMM_SUCCESS_COLUMNS.
GLMM_OUTCOMES = []

GLMM_SUCCESS_COLUMNS = {
    # outcome_name: (successes_column, trials_column)
}

# Outcomes fit with the continuous LMM.
LMM_OUTCOMES = NLP_METRICS + [
    "completeness",
    "field_correctness",
    "type_correctness",
]

# Outcome families for FDR correction. The three structural
# outcomes stay grouped here conceptually, even though all three
# are now modeled with an LMM rather than a binomial GLMM.
STRUCTURAL_FAMILY = [
    "completeness",
    "field_correctness",
    "type_correctness",
]

NLP_FAMILY = [
    "bleu",
    "rouge",
    "meteor",
]

EXPECTED_SCHEMAS = 16
EXPECTED_PROMPTS = 6
EXPECTED_MODELS = 3


# ============================================================
# Utility functions
# ============================================================

def ensure_output_directory():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_csv(df, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def print_header(title):
    print("\n")
    print("=" * 75)
    print(title)
    print("=" * 75)


# ============================================================
# 1. Load and validate data
# ============================================================

def load_data():

    print_header("LOADING DATA")

    df = pd.read_csv(INPUT_FILE)

    print(f"Input file: {INPUT_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    required_columns = [
        SCHEMA,
        PROMPT,
        MODEL,
        TOTAL_ATTRIBUTES,
        COMPLETE_ATTRIBUTES,
        CORRECT_FIELDS,
        CORRECT_TYPES,
        *NLP_METRICS,
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    # Strip stray whitespace before treating these as factor
    # levels. Without this, e.g. "RISEN" and "RISEN " (trailing
    # space) are silently treated as two different prompt levels.
    for col in (SCHEMA, PROMPT, MODEL):
        df[col] = df[col].astype(str).str.strip()

    # Tolerate mixed decimal separators. If some rows were edited
    # with a comma decimal (e.g. "1,33" instead of "1.33"), pandas
    # reads the whole column as text instead of raising an error,
    # which silently breaks every computation downstream. Convert
    # comma -> period before parsing as numeric.
    numeric_columns = [
        TOTAL_ATTRIBUTES,
        COMPLETE_ATTRIBUTES,
        CORRECT_FIELDS,
        CORRECT_TYPES,
        *NLP_METRICS,
    ]

    for col in numeric_columns:

        if not pd.api.types.is_numeric_dtype(df[col]):

            n_comma = df[col].astype(str).str.contains(
                ",", na=False
            ).sum()

            if n_comma > 0:
                print(
                    f"NOTE: '{col}' had {n_comma} value(s) using "
                    "a comma decimal separator; converted to "
                    "period before parsing."
                )

            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
            )

        df[col] = pd.to_numeric(df[col], errors="raise")

    # Factors as categorical
    df[SCHEMA] = df[SCHEMA].astype("category")
    df[PROMPT] = df[PROMPT].astype("category")
    df[MODEL] = df[MODEL].astype("category")

    return df


# ============================================================
# 1b. Data sanity checks
# ============================================================

def sanity_check_ranges(df):

    print_header("DATA SANITY CHECKS")

    issues = []

    # bleu/rouge/meteor and correct_types are expected to be
    # proportions/scores bounded in [0, 1]. Values outside this
    # range indicate shifted or corrupted cells upstream (e.g. in
    # how the CSV was generated), not something this script can
    # infer or correct on its own.
    for col in NLP_METRICS + [CORRECT_TYPES]:

        bad = df[
            (df[col] < 0)
            | (df[col] > 1)
        ]

        if len(bad) > 0:
            issues.append((col, bad))

    if issues:

        print(
            "WARNING: found values outside the expected [0, 1] "
            "range. These rows likely have shifted/corrupted "
            "values and should be checked against the raw "
            "experiment logs before trusting the analysis:\n"
        )

        for col, bad in issues:

            print(f"  Column '{col}':")
            print(
                bad[[SCHEMA, PROMPT, MODEL, col]]
                .to_string(index=False)
            )
            print()

    else:
        print(
            "No out-of-range values detected in "
            "bleu / rouge / meteor / correct_types."
        )

    return issues


# ============================================================
# 2. Create dependent variables
# ============================================================

def create_outcomes(df):

    print_header("CREATING DEPENDENT VARIABLES")

    # Completeness:
    #
    # complete_attributes is already the ratio of generated
    # attribute count to reference (total_attributes) attribute
    # count. It has no fixed upper bound (can exceed 1 if the LLM
    # over-generates attributes), so it is used as-is rather than
    # divided again, and modeled with a continuous LMM rather than
    # the binomial GLMM (which would require a bounded
    # successes/trials pair).
    df["completeness"] = df[COMPLETE_ATTRIBUTES]

    # Field correctness:
    #
    # correct_fields is the mean Levenshtein distance (LD) between
    # matched attribute names/types — a non-negative continuous
    # distance, not a count out of total_attributes. Dividing by
    # total_attributes would mix incompatible units (edit distance
    # in characters vs. attribute count), so it is used as-is.
    # NOTE: this metric is inverted relative to the others — LOWER
    # values mean BETTER correctness (0 = exact match). Modeled
    # with a continuous LMM, not the binomial GLMM.
    df["field_correctness"] = df[CORRECT_FIELDS]

    # Type correctness:
    #
    # correct_types is already the proportion of PAIRED attributes
    # with correct types (confirmed range: 0-1). It is NOT a raw
    # count out of total_attributes, and the true denominator
    # (number of paired attributes) isn't in this CSV, so it's
    # used as-is rather than divided again. Modeled with a
    # continuous LMM, not the binomial GLMM.
    df["type_correctness"] = df[CORRECT_TYPES]

    return df


# ============================================================
# 3. Validate experimental design
# ============================================================

def validate_design(df):

    print_header("EXPERIMENTAL DESIGN VALIDATION")

    n_schemas = df[SCHEMA].nunique()
    n_prompts = df[PROMPT].nunique()
    n_models = df[MODEL].nunique()

    print(f"Schemas : {n_schemas}")
    print(f"Prompts : {n_prompts}")
    print(f"Models  : {n_models}")

    if (
        n_schemas != EXPECTED_SCHEMAS
        or n_prompts != EXPECTED_PROMPTS
        or n_models != EXPECTED_MODELS
    ):
        print(
            "\nWARNING: detected factor levels do not match the "
            f"documented design (expected {EXPECTED_SCHEMAS} schemas, "
            f"{EXPECTED_PROMPTS} prompts, {EXPECTED_MODELS} models). "
            "Update EXPECTED_SCHEMAS/EXPECTED_PROMPTS/EXPECTED_MODELS "
            "at the top of the script if the design has changed."
        )

    expected = (
        n_schemas
        * n_prompts
        * n_models
    )

    print(f"Expected observations: {expected}")
    print(f"Actual observations:   {len(df)}")

    if len(df) != expected:
        print(
            "\nWARNING: Number of observations does not equal "
            "Schema x Prompt x Model."
        )

    # Check uniqueness of treatment combinations
    combinations = (
        df.groupby(
            [SCHEMA, PROMPT, MODEL],
            observed=True
        )
        .size()
        .reset_index(name="n")
    )

    duplicates = combinations[
        combinations["n"] > 1
    ]

    if len(duplicates) > 0:
        print("\nWARNING: Duplicate Schema × Prompt × Model observations:")
        print(duplicates)
    else:
        print("\nNo duplicate treatment combinations found.")

    # Check missing combinations
    full_index = pd.MultiIndex.from_product(
        [
            df[SCHEMA].cat.categories,
            df[PROMPT].cat.categories,
            df[MODEL].cat.categories,
        ],
        names=[SCHEMA, PROMPT, MODEL],
    )

    observed_index = pd.MultiIndex.from_frame(
        df[[SCHEMA, PROMPT, MODEL]]
    )

    missing = full_index.difference(observed_index)

    if len(missing) > 0:
        print("\nWARNING: Missing combinations:")
        for x in missing:
            print(" ", x)
    else:
        print("All Schema × Prompt × Model combinations are present.")

    # Check whether each schema appears in all treatments
    schema_counts = (
        df.groupby(SCHEMA, observed=True)
        .size()
    )

    print("\nObservations per schema:")
    print(schema_counts)

    if not all(schema_counts == n_prompts * n_models):
        print(
            "\nWARNING: Not all schemas have all treatment combinations."
        )


# ============================================================
# 4. Descriptive statistics
# ============================================================

def descriptive_statistics(df):

    print_header("DESCRIPTIVE STATISTICS")

    all_outcomes = (
        COUNT_OUTCOMES
        + NLP_METRICS
    )

    rows = []

    for outcome in all_outcomes:

        grouped = (
            df.groupby(
                [PROMPT, MODEL],
                observed=True
            )[outcome]
        )

        summary = grouped.agg(
            [
                "count",
                "mean",
                "std",
                "median",
                "min",
                "max",
            ]
        ).reset_index()

        summary.insert(
            0,
            "outcome",
            outcome
        )

        rows.append(summary)

    result = pd.concat(
        rows,
        ignore_index=True
    )

    save_csv(
        result,
        "descriptive_statistics.csv"
    )

    print(result.to_string(index=False))

    return result


# ============================================================
# 5. Correlation analysis
# ============================================================

def correlation_analysis(df):

    print_header("DEPENDENT VARIABLE CORRELATIONS")

    families = {
        "structural": STRUCTURAL_FAMILY,
        "nlp": NLP_FAMILY,
    }

    for family_name, outcomes in families.items():

        corr = df[outcomes].corr(
            method="spearman"
        )

        print(f"\n{family_name.upper()} OUTCOMES")
        print(corr.round(3))

        save_csv(
            corr.reset_index(),
            f"correlations_{family_name}.csv"
        )


# ============================================================
# 6. LMM
# ============================================================

def fit_lmm(df, outcome):

    print_header(
        f"LMM: {outcome.upper()}"
    )

    formula = (
        f"{outcome} ~ "
        f"C({PROMPT}) * C({MODEL})"
    )

    print("Model:")
    print(
        f"  {outcome} ~ "
        f"Prompt * Model + (1 | Schema)"
    )

    model = smf.mixedlm(
        formula,
        data=df,
        groups=df[SCHEMA],
        re_formula="1",
    )

    # NOTE: "lbfgs" reliably converges but its approximate Hessian is
    # often not invertible for this design, which makes statsmodels
    # raise `LinAlgError: Singular matrix` while computing standard
    # errors (the optimization itself succeeds; only the SE step
    # fails). We fall back to sturdier optimizers instead of letting
    # the whole model fail.
    last_error = None
    result = None

    for method in ("lbfgs", "bfgs", "cg", "powell", "nm"):

        try:

            result = model.fit(
                reml=False,
                method=method,
                maxiter=2000,
                disp=False,
            )

            if method != "lbfgs":
                print(f"(fit converged using method='{method}')")

            break

        except Exception as e:

            last_error = e
            continue

    if result is None:
        raise RuntimeError(
            f"All optimizers failed for {outcome}: {last_error}"
        )

    print(result.summary())

    # Fixed effects
    fixed = pd.DataFrame({
        "parameter": result.params.index,
        "estimate": result.params.values,
        "std_error": result.bse.values,
        "z_or_t": result.tvalues.values,
        "p_value": result.pvalues.values,
    })

    save_csv(
        fixed,
        f"lmm_{outcome}_fixed_effects.csv"
    )

    # Full coefficient table
    summary = pd.DataFrame({
        "parameter": result.params.index,
        "estimate": result.params.values,
        "std_error": result.bse.values,
        "statistic": result.tvalues.values,
        "p_value": result.pvalues.values,
    })

    save_csv(
        summary,
        f"lmm_{outcome}.csv"
    )

    return result


# ============================================================
# 7. Binomial GLMM
# ============================================================

def check_bambi():

    try:
        import bambi
        import arviz

        return True

    except ImportError:

        print(
            "\nBambi/PyMC is required for the binomial GLMMs."
        )

        print(
            "\nInstall with:\n"
            "    pip install bambi pymc arviz"
        )

        return False


def fit_binomial_glmm(
    df,
    successes,
    trials,
    outcome_name,
):

    print_header(
        f"BINOMIAL GLMM: {outcome_name.upper()}"
    )

    import bambi as bmb
    import arviz as az

    data = df.copy()

    data["successes"] = data[successes].astype(int)
    data["trials"] = data[trials].astype(int)

    # Bambi/PyMC model.
    #
    # bmb.Model() has no `weights=` argument. Aggregated binomial
    # data (successes out of trials, rather than 0/1 outcomes) is
    # passed through the formula itself using Bambi's `p(successes,
    # trials)` syntax.
    formula = (
        f"p(successes, trials) ~ "
        f"{PROMPT} * {MODEL} + (1|{SCHEMA})"
    )

    print("Model:")
    print(
        f"  {outcome_name} ~ "
        f"Prompt * Model + (1 | Schema)"
    )

    model = bmb.Model(
        formula,
        data=data,
        family="binomial",
    )

    result = model.fit(
        draws=2000,
        tune=1000,
        chains=4,
        cores=4,
        target_accept=0.9,
        progressbar=False,
    )

    # arviz renamed `hdi_prob` to `ci_prob`/`ci_kind` in newer
    # releases; support both so the script works across versions.
    try:
        summary = az.summary(
            result,
            ci_prob=0.95,
            ci_kind="hdi",
        )
    except TypeError:
        summary = az.summary(
            result,
            hdi_prob=0.95
        )

    print(summary)

    summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"glmm_{outcome_name}.csv"
        )
    )

    return model, result


# ============================================================
# 8. Interaction plots
# ============================================================

def interaction_plot(df, outcome):

    means = (
        df.groupby(
            [PROMPT, MODEL],
            observed=True
        )[outcome]
        .mean()
        .reset_index()
    )

    plt.figure(
        figsize=(9, 6)
    )

    for model in means[MODEL].unique():

        subset = means[
            means[MODEL] == model
        ]

        plt.plot(
            subset[PROMPT].astype(str),
            subset[outcome],
            marker="o",
            linewidth=2,
            label=str(model),
        )

    plt.xlabel("Prompt")
    plt.ylabel(
        outcome.replace("_", " ").title()
    )

    plt.title(
        f"Prompt × Model Interaction\n"
        f"{outcome.replace('_', ' ').title()}"
    )

    plt.legend(
        title="Model"
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        f"interaction_{outcome}.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")


# ============================================================
# 9. LMM residual diagnostics
# ============================================================

def residual_diagnostics(
    result,
    outcome
):

    residuals = result.resid
    fitted = result.fittedvalues

    # Residual vs fitted
    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        fitted,
        residuals,
        alpha=0.7
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")

    plt.title(
        f"Residual Diagnostics: {outcome}"
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        f"residuals_{outcome}.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Q-Q plot
    plt.figure(
        figsize=(8, 6)
    )

    stats.probplot(
        residuals,
        dist="norm",
        plot=plt
    )

    plt.title(
        f"Normal Q-Q Plot: {outcome}"
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        f"qq_{outcome}.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 10. Paired descriptive comparisons
# ============================================================

def calculate_paired_comparisons(
    df,
    outcome,
    factor,
):

    """
    Provides simple paired comparisons for exploratory
    purposes.

    IMPORTANT:
        These are NOT the primary inferential tests.

    Primary inference comes from the mixed-effects model.
    """

    print_header(
        f"EXPLORATORY PAIRED COMPARISONS: {outcome.upper()}"
    )

    levels = list(
        df[factor].cat.categories
    )

    rows = []

    # Compare levels while pairing on Schema
    # and the other experimental factor.
    other_factor = (
        MODEL
        if factor == PROMPT
        else PROMPT
    )

    for other_level in df[
        other_factor
    ].cat.categories:

        subset = df[
            df[other_factor] == other_level
        ]

        for i in range(len(levels)):

            for j in range(i + 1, len(levels)):

                a = levels[i]
                b = levels[j]

                xa = (
                    subset[
                        subset[factor] == a
                    ]
                    .set_index(SCHEMA)[outcome]
                )

                xb = (
                    subset[
                        subset[factor] == b
                    ]
                    .set_index(SCHEMA)[outcome]
                )

                common = (
                    xa.index.intersection(
                        xb.index
                    )
                )

                if len(common) < 3:
                    continue

                statistic, p = stats.wilcoxon(
                    xa.loc[common],
                    xb.loc[common]
                )

                rows.append({
                    "other_factor": other_factor,
                    "other_level": str(other_level),
                    "factor": factor,
                    "level_1": str(a),
                    "level_2": str(b),
                    "n_pairs": len(common),
                    "statistic": statistic,
                    "p_value": p,
                })

    result = pd.DataFrame(rows)

    if len(result) > 0:

        result["p_adjusted"] = multipletests(
            result["p_value"],
            method="fdr_bh"
        )[1]

        save_csv(
            result,
            f"posthoc_{outcome}_{factor}.csv"
        )

        print(
            result.round(5)
            .to_string(index=False)
        )

    return result


# ============================================================
# 11. FDR correction across outcome families
# ============================================================

def collect_model_pvalues(
    model_results,
    family_name,
):

    rows = []

    for outcome, result in model_results.items():

        if result is None:
            continue

        # Exclude random variance parameters.
        for parameter, p in result.pvalues.items():

            if (
                "Group Var" in parameter
                or "Var" in parameter
            ):
                continue

            rows.append({
                "family": family_name,
                "outcome": outcome,
                "parameter": parameter,
                "p_value": p,
            })

    return pd.DataFrame(rows)


def collect_glmm_pvalues(
    glmm_results,
    family_name,
):

    """
    GLMM results are Bayesian (posterior draws from bambi/PyMC),
    so there is no classical p-value to collect the way
    collect_model_pvalues() does for the LMMs. As a frequentist-
    style analog for FDR correction, this computes a two-sided
    "probability of direction" p-value for each fixed-effect
    parameter: p = 2 * min(P(beta > 0), P(beta < 0)).

    Group-level (random effect) terms are excluded, matching how
    collect_model_pvalues() excludes variance parameters for the
    LMMs.
    """

    rows = []

    for outcome, result in glmm_results.items():

        if result is None:
            continue

        _, idata = result

        posterior = idata.posterior

        for varname in posterior.data_vars:

            # Skip group-level (random effect) terms, e.g.
            # "1|schema" and "1|schema_sigma".
            if "|" in varname or "sigma" in varname:
                continue

            da = posterior[varname]

            extra_dims = [
                d for d in da.dims
                if d not in ("chain", "draw")
            ]

            if extra_dims:

                dim = extra_dims[0]

                for level in da.coords[dim].values:

                    samples = da.sel(
                        {dim: level}
                    ).values.flatten()

                    p_dir = min(
                        (samples > 0).mean(),
                        (samples < 0).mean(),
                    )

                    rows.append({
                        "family": family_name,
                        "outcome": outcome,
                        "parameter": f"{varname}[{level}]",
                        "p_value": 2 * p_dir,
                    })

            else:

                samples = da.values.flatten()

                p_dir = min(
                    (samples > 0).mean(),
                    (samples < 0).mean(),
                )

                rows.append({
                    "family": family_name,
                    "outcome": outcome,
                    "parameter": varname,
                    "p_value": 2 * p_dir,
                })

    return pd.DataFrame(rows)


def apply_fdr(
    result_df,
    filename,
):

    if len(result_df) == 0:
        return result_df

    result_df = result_df.copy()

    result_df["p_fdr"] = multipletests(
        result_df["p_value"],
        method="fdr_bh"
    )[1]

    save_csv(
        result_df,
        filename
    )

    return result_df


# ============================================================
# 12. Main analysis
# ============================================================

def main():

    ensure_output_directory()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    sanity_check_ranges(df)

    # --------------------------------------------------------
    # Outcomes
    # --------------------------------------------------------

    df = create_outcomes(df)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_design(df)

    # --------------------------------------------------------
    # Descriptive analysis
    # --------------------------------------------------------

    descriptive_statistics(df)

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    correlation_analysis(df)

    # --------------------------------------------------------
    # Interaction plots
    # --------------------------------------------------------

    for outcome in (
        COUNT_OUTCOMES
        + NLP_METRICS
    ):
        interaction_plot(
            df,
            outcome
        )

    # --------------------------------------------------------
    # LMMs
    # --------------------------------------------------------

    lmm_results = {}

    for outcome in LMM_OUTCOMES:

        try:

            result = fit_lmm(
                df,
                outcome
            )

            lmm_results[outcome] = result

            residual_diagnostics(
                result,
                outcome
            )

        except Exception as e:

            print(
                f"\nERROR fitting LMM for {outcome}:"
            )
            print(e)

            lmm_results[outcome] = None

    # --------------------------------------------------------
    # GLMMs
    # --------------------------------------------------------

    glmm_results = {}

    if not GLMM_OUTCOMES:

        print_header("BINOMIAL GLMMs")

        print(
            "No outcomes currently use the binomial GLMM -- "
            "completeness, field_correctness and type_correctness "
            "are all ready-made ratios/distances rather than raw "
            "counts out of total_attributes, so they are fit with "
            "the continuous LMM instead (see GLMM_OUTCOMES / "
            "LMM_OUTCOMES)."
        )

    elif check_bambi():

        for outcome in GLMM_OUTCOMES:

            successes_col, trials_col = (
                GLMM_SUCCESS_COLUMNS[outcome]
            )

            try:

                glmm_results[outcome] = (
                    fit_binomial_glmm(
                        df,
                        successes=successes_col,
                        trials=trials_col,
                        outcome_name=outcome,
                    )
                )

            except Exception as e:

                print(
                    f"\nERROR fitting {outcome} GLMM:"
                )
                print(e)

    # --------------------------------------------------------
    # Exploratory post-hoc comparisons
    # --------------------------------------------------------

    for outcome in (
        COUNT_OUTCOMES
        + NLP_METRICS
    ):

        # Model differences within each prompt
        calculate_paired_comparisons(
            df,
            outcome,
            MODEL
        )

        # Prompt differences within each model
        calculate_paired_comparisons(
            df,
            outcome,
            PROMPT
        )

    # --------------------------------------------------------
    # FDR correction for LMM coefficient tests
    # --------------------------------------------------------

    nlp_lmm_results = {
        k: v for k, v in lmm_results.items()
        if k in NLP_FAMILY
    }

    lmm_pvalues = collect_model_pvalues(
        nlp_lmm_results,
        "nlp"
    )

    if len(lmm_pvalues) > 0:

        apply_fdr(
            lmm_pvalues,
            "lmm_fdr_results.csv"
        )

    # --------------------------------------------------------
    # FDR correction for the structural family (completeness from
    # its GLMM; field_correctness and type_correctness from their
    # LMMs) — combined into a single correction, since all three
    # belong to the same outcome family.
    # --------------------------------------------------------

    structural_lmm_results = {
        k: v for k, v in lmm_results.items()
        if k in STRUCTURAL_FAMILY
    }

    structural_pvalues = pd.concat(
        [
            collect_glmm_pvalues(
                glmm_results,
                "structural"
            ),
            collect_model_pvalues(
                structural_lmm_results,
                "structural"
            ),
        ],
        ignore_index=True,
    )

    if len(structural_pvalues) > 0:

        apply_fdr(
            structural_pvalues,
            "structural_fdr_results.csv"
        )

    # --------------------------------------------------------
    # Save analysis summary
    # --------------------------------------------------------

    summary_path = os.path.join(
        OUTPUT_DIR,
        "analysis_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "LLM Schema Denormalization Experiment\n"
        )

        f.write(
            "======================================\n\n"
        )

        f.write(
            "Experimental design:\n"
        )

        f.write(
            f"{EXPECTED_PROMPTS} prompts x {EXPECTED_MODELS} "
            f"models x {EXPECTED_SCHEMAS} schemas\n"
        )

        f.write(
            "Expected observations: "
            f"{EXPECTED_PROMPTS * EXPECTED_MODELS * EXPECTED_SCHEMAS}"
            "\n\n"
        )

        f.write(
            "Primary model:\n"
        )

        f.write(
            "Outcome ~ Prompt * Model + (1 | Schema)\n\n"
        )

        f.write(
            "LMM outcomes:\n"
        )

        for outcome in LMM_OUTCOMES:
            f.write(f"  - {outcome}\n")

        f.write(
            "\nBinomial GLMM outcomes:\n"
        )

        for outcome in GLMM_OUTCOMES:
            f.write(f"  - {outcome}\n")

        f.write(
            "\nOutcome families:\n"
        )

        f.write(
            "Structural: completeness, field_correctness, "
            "type_correctness\n"
        )

        f.write(
            "NLP: BLEU, ROUGE, METEOR\n"
        )

    print(
        f"\nAnalysis completed.\n"
        f"Results saved in: {OUTPUT_DIR}/"
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()