#!/bin/bash
#################################################################################
# Script: run_all_evaluations.sh
# Description: Runs evaluate_denormalization.py for all denormalized models
#              in Results/{Gemini,Ollama_API,Ollama_local} directories
#################################################################################

# Get the script directory (Scripts folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Define paths
EXPERIMENT_DIR="$PROJECT_ROOT/Experiment_schemes/Train"
RESULTS_DIR="$PROJECT_ROOT/Results"
REPORTS_DIR="$PROJECT_ROOT/Reports"

# Counter for statistics
TOTAL=0
SUCCESS=0
FAILED=0
SKIPPED=0

echo "================================================================================"
echo "Running Denormalization Evaluations"
echo "================================================================================"
echo "Script Dir:        $SCRIPT_DIR"
echo "Project Root:      $PROJECT_ROOT"
echo "Experiment Dir:    $EXPERIMENT_DIR"
echo "Results Dir:       $RESULTS_DIR"
echo "Reports Dir:       $REPORTS_DIR"
echo "Report Structure:  Reports/{Provider}/{Database}/report_{prompt}.txt"
echo "================================================================================"
echo ""

# ── Validate that the evaluator script exists ────────────────────────────────
EVALUATOR="$SCRIPT_DIR/evaluate_denormalization.py"
if [ ! -f "$EVALUATOR" ]; then
    echo "✗ FATAL: evaluate_denormalization.py not found at: $EVALUATOR"
    exit 1
fi

# ── Helper: extract prompt tag from filename ──────────────────────────────────
# Expected filename format: experiment_<TIMESTAMP>_<anything>_<PROMPT>.txt
# The PROMPT is the last underscore-delimited segment before .txt.
# We use the folder's db_name (which already handles multi-word DB names like
# adv_works, TPC_DS) so we do NOT try to re-parse the DB name from the filename.
extract_prompt() {
    local filename
    filename="$(basename "$1" .txt)"   # strip directory and .txt

    # Last field after the final underscore
    local prompt
    prompt="${filename##*_}"

    # Guard: if extraction produced an empty string or the whole filename
    # (no underscore found), fall back to "unknown"
    if [ -z "$prompt" ] || [ "$prompt" = "$filename" ]; then
        echo "unknown"
    else
        echo "$prompt"
    fi
}

# ── Core evaluation function ──────────────────────────────────────────────────
run_evaluation() {
    local result_file="$1"
    local provider="$2"
    local db_name="$3"

    ((TOTAL++))

    # Find the corresponding relational schema
    local experiment_file="$EXPERIMENT_DIR/${db_name}.txt"
    if [ ! -f "$experiment_file" ]; then
        echo "  ⚠ WARNING: Relational schema not found: $experiment_file"
        echo "             Skipping $(basename "$result_file")"
        ((SKIPPED++))
        echo ""
        return 1
    fi

    # Extract the prompt engineering technique from the filename
    local prompt
    prompt="$(extract_prompt "$result_file")"

    # Build output directory and report path
    local report_dir="$REPORTS_DIR/$provider/$db_name"
    mkdir -p "$report_dir"

    local report_file="$report_dir/report_${prompt}.txt"

    echo "  File:     $(basename "$result_file")"
    echo "  Schema:   $(basename "$experiment_file")"
    echo "  Prompt:   $prompt"
    echo "  Report:   Reports/$provider/$db_name/report_${prompt}.txt"

    # Run the Python evaluator
    if python "$EVALUATOR" \
            "$experiment_file" \
            "$result_file" \
            -o "$report_file"; then
        echo "  ✓ SUCCESS"
        ((SUCCESS++))
    else
        echo "  ✗ FAILED (evaluate_denormalization.py returned non-zero)"
        ((FAILED++))
    fi

    echo ""
}

# ── Main loop: iterate Providers → Databases → Result files ──────────────────
for provider in "Gemini" "Ollama_API" "Ollama_local"; do

    PROVIDER_DIR="$RESULTS_DIR/$provider"

    if [ ! -d "$PROVIDER_DIR" ]; then
        echo "ℹ  Provider directory not found, skipping: $PROVIDER_DIR"
        echo ""
        continue
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Provider: $provider"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Iterate through each database sub-folder
    for db_dir in "$PROVIDER_DIR"/*/; do

        [ -d "$db_dir" ] || continue          # skip stray non-directories

        db_name="$(basename "$db_dir")"

        echo "  ── Database: $db_name"

        # Collect .txt files; handle the case where none exist
        shopt -s nullglob
        result_files=("$db_dir"*.txt)
        shopt -u nullglob

        if [ ${#result_files[@]} -eq 0 ]; then
            echo "  ⚠ No .txt result files found in $db_dir"
            echo ""
            continue
        fi

        for result_file in "${result_files[@]}"; do
            [ -f "$result_file" ] || continue
            run_evaluation "$result_file" "$provider" "$db_name"
        done

    done

done

# ── Summary ───────────────────────────────────────────────────────────────────
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
printf "  Total processed : %d\n"  "$TOTAL"
printf "  ✓ Successful    : %d\n"  "$SUCCESS"
printf "  ✗ Failed        : %d\n"  "$FAILED"
printf "  ⚠ Skipped       : %d\n"  "$SKIPPED"
echo "================================================================================"

# Exit non-zero only when actual evaluation calls failed (not skipped)
if [ "$FAILED" -gt 0 ]; then
    exit 1
else
    exit 0
fi