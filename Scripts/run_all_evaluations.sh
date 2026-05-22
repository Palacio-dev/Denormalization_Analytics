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

echo "================================================================================"
echo "Running Denormalization Evaluations"
echo "================================================================================"
echo "Experiment Schemes: $EXPERIMENT_DIR"
echo "Results Directory: $RESULTS_DIR"
echo "Reports Directory: $REPORTS_DIR"
echo "Report Structure: Reports/{Provider}/{Database}/"
echo ""

# Function to run evaluation for a single result file
run_evaluation() {
    local result_file="$1"
    local provider="$2"
    local db_name="$3"
    
    # Find corresponding experiment scheme
    local experiment_file="$EXPERIMENT_DIR/${db_name}.txt"
    
    if [ ! -f "$experiment_file" ]; then
        echo "⚠ WARNING: Experiment scheme not found for '$db_name': $experiment_file"
        return 1
    fi
    
    # Extract prompt type from filename
    # Format: experiment_TIMESTAMP_PROVIDER_DATABASE_PROMPT.txt
    local filename=$(basename "$result_file" .txt)
    local prompt=$(echo "$filename" | rev | cut -d'_' -f1 | rev)
    
    # Create directory structure: Reports/{Provider}/{Database_Name}/
    local report_dir="$REPORTS_DIR/$provider/$db_name"
    mkdir -p "$report_dir"
    
    # Create report filename with prompt type
    local report_name="report_${prompt}.txt"
    local report_file="$report_dir/$report_name"
    
    echo "Processing: $(basename "$result_file")"
    echo "  Provider: $provider"
    echo "  Database: $db_name"
    echo "  Prompt:   $prompt"
    echo "  Report:   $provider/$db_name/$report_name"
    
    # Run evaluation
    if python "$SCRIPT_DIR/evaluate_denormalization.py" \
        "$experiment_file" \
        "$result_file" \
        -o "$report_file"; then
        echo "  ✓ SUCCESS"
        ((SUCCESS++))
    else
        echo "  ✗ FAILED"
        ((FAILED++))
    fi
    
    ((TOTAL++))
    echo ""
}

# Process each result directory
for provider in "Gemini" "Ollama_API" "Ollama_local"; do
    PROVIDER_DIR="$RESULTS_DIR/$provider"
    
    if [ ! -d "$PROVIDER_DIR" ]; then
        echo "ℹ Directory not found: $PROVIDER_DIR (skipping)"
        continue
    fi
    
    echo "Processing $provider Results..."
    echo "--------"
    
    # Iterate through database folders: Results/{Provider}/{Database}/
    for db_dir in "$PROVIDER_DIR"/*; do
        # Skip if not a directory
        if [ ! -d "$db_dir" ]; then
            continue
        fi
        
        db_name=$(basename "$db_dir")
        
        # Iterate through all result files in the database folder
        for result_file in "$db_dir"/*.txt; do
            # Skip if no files found
            if [ ! -f "$result_file" ]; then
                continue
            fi
            
            # Run evaluation
            run_evaluation "$result_file" "$provider" "$db_name"
        done
    done
done

# Print summary
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo "Total Evaluations: $TOTAL"
echo "Successful:        $SUCCESS"
echo "Failed:            $FAILED"
echo "================================================================================"

# Exit with appropriate code
if [ $FAILED -gt 0 ]; then
    exit 1
else
    exit 0
fi
