# SQL Model Denormalization Evaluation Script

## Overview
This script automatically evaluates the quality of denormalized SQL models compared to their relational counterparts using NLP metrics (BLEU, ROUGE, METEOR) and structural analysis.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python evaluate_denormalization.py <relational_model.txt> <denormalized_model.txt>
```

### With Output File
```bash
python evaluate_denormalization.py <relational_model.txt> <denormalized_model.txt> -o report.txt
```

### Example
```bash
python evaluate_denormalization.py \
    ../Esquemas_Benchmarks/harperdb.txt \
    ../Res/from\ 15-01/harper.txt \
    -o harper_evaluation_report.txt
```

## What the Script Does

1. **Parses SQL Files**: Reads and parses both SQL model files
2. **Pairs Identifiers**: Automatically matches equivalent columns between models
3. **Calculates Completeness**: Evaluates if information was lost or added
4. **Evaluates Correctness**: Checks data type preservation and semantic consistency
5. **Computes Metrics**: Calculates BLEU, ROUGE-1, ROUGE-2, ROUGE-L, and METEOR scores
6. **Generates Report**: Produces a comprehensive evaluation report

## Evaluation Criteria

### Completeness
- **Ratio = 1.0**: Ideal - no information added or lost
- **Ratio > 1.0**: Information may have been lost
- **Ratio < 1.0**: New information may have been added

### Correctness
- Data type preservation
- Semantic identifier preservation
- Constraint preservation

### Metrics
- **BLEU**: Measures n-gram overlap (scale: 0-1, higher is better)
- **ROUGE**: Measures recall-oriented similarity (scale: 0-1, higher is better)
- **METEOR**: Considers synonyms and stemming (scale: 0-1, higher is better)

## Output Format

The script generates a detailed report with:
1. Model overview (tables and attributes count)
2. Completeness analysis
3. Correctness evaluation with issue detection
4. Pair-by-pair metric comparison
5. Overall metrics summary
6. Final conclusion and recommendations

## Notes

- The script automatically downloads required NLTK resources on first run
- Attribute pairing uses intelligent matching based on column names and types
- The script handles composite keys and foreign key relationships
