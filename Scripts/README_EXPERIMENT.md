#!/usr/bin/env python3
"""
DENORMALIZATION EXPERIMENT RUNNER
==================================

This script provides an interactive CLI for running denormalization experiments
with three different LLM models, seven prompts, and seventeen schemas.

SETUP REQUIREMENTS
==================

1. Install required packages:
   pip install google-genai ollama python-dotenv

2. Create a .env file in the Scripts/ directory with:
   - For Gemini: GEMINI_API_KEY=your_key_here
   - For Ollama Cloud API: OLLAMA_API_KEY=your_key_here

3. For local Ollama, ensure your local instance is running on localhost:11434

AVAILABLE OPTIONS
=================

Models (3 options):
  1. Gemini 3.1 Pro-Preview (API) - requires GEMINI_API_KEY
  2. Ollama Qwen (Cloud API) - requires OLLAMA_API_KEY  
  3. Ollama Qwen (Local) - requires local Ollama instance running

Prompts (7 options):
  - naive.txt (simple prompt)
  - naive_with_example.txt (with SQL examples)
  - RACE.txt (framework: Role-Action-Context-Examples)
  - RISEN.txt (framework: Role-Input-Steps-Expectation-Narrowing)
  - begginer.txt (beginner-level guidance)
  - intermediate.txt (intermediate-level guidance)
  - expert.txt (expert-level guidance)

Schemas (17 options):
  - All schema files from Experiment_schemes/Train/

WORKFLOW
========

1. Run the experiment:
   python3 run_experiment.py

2. Choose your model (1-3)

3. Choose your prompt (1-7)

4. Choose your schema (1-17)

5. Confirm your selection

6. The LLM will generate the denormalized schema:
   - Output streams to terminal in real-time
   - Result is automatically saved to Results/ folder

OUTPUT FILES
============

Results are saved as:
  Results/experiment_YYYY-MM-DD_HH-MM-SS_MODEL_SCHEMA_PROMPT.txt

Each file contains:
  - Metadata header (timestamp, model, prompt, schema)
  - Full LLM-generated denormalized schema output

EXAMPLE RUNS
============

Example 1: Gemini + RACE + harperdb
  1. Select: 1 (Gemini)
  2. Select: 3 (RACE.txt)
  3. Select: Harperdb schema
  4. Confirm: y

Example 2: Ollama Local + beginner + employees
  1. Select: 3 (Ollama Local)
  2. Select: 5 (begginer.txt)
  3. Select: employees schema
  4. Confirm: y

TECHNICAL NOTES
===============

Schema Injection:
  - For Gemini: Schema is injected directly into prompt text
  - For Qwen: Schema is injected directly into prompt text
  - All prompts use {SCHEMA_CONTENT} placeholder

Result Handling:
  - LLM output streams to stdout as it's generated
  - Full result is captured and saved after generation completes
  - Timestamps are in YYYY-MM-DD_HH-MM-SS format

File Caching (Gemini only):
  - Uploaded schema file IDs are cached in gemini_cache.json
  - Subsequent runs with the same schema may reuse cached IDs

ERROR HANDLING
==============

If you encounter errors:

1. ImportError (google-genai, ollama):
   - Install missing packages: pip install <package>

2. API key errors:
   - Check .env file has correct keys
   - Verify keys are valid and have proper permissions

3. Connection errors (Ollama):
   - For local: Ensure Ollama is running on localhost:11434
   - For cloud: Check network connection and API endpoint

4. Schema/Prompt loading errors:
   - Verify file structure in Experiment_schemes/Train/
   - Verify prompts exist in Prompts/Prompts_written/ and Prompts_generated/

TROUBLESHOOTING
===============

Q: "GEMINI_API_KEY not found"
A: Create .env file with: GEMINI_API_KEY=your_actual_key

Q: Connection refused (Ollama local)
A: Start local Ollama first: ollama serve

Q: No prompts found
A: Check Prompts/ directory structure is correct

Q: Results directory not created
A: Script auto-creates Results/ - check directory permissions
"""

print(__doc__)

# Show quick status
if __name__ == "__main__":
    from experiment_utils import ExperimentConfig, load_prompts, load_schemas
    
    print("\n" + "=" * 70)
    print("SYSTEM STATUS")
    print("=" * 70)
    
    config = ExperimentConfig()
    prompts = load_prompts(config)
    schemas = load_schemas(config)
    
    print(f"✓ Configuration initialized")
    print(f"  Base directory: {config.base_dir}")
    print(f"  Results directory: {config.results_dir}")
    print(f"  Gemini cache: {config.gemini_cache_file}")
    print()
    print(f"✓ Prompts loaded: {len(prompts)}")
    for name in sorted(prompts.keys()):
        if name.endswith('.txt'):
            clean_name = name.replace('.txt', '')
            has_placeholder = "✓" if "{SCHEMA_CONTENT}" in prompts[name] else "✗"
            print(f"  {has_placeholder} {clean_name}")
    print()
    print(f"✓ Schemas loaded: {len(schemas)}")
    print(f"  Available schemas in Train/ folder")
    print()
    print("=" * 70)
    print("Ready to run: python3 run_experiment.py")
    print("=" * 70)
