# Denormalization Analytics 

This repository is part of the project **Prompt Engineering for Data Engineering aimed at Data Analytics**, which explores how prompt engineering techniques and LLMs can help automate the denormalization of relational models and the creation of ETL pipelines to better serve analytical workloads.

Advisor: Professor Breno Bernard Nicolau de França

## Getting started 

Follow these steps to run the main scripts locally. The examples below assume a Linux / bash environment.

1. Clone the repository

```bash
git clone https://github.com/Palacio-dev/Denormalization_Analytics.git
cd Denormalization_Analytics
```

2. Create and activate a Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Set the Gemini API key , you can read more about it here : [GEMNI_API](https://ai.google.dev/gemini-api/docs?hl=pt-br)   
As the scripts use the API of GEMINI, you will have to get an API key and put it into a file named .env in the root of the project.
In the .env file put:
```bash
GEMINI_API_KEY="your_gemini_api_key_here"
```

5. Run the main scripts

- To generate denormalized models via the Risen pipeline:
- Important: You have to set the schema that will be denormalized at the top of the Risen.py file giving the path to a txt file. Example below:

```bash
cd Scripts
schema_file = "../Benchmarks_schemes/TPC_H.txt"
python3 Risen.py
```

- To evaluate denormalization outputs with available metrics (BLEU, ROUGE, METEOR):

```bash
cd Scripts
python3 evaluate_denormalization.py   -"Path to the relational model file"  -"Path to the denormalized model file"
-o "Output file for the report (optional)"
```

## Repository structure

- `Benchmarks_schemes/` — Normalized relational schemas used as input benchmarks for denormalization experiments. Files include schemas for HarperDB, RTA benchmarks, SolarWinds, TPC-H.
- `Denormalized_models/` — Denormalized relational models produced by the API (organized by experiment / timestamp). These are the outputs generated during model runs.
- `Metrics/` — Notes and intermediate outputs about the evaluation metrics used (BLEU, ROUGE, METEOR) to measure similarity between generated denormalized models and reference outputs.
- `Prompts/` — Prompt templates and examples used to interact with the LLMs during experiments. These contain variations used to test prompt engineering strategies.
- `Reports/` — Evaluation results and human-readable reports produced by `Scripts/evaluate_denormalization.py`.
- `Scripts/` — Python scripts that run experiments and evaluations. The most relevant scripts:
    - `RACE.py` — pipeline script used to generate denormalized models via the LLM (RACE pipeline).
	- `RISEN.py` — pipeline script used to generate denormalized models via the LLM (RISEN pipeline).
	- `evaluate_denormalization.py` — evaluates generated denormalized models using BLEU / ROUGE / METEOR and produces reports in `Reports/`.

## Evaluation metrics

This project uses common textual similarity metrics to evaluate the plausibility and fidelity of denormalized outputs:

- BLEU — n-gram overlap precision-based metric.
- ROUGE — recall-focused metric family often used for summarization comparisons.
- METEOR — alignment- and synonym-aware metric that can complement BLEU and ROUGE.

Results and intermediate outputs for these metrics are stored and discussed in the `Metrics/` and `Reports/` folders.



