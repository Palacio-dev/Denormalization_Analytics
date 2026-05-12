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

4. Set the enviromment
- In order to run the experiment with the Gemini's API, you first need to set one Gemini API key , you can read more about it here : [GEMNI_API](https://ai.google.dev/gemini-api/docs?hl=pt-br). Once you got acess to a API, create a .env file in the root of the repository and create a variable called GEMINI_API_KEY.
 ```bash
	GEMINI_API_KEY="your_gemini_api_key_here"
```
 - You can also run the experiment locally with free models available at Ollama. There are two possibilites, running locally with the model qwen3.5:9b or running via the Ollama's API with the model qwen3-coder:480b-cloud. For running locally, you need to download the model on your machine. It can be done with the command in the terminal:
     
```bash
 ollama pull qwen3.5:9b
```
Then, run ollama:
```bash
 ollama serve
```

For running via the Ollama's API, you first need to create an account in [Ollama's site](https://ollama.com/) and get an API key. Then, put it on the .env file with the variable OLLAMA_API_KEY.
 ```bash
	OLLAMA_API_KEY="your_ollama_api_key_here"
```
   


5. Run the denormalization experiment

Currently there are 3 available LLM models, 7 prompts and 17 normalziled schemas to run the experiment. You will select one possibility of each when running the experiment.

```bash
cd Scripts
python3 run_experiment.py
```

The results are automatically saved in the Results folder.  

6. Evaluate denormalization
   
- To evaluate denormalization outputs with available metrics (BLEU, ROUGE, METEOR):

```bash
cd Scripts
python3 evaluate_denormalization.py   -"Path to the relational model file"  -"Path to the denormalized model file"
-o "Output file for the report (optional)"
```

## Evaluation metrics

This project uses common textual similarity metrics to evaluate the plausibility and fidelity of denormalized outputs:

- BLEU — n-gram overlap precision-based metric.
- ROUGE — recall-focused metric family often used for summarization comparisons.
- METEOR — alignment- and synonym-aware metric that can complement BLEU and ROUGE.





