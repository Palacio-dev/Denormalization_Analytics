MODELS_MAP = {
    "Gemini": "Gemini",
    "Ollama_api": "Qwen3-code:480b",
    "Ollama_local": "Qwen3.5:4b"
}

SCHEMAS_MAP = {
    "adv_works": "Adventure_Works",
    "Airplane": "Airplane",
    "chinook": "Chinook",
    "employees": "Employees",
    "ex02": "Ex02",
    "ex04": "Ex04",
    "ex06": "Ex06",
    "ex07": "Ex07",
    "ex08": "Ex08",
    "ex10": "Ex10",
    "harperdb": "HarperDB",
    "HFC": "HFC",
    "northwind": "NorthWind",
    "solarwinds": "SolarWinds",
    "TPC_DS": "TPC-DS",
    "TPC_H": "TPC-H"
}

PROMPTS_MAP = {
    "JUNIOR": "junior",
    "MID-LEVEL": "mid-level",
    "OBT": "OBT",
    "RISEN-FEW-SHOT": "RISEN-few-shot",
    "RISEN": "RISEN",
    "SENIOR": "senior"
}

METRICS_MAP = {
    "Completeness Ratio":"Completude",
    "Data Type Preservation": "Corretude de tipo",
    "Avg Levenshtein Distance": "LD médio",
    "BLEU Score": "BLEU",
    "ROUGE-1": "ROUGE",
    "METEOR Score": "METEOR"
}

def main():
    csv = []
    for schema in SCHEMAS_MAP:
        for metrics in METRICS_MAP:
            row = [SCHEMAS_MAP[schema]]
            row.append(METRICS_MAP[metrics])
            for model in MODELS_MAP:
                for prompt in PROMPTS_MAP:
                    try:
                        with open(f"Reports/{model}/{schema}/report_{prompt}.txt", "r") as f:
                            lines = f.readlines()
                    except FileNotFoundError:
                        row.append("-1")
                        continue

                    line = next((l for l in lines if l.startswith(f"{metrics}:")), None)
                    row.append(line.split(":")[1].strip() if line else "N/A")
                row.append("")
            csv.append(','.join(row))
        csv.append("\n")

    print('\n'.join(csv))

if __name__ == "__main__":
    main()