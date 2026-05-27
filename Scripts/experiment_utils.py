"""
Utility functions for denormalization experiment runner.
Handles prompt loading, schema loading, schema injection, and result saving.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple


class ExperimentConfig:
    """Configuration paths for the experiment runner."""
    
    def __init__(self, base_dir: str = None):
        """Initialize config with base directory."""
        if base_dir is None:
            # Assume we're in Scripts/ directory
            base_dir = Path(__file__).parent.parent
        else:
            base_dir = Path(base_dir)
        
        self.base_dir = base_dir
        self.prompts_written_dir = base_dir / "Prompts" / "Written"
        self.prompts_generated_dir = base_dir / "Prompts" / "By_user_level"
        self.schemas_dir = base_dir / "Experiment_schemes" / "Train"
        self.results_dir = base_dir / "Results"
        self.scripts_dir = base_dir / "Scripts"
        self.results_dir.mkdir(parents=True, exist_ok=True)


def load_prompts(config: ExperimentConfig) -> Dict[str, str]:
    """
    Load all available prompts (written + generated).
    
    Args:
        config: ExperimentConfig object with paths
        
    Returns:
        Dictionary mapping prompt filename to prompt content
    """
    prompts = {}
    
    # Load written prompts
    if config.prompts_written_dir.exists():
        for prompt_file in sorted(config.prompts_written_dir.glob("*.txt")):
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompts[prompt_file.name] = f.read()
            except Exception as e:
                print(f"Warning: Failed to load prompt {prompt_file.name}: {e}")
    
    # Load generated prompts
    if config.prompts_generated_dir.exists():
        for prompt_file in sorted(config.prompts_generated_dir.glob("*.txt")):
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompts[prompt_file.name] = f.read()
            except Exception as e:
                print(f"Warning: Failed to load prompt {prompt_file.name}: {e}")
    
    return prompts


def load_schemas(config: ExperimentConfig) -> Dict[str, str]:
    """
    Load all available schemas from Train folder.
    
    Args:
        config: ExperimentConfig object with paths
        
    Returns:
        Dictionary mapping schema filename to schema content
    """
    schemas = {}
    
    if config.schemas_dir.exists():
        for schema_file in sorted(config.schemas_dir.glob("*.txt")):
            try:
                with open(schema_file, 'r', encoding='utf-8') as f:
                    schemas[schema_file.name] = f.read()
            except Exception as e:
                print(f"Warning: Failed to load schema {schema_file.name}: {e}")
    
    return schemas


def inject_schema(prompt_content: str, schema_content: str) -> str:
    """
    Inject schema content into prompt by replacing {SCHEMA_CONTENT} placeholder.
    
    Args:
        prompt_content: Prompt text with {SCHEMA_CONTENT} placeholder
        schema_content: Actual schema content to inject
        
    Returns:
        Prompt with schema injected
    """
    return prompt_content.replace("{SCHEMA_CONTENT}", schema_content)

def get_formatted_model_name(model_key: str) -> str:
    """Convert model key to formatted name for directory structure."""

    if model_key == "gemini":
        return "Gemini"
    elif model_key == "ollama_api":
        return "Ollama_API"
    elif model_key == "ollama_local":
        return "Ollama_local"
    
    raise ValueError(f"Unknown model key: {model_key}")

def save_result(model_name: str, prompt_name: str, schema_name: str, 
                output_text: str, config: ExperimentConfig) -> str:
    """
    Save experiment result to timestamped file.
    
    Args:
        model_name: Name of model used (e.g., "gemini", "ollama_api", "ollama_local")
        prompt_name: Name of prompt file (without extension)
        schema_name: Name of schema file (without extension)
        output_text: LLM output to save
        config: ExperimentConfig object with paths
        
    Returns:
        Path to saved file
    """
    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Clean up names (remove .txt extensions if present)
    prompt_clean = prompt_name.replace(".txt", "")
    schema_clean = schema_name.replace(".txt", "")

    filename = f"experiment_{timestamp}_{model_name}_{schema_clean}_{prompt_clean}.txt"
    filepath = config.results_dir / get_formatted_model_name(model_name) / schema_clean / filename
    
    # Save result
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # Write metadata header
            f.write(f"=== EXPERIMENT RESULT ===\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Prompt: {prompt_name}\n")
            f.write(f"Schema: {schema_name}\n")
            f.write(f"=" * 50 + "\n\n")
            # Write actual output
            f.write(output_text)
        
        return str(filepath)
    except Exception as e:
        print(f"Error saving result: {e}")
        raise


def get_prompt_schema_names(prompts: Dict[str, str], schemas: Dict[str, str]) -> Tuple[list, list]:
    """
    Get sorted lists of prompt and schema names for display.
    
    Args:
        prompts: Dictionary of prompts
        schemas: Dictionary of schemas
        
    Returns:
        Tuple of (sorted prompt names, sorted schema names)
    """
    return sorted(prompts.keys()), sorted(schemas.keys())
