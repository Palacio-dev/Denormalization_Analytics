#!/usr/bin/env python3
"""
Denormalization Experiment Runner.
Interactive CLI for running denormalization experiments with different LLMs, prompts, and schemas.
"""

import sys
from pathlib import Path
from experiment_utils import (
    ExperimentConfig,
    load_prompts,
    load_schemas,
    save_result,
    get_prompt_schema_names,
)
from gemini_handler import GeminiHandler
from qwen_handler import QwenHandler


def print_header():
    """Print application header."""
    print("\n" + "=" * 60)
    print("  DENORMALIZATION EXPERIMENT RUNNER")
    print("=" * 60 + "\n")


def select_model() -> tuple[str, str]:
    """
    Interactive menu to select model.
    
    Returns:
        Tuple of (model_key, model_display_name)
    """
    print("SELECT MODEL:")
    print("  1. Gemini 3.1 Pro-Preview (API)")
    print("  2. Ollama Qwen (Cloud API)")
    print("  3. Ollama Qwen (Local)")
    print()
    
    while True:
        try:
            choice = input("Enter choice (1-3): ").strip()
            if choice == "1":
                return ("gemini", "Gemini 3.1 Pro-Preview")
            elif choice == "2":
                return ("ollama_api", "Ollama Qwen (Cloud API)")
            elif choice == "3":
                return ("ollama_local", "Ollama Qwen (Local)")
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)


def select_prompt(prompt_names: list[str]) -> tuple[str, str]:
    """
    Interactive menu to select prompt.
    
    Args:
        prompt_names: List of available prompt filenames
        
    Returns:
        Tuple of (filename, display_name without .txt)
    """
    print("\nSELECT PROMPT:")
    for i, name in enumerate(prompt_names, 1):
        display_name = name.replace(".txt", "")
        print(f"  {i}. {display_name}")
    print()
    
    while True:
        try:
            choice = input(f"Enter choice (1-{len(prompt_names)}): ").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(prompt_names):
                    return (prompt_names[idx], prompt_names[idx].replace(".txt", ""))
                else:
                    print(f"Invalid choice. Please enter 1-{len(prompt_names)}.")
            except ValueError:
                print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)


def select_schema(schema_names: list[str]) -> tuple[str, str]:
    """
    Interactive menu to select schema.
    
    Args:
        schema_names: List of available schema filenames
        
    Returns:
        Tuple of (filename, display_name without .txt)
    """
    print("\nSELECT SCHEMA:")
    for i, name in enumerate(schema_names, 1):
        display_name = name.replace(".txt", "")
        print(f"  {i:2d}. {display_name}")
    print()
    
    while True:
        try:
            choice = input(f"Enter choice (1-{len(schema_names)}): ").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(schema_names):
                    return (schema_names[idx], schema_names[idx].replace(".txt", ""))
                else:
                    print(f"Invalid choice. Please enter 1-{len(schema_names)}.")
            except ValueError:
                print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)


def run_experiment(model_key: str, prompt_name: str, prompt_content: str,
                   schema_name: str, schema_content: str,
                   config: ExperimentConfig) -> str:
    """
    Run experiment with selected model, prompt, and schema.
    
    Args:
        model_key: Model identifier ("gemini", "ollama_api", "ollama_local")
        prompt_name: Name of prompt file
        prompt_content: Content of prompt
        schema_name: Name of schema file
        schema_content: Content of schema
        config: ExperimentConfig object
        
    Returns:
        LLM output
    """
    print("\n" + "=" * 60)
    print(f"Running experiment: {model_key} + {prompt_name} + {schema_name}")
    print("=" * 60 + "\n")
    
    try:
        if model_key == "gemini":
            handler = GeminiHandler(config)
            result = handler.run_experiment(prompt_content, schema_content)
        
        elif model_key == "ollama_api":
            handler = QwenHandler(config, mode="api")
            result = handler.run_experiment(prompt_content, schema_content)
        
        elif model_key == "ollama_local":
            handler = QwenHandler(config, mode="local")
            result = handler.run_experiment(prompt_content, schema_content)
        
        else:
            raise ValueError(f"Unknown model: {model_key}")
        
        return result
    
    except Exception as e:
        print(f"\nError running experiment: {e}")
        raise


def main():
    """Main entry point."""
    print_header()
    
    # Initialize config
    config = ExperimentConfig()
    
    # Load available prompts and schemas
    print("Loading prompts and schemas...")
    prompts = load_prompts(config)
    schemas = load_schemas(config)
    
    if not prompts:
        print("Error: No prompts found!")
        sys.exit(1)
    
    if not schemas:
        print("Error: No schemas found!")
        sys.exit(1)
    
    prompt_names, schema_names = get_prompt_schema_names(prompts, schemas)
    print(f"✓ Loaded {len(prompts)} prompts and {len(schemas)} schemas\n")
    
    # Interactive selection
    model_key, model_display = select_model()
    prompt_file, prompt_display = select_prompt(prompt_names)
    schema_file, schema_display = select_schema(schema_names)
    
    # Confirmation
    print("\n" + "-" * 60)
    print(f"Model:   {model_display}")
    print(f"Prompt:  {prompt_display}")
    print(f"Schema:  {schema_display}")
    print("-" * 60)
    
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != "y":
        print("Experiment cancelled.")
        sys.exit(0)
    
    # Get prompt and schema content
    prompt_content = prompts[prompt_file]
    schema_content = schemas[schema_file]
    
    # Run experiment
    try:
        result = run_experiment(
            model_key,
            prompt_display,
            prompt_content,
            schema_display,
            schema_content,
            config
        )
        
        # Save result
        print("\n" + "=" * 60)
        print("Saving result...")
        result_path = save_result(model_key, prompt_display, schema_display, result, config)
        print(f"✓ Result saved to: {result_path}")
        print("=" * 60 + "\n")
    
    except Exception as e:
        print(f"\nExperiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
