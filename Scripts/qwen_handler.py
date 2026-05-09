"""
Ollama/Qwen handler for denormalization experiments.
Supports both Cloud API and local Ollama instances.
"""

import os
from typing import Literal
from ollama import Client
from dotenv import load_dotenv
from experiment_utils import ExperimentConfig


class QwenHandler:
    """Handler for Ollama Qwen model interactions."""
    
    def __init__(self, config: ExperimentConfig, mode: Literal["api", "local"] = "api"):
        """
        Initialize Qwen handler.
        
        Args:
            config: ExperimentConfig object with paths and settings
            mode: "api" for cloud API, "local" for local instance
        """
        self.config = config
        self.mode = mode
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Ollama client based on mode."""
        load_dotenv()
        
        if self.mode == "api":
            # Cloud API mode
            api_key = os.getenv("OLLAMA_API_KEY")
            if not api_key:
                raise ValueError(
                    "OLLAMA_API_KEY not found in environment. "
                    "Please set it in .env file for cloud API mode."
                )
            
            self.client = Client(
                host="https://ollama.com",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            self.model = "qwen3-coder:480b-cloud"
        
        elif self.mode == "local":
            # Local instance mode
            self.client = Client()  # Defaults to localhost
            self.model = "qwen3.5:9b"
        
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Use 'api' or 'local'.")
    
    def generate_denormalization(self, formatted_prompt: str) -> str:
        """
        Generate denormalized schema using Qwen model.
        
        Args:
            formatted_prompt: Complete formatted prompt with schema embedded
            
        Returns:
            LLM generated denormalized schema as string
        """
        try:
            mode_desc = "Cloud API" if self.mode == "api" else "Local"
            print(f"  Calling Ollama Qwen ({mode_desc})...")
            
            response = self.client.generate(
                model=self.model,
                prompt=formatted_prompt,
                stream=False,  # Get full response at once
            )
            
            response_text = response.get("response", "")
            return response_text
        
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate denormalization with Qwen ({self.mode}): {e}"
            )
    
    def run_experiment(self, prompt_content: str, schema_content: str) -> str:
        """
        Run complete denormalization experiment with Qwen.
        
        Args:
            prompt_content: Prompt text (will have schema injected)
            schema_content: Schema content to denormalize
            
        Returns:
            LLM generated denormalized schema
        """
        from experiment_utils import inject_schema
        
        # Inject schema into prompt
        formatted_prompt = inject_schema(prompt_content, schema_content)
        
        # Generate denormalization
        result = self.generate_denormalization(formatted_prompt)
        
        # Stream output to stdout for user to see
        if result:
            print(result)
        
        return result
