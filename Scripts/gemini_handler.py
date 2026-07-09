"""
Gemini API handler for denormalization experiments.
Handles file uploads and LLM calls.
"""

import os
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv
from experiment_utils import ExperimentConfig


class GeminiHandler:
    """Handler for Gemini API interactions."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize Gemini handler.
        
        Args:
            config: ExperimentConfig object with paths and settings
        """
        self.config = config
        self.model = "gemini-3.1-pro-preview"
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Gemini client with API key from environment."""
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment. "
                "Please set it in .env file or as environment variable."
            )
        
        self.client = genai.Client(api_key=api_key)
    

    def generate_denormalization(self, prompt_with_schema: str) -> str:
        """
        Generate denormalized schema using Gemini API.
        
        Args:
            prompt_with_schema: Prompt text with schema already injected
            
        Returns:
            LLM generated denormalized schema as string
        """
        try:
            # Configure generation parameters
            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_level="high"
                ),
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_LOW_AND_ABOVE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_LOW_AND_ABOVE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_LOW_AND_ABOVE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_LOW_AND_ABOVE",
                    ),
                ],
                temperature=1.0,
                topP=0.65,
                topK=10,
                maxOutputTokens=65536,
            )
            print("  Calling Gemini API ...")
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt_with_schema,
                config=generate_content_config,
            )
            return response.text
        
        except Exception as e:
            raise RuntimeError(f"Failed to generate denormalization with Gemini: {e}")
    
    def run_experiment(self, prompt_content: str, schema_content: str) -> str:
        """
        Run complete denormalization experiment with Gemini.
        
        Args:
            prompt_content: Prompt text (will have schema injected)
            schema_content: Schema content to denormalize
            
        Returns:
            LLM generated denormalized schema
        """
        # For Gemini, we inject schema into prompt text
        # but also upload the file separately for file API usage
        from experiment_utils import inject_schema
        
        prompt_with_schema = inject_schema(prompt_content, schema_content)
            
        result = self.generate_denormalization(prompt_with_schema)
        return result
        