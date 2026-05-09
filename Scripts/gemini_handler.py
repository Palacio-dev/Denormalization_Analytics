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
    
    def upload_schema_file(self, schema_path: str) -> object:
        """
        Upload schema file to Gemini Files API.
        
        Args:
            schema_path: Path to schema file
            
        Returns:
            Uploaded file object
        """
        try:
            schema_name = Path(schema_path).name
            print(f"  Uploading schema file to Gemini API: {schema_name}")
            return self.client.files.upload(file=schema_path)
        except Exception as e:
            raise RuntimeError(f"Failed to upload schema file to Gemini: {e}")
    
    def generate_denormalization(self, prompt_with_schema: str, schema_file) -> str:
        """
        Generate denormalized schema using Gemini API.
        
        Args:
            prompt_with_schema: Prompt text with schema already injected
            schema_file: Uploaded schema file object from Gemini Files API
            
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
                maxOutputTokens=16384,
            )
            
            # Build prompt with file reference
            prompt_parts = [
                prompt_with_schema,
                types.Part.from_uri(
                    file_uri=schema_file.uri,
                    mime_type=schema_file.mime_type,
                ),
            ]
            
            # Stream response
            print("  Calling Gemini API (streaming response)...")
            response_text = ""
            
            stream = self.client.models.generate_content_stream(
                model=self.model,
                contents=prompt_parts,
                config=generate_content_config,
            )
            
            for chunk in stream:
                response_text += chunk.text
                print(chunk.text, end="", flush=True)
            
            print("\n")  # Add newline after streaming completes
            return response_text
        
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
        
        # Create temporary schema file for upload
        schema_file_path = self.config.results_dir / "temp_schema.txt"
        try:
            with open(schema_file_path, 'w', encoding='utf-8') as f:
                f.write(schema_content)
            
            # Upload schema
            schema_file = self.upload_schema_file(str(schema_file_path))
            
            # Generate denormalization
            result = self.generate_denormalization(prompt_with_schema, schema_file)
            
            return result
        
        finally:
            # Clean up temp file
            if schema_file_path.exists():
                schema_file_path.unlink()