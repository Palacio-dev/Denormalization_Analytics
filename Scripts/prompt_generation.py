import base64
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

def generate():
    load_dotenv()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    model = "gemini-3.1-pro-preview"
    prompt = '''
        Generate prompts for a LLM to perform the following task:
        "Denormalize the physical model of a relational database to improve read performance and create a new physical
        schema for the denormalized version."
        The prompts should be designed for different levels of user experience with databases, from beginner to expert. 
            '''
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        )
    print(response.text)


if __name__ == "__main__":
    generate()