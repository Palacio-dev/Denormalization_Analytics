import base64
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv


def generate():
    load_dotenv()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    model = "gemini-2.5-pro"
    schema_file = "../Esquemas_Benchmarks/harperdb.txt"
    uploaded_file = client.files.upload(file=schema_file)
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=-1 #dá pra mudar esse parâmetro
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
        #PARÂMETROS QUE POSSO VARIAR PARA TESTAR POSSIBILIDADES
        temperature=0.3,
        topP=0.65,
        topK=10,
        maxOutputTokens=16384,
        system_instruction=[
            types.Part.from_text(text="""You are a senior Database Manager with 15 years of experience specialized in Relational and No-relational databases,
SQL and analyses of performance of systems.Your job is to , given a ddl file of a relational database, denormalize it
focusing on performance, creating a new schema of the denormalized version.You are doing this job because the system you
are working with escalated immensivaly, resulting on lots of complexity on its stcuture , like perfomance involving 
multiples joins and maintainability. Example: 
CREATE TABLE customers (
  customer_id SERIAL PRIMARY KEY,
  customer_name VARCHAR(100),
  customer_email VARCHAR(100),
  customer_address TEXT
);

CREATE TABLE orders (
  order_id SERIAL PRIMARY KEY,
  customer_id INT REFERENCES customers(customer_id),
  order_date DATE
);

CREATE TABLE products (
  product_id SERIAL PRIMARY KEY,
  product_name VARCHAR(100),
  price DECIMAL(10,2)
);

CREATE TABLE order_details (
  order_detail_id SERIAL PRIMARY KEY,
  order_id INT REFERENCES orders(order_id),
  product_id INT REFERENCES products(product_id),
  quantity INT
);

From this ddl file, you could create :
CREATE TABLE order_summary (
    customer_id INT,
    customer_name VARCHAR(100),
    customer_email VARCHAR(100),
    customer_address TEXT,
    order_date DATE,
    product_name VARCHAR(100),
    quantity INT,
    price DECIMAL(10,2)
);
.
"""),
        ],
    )
    message = [
            "Com base no esquema de banco de dados relacional fornecido, crie um novo esquema físico desnormalizado" \
            "visando melhorar a performance para consultas analíticas. Forneça o esquema desnormalizado no formato SQL."
    ]
    contents=[message, uploaded_file]
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")

if __name__ == "__main__":
    generate()
