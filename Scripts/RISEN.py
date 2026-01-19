import base64
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv


def generate():
    load_dotenv()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    model = "gemini-2.5-pro"
    schema_file = "../Benchmarks_schemes/TPC_H.txt"
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
        #tava 8192
        maxOutputTokens=16384,
        system_instruction=[
            types.Part.from_text(text="""
                <Role>
                    You are a Senior Data Enginner specialized in relational and non-relational databases, 
                    SQL, and analytical database performance.
                </Role>

                <Input>
                    Your task is to denormalize the physical model of a relational database to
                    improve performance, creating a new physical schema for the denormalized version.
                </Input>

                <Steps>
                    First, perform a deep analysis of the SQL file containing the creation script of the relational model.
                    Then, consider possible queries that would require many JOIN operations and therefore could lead to performance issues.
                    After that, evaluate [COMMON DENORMALIZATION TECHNIQUES] that could be useful for the analyzed case. Finally, apply these
                    techniques to create a new physical schema representing the denormalized version.
                </Steps>

                <COMMON DENORMALIZATION TECHNIQUES>
                    1 - Pre-joined Tables: Consists of creating tables that are already joined (pre-computed
                        joins) to avoid join operations at runtime. This is useful in queries involving multiple tables
                        that are executed frequently, reducing query complexity and improving performance
                    2 - Split Tables: Divides a large table into multiple smaller ones, usually to separate rarely
                        accessed data from frequently accessed data. This improves read performance and can
                        reduce index size.
                    3 - Combined Tables: The opposite of Split Tables, it merges data from multiple tables into
                        one. It is used when queries frequently require distributed data, eliminating the need for
                        joins and reducing latency.
                    4 - Derivable Data: Stores data that could be derived or calculated from other columns,
                        such as total sales or a customer's age. This reduces query processing time but increases
                        the risk of inconsistency if not properly managed.
                    5 - Collapsing Relations: Reduces the number of related tables by merging complex
                        relationships into fewer entities. This technique decreases the need for multiple joins in
                        queries and can greatly simplify the data schema in analytical scenarios.
                </COMMON DENORMALIZATION TECHNIQUES>

                <Objective>
                    An SQL file containing the creation script of the denormalized model, keeping the identifiers as
                    similar as possible to the relational model.
                </Objective>

                <Example>
                    Normalized Schema: 
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

                    Desnormalized Schema:
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
                </Example>            
                """),
        ],
    )
    message = [
        "Based on the provided relational database schema, create a new denormalized physical schema aimed at improving "
        "performance for analytical queries. Provide the denormalized schema in SQL format."
        "DO NOT include explanations or comments, only provide the SQL code."
        
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
