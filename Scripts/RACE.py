import base64
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv


def generate():
    load_dotenv()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    model="gemini-3.1-pro-preview"
    schema_file = "../Benchmarks_schemes/harperdb.txt"
    uploaded_file = client.files.upload(file=schema_file)
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level= "high"
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
        temperature=1.0,
        topP=0.65,
        topK=10,
        #tava 8192
        maxOutputTokens=16384,
    )

    prompt = ['''
        <Role>
        You are a Senior Data Enginner with 15 years of experience specialized in relational and non-relational databases, 
        SQL, and analytical database performance. 
        </Role>

        <Action>
        Your task is to denormalize the physical model of a relational database to
        improve performance, creating a new physical schema for the denormalized version.
        
        Steps to follow:
            1 - Perform a deep analysis of the provided SQL creation script.
            2 - Analyze the 'Frequent Queries / Workload Context'. If none is provided, infer the most likely heavy read queries and JOIN 
                bottlenecks based on the table relationships.
            3 - Evaluate the [COMMON DENORMALIZATION TECHNIQUES] provided below to determine which are best suited for this specific model.
            4 - Desing multiple possible denormalized schemas using/combining the [COMMON DENORMALIZATION TECHNIQUES].
            5 - Evaluate each one of the new schemas and choose the most consistent one.
            6 - Design the new denormalized schema.
        </Action>

        <Context>
        We are optimizing a database for read-heavy analytical workloads. 
        </Context>

        <COMMON DENORMALIZATION TECHNIQUES>
            1 - Adding Redundant Attributes: Adding redundant attributes is useful when certain queries involving multiple joins
                are executed frequently. In this technique, a new table is created containing selected attributes from two or more
                related tables, effectively materializing the result of a join. This allows queries to be executed using a single
                table access, eliminating join overhead. 

            2 - Adding Derived Attributes: In analytical environments, efficient execution of aggregation functions such as MIN,
                MAX, SUM, AVG, and COUNT is crucial. Adding derived attributes involves precomputing commonly used
                aggregated values and storing them directly in a table column. This approach can dramatically reduce query execution
                time by avoiding repetitive aggregation operations. 

            3 - Collapsing Relations: Consists of reducing the number of related tables by merging complex
                relationships into fewer tables. This technique decreases the need for multiple joins in
                queries and can greatly simplify the data schema in analytical scenarios.

            4 - Vertical Partitioning : Consists of dividing a large table into smaller, more manageable talbes.Should be applied
                when a table contains many attributes, but only subsets of those attributes are frequently accessed together.
                For example, in an e-commerce database, some columns may represent physical product characteristics (such as width,
                height, and weight), while others represent pricing and commercial information. In this case, the original table
                can be split into two tables: one containing the product ID and physical attributes, and another containing the
                product ID and pricing-related attributes. This approach reduces unnecessary column access and improves cache
                efficiency for common queries.
        </COMMON DENORMALIZATION TECHNIQUES>

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

        Denormalized Schema:
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
                '''
    ]
    response = client.models.generate_content_stream(
        model=model,
        contents=[prompt, uploaded_file],
        config=generate_content_config,)
    
    for chunk in response:
        print(chunk.text, end="")

if __name__ == "__main__":
    generate()
