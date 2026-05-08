from ollama import Client
import os
from dotenv import load_dotenv


load_dotenv()
client = Client(
    host="https://ollama.com",
    headers={"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}
)
    
response = client.generate(
    model='qwen3-coder:480b-cloud',
    prompt= '''
        <Role>
            You are a Senior Data Engineer with 15 years of experience specialized in relational and non-relational databases, SQL,
            and analytical database performance tuning.
        </Role>

        <Input>
            Your task is to denormalize the physical model of a relational database to improve read performance and create a new physical 
            schema for the denormalized version. 
        </Input>

        <Steps>
            1 - Perform a deep analysis of the [PROVIDED SQL CREATION SCRIPT].
            2 - Analyze the 'Frequent Queries / Workload Context'. If none is provided, infer the most likely heavy read queries and JOIN 
                bottlenecks based on the table relationships.
            3 - Evaluate the [COMMON DENORMALIZATION TECHNIQUES] provided below to determine which are best suited for this specific model.
            4 - Desing multiple possible denormalized schemas using/combining the [COMMON DENORMALIZATION TECHNIQUES].
            5 - Evaluate each one of the new schemas and choose the most consistent one.
            6 - Design the new denormalized schema.
        </Steps>

        <PROVIDED SQL CREATION SCRIPT>
            CREATE TABLE entitlements (
                id INT PRIMARY KEY,
                activeDate DATE,
                inactiveDate DATE,
                useLimit INT,
                useCount INT,
                featureType VARCHAR(4),
                packageType VARCHAR(4),
                activeFlag INT
            );

        CREATE TABLE users (
            id INT PRIMARY KEY
        );

        CREATE TABLE user_entitlements (
            user_id INT,
            entitlement_id INT,
            PRIMARY KEY (user_id, entitlement_id),
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (entitlement_id) REFERENCES entitlements(id)
        );
        </PROVIDED SQL CREATION SCRIPT>

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

        <Objective>
            An SQL file containing the complete DDL creation script of the denormalized model.
        </Expectation_and_Objective>

        <Narrowing_and_Constraints>
            - Ensure the output SQL dialect matches the input SQL dialect (e.g., PostgreSQL, MySQL).
            - Preserve appropriate Semantics and Data Types from the original schema.
            - Do not remove tables that are still necessary; only alter/combine them where it strictly benefits analytical read performance.
            - Only output the SQL code.Do not include unnecessary conversational filler.
        </Narrowing_and_Constraints>

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

        </Example>'''
)
print(response.response)