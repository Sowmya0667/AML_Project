import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

uri = "neo4j+ssc://8949059d.databases.neo4j.io"
user = "8949059d"
pwd = os.getenv("NEO4J_PASSWORD")

print(f"Connecting to URI: {uri}")
print(f"Username: {user}")

try:
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    driver.verify_connectivity()
    print("SUCCESS: Connected to Neo4j Aura!")
    driver.close()
except Exception as e:
    print(f"FAILED: {e}")
