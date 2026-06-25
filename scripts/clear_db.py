import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from neo4j import GraphDatabase
from ner_graph.config import load_config

def clear_database():
    config = load_config(PROJECT_ROOT)
    uri = config.neo4j_uri
    username = config.neo4j_username
    password = config.neo4j_password
    
    print(f"Kết nối tới Neo4j tại {uri}...")
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    query = "MATCH (n) DETACH DELETE n;"
    
    with driver.session() as session:
        session.run(query)
        print("Đã xóa sạch toàn bộ Data và Node trong Neo4j!")
        
    driver.close()

if __name__ == "__main__":
    clear_database()
