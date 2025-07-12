import google.generativeai as genai
import networkx as nx
import pandas as pd
import os
import time
import streamlit as st

API_KEY = st.secrets["GOOGLE_API_KEY"]

GRAPH_FILE = "university_kg.graphml"
OUTPUT_FILE = "embeddings.csv"
NODE_TYPES_TO_EMBED = ['Program', 'Skill', 'Career', 'Course', 'Topic']

genai.configure(api_key=API_KEY)

def create_and_save_embeddings(graph_file, output_file, node_types):
    """
    Generates embeddings for all specified node types in the graph and saves them.
    """
    print(f"Loading knowledge graph from {graph_file}...")

    try:
        g = nx.read_graphml(graph_file)
    except FileNotFoundError:
        print(f"ERROR: Graph file not found at '{graph_file}'. Please make sure it's in the same directory.")
        return

    model_name = 'models/text-embedding-004'
    embedding_rows = []

    nodes_to_embed = [
        (node_name, data.get('type'))
        for node_name, data in g.nodes(data=True)
        if data.get('type') in node_types
    ]

    print(f"Found {len(nodes_to_embed)} total nodes to embed. Generating embeddings...")

    for node_name, node_type in nodes_to_embed:
        try:
            response = genai.embed_content(
                model=model_name,
                content=node_name,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embedding = response['embedding']
            embedding_rows.append({
                'name': node_name,
                'type': node_type,
                'embedding': embedding
            })
            print(f"Embedded: {node_name} ({node_type})")
        except Exception as e:
            print(f"ERROR embedding '{node_name}': {e}")

    df = pd.DataFrame(embedding_rows)
    df.to_csv(output_file, index=False)
    print(f"\nSUCCESS! Embeddings saved to '{output_file}'. Ready for use in your Streamlit app.")

if __name__ == "__main__":
    create_and_save_embeddings(GRAPH_FILE, OUTPUT_FILE, NODE_TYPES_TO_EMBED)
