import json
import os
import numpy as np
from pinecone import Pinecone
from sagemaker_utils import get_openai_embeddings, get_text_embedding, parse_text_embedding_response



# Step 1: Read text files
folder_path = 'apra_knowledgebase'
text_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
texts = []
file_names = []  # List to keep track of filenames
for file_path in text_files:
    with open(file_path, 'r', encoding='utf-8') as file:
        texts.append(file.read())
        file_names.append(os.path.basename(file_path))  # Save the filename

# Get embeddings from OpenAI
embeddings = get_openai_embeddings(texts)

print('here 01', embeddings)

# Initialize Pinecone environment
api_key = "19ff39f1-6252-456e-a6ca-a257faa2df8b"
pc = Pinecone(api_key=api_key)

# Connect to an existing Pinecone index
index = pc.Index(host="https://langchainpinecone-of7knlv.svc.apw5-4e34-81fa.pinecone.io")

# Update Pinecone index with embeddings
for i, ((text, file_name), embedding) in enumerate(zip(zip(texts, file_names), embeddings)):
    if embedding:  # Ensure embedding is not None
        vector_id = str(i)  # Unique identifier for the vector
        metadata = {"original_text": text, "source": file_name}  # Include original text, filename in metadata
        index.upsert(vectors=[(vector_id, embedding, metadata)])
