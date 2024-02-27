import json
import os
import boto3
import numpy as np
from pinecone import Pinecone

def query_endpoint_with_json_payload(encoded_json):
    client = boto3.client('runtime.sagemaker')
    response = client.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/json', Body=encoded_json)
    return response


def parse_response_multiple_texts(query_response):
    model_predictions = json.loads(query_response['Body'].read())
    embeddings = model_predictions['embedding']
    return embeddings

# Step 1: Read text files
folder_path = 'apra_knowledgebase'
text_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
texts = []
print('text_files', text_files)
for file_path in text_files:
    with open(file_path, 'r', encoding='utf-8') as file:
        texts.append(file.read())

session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    region_name=os.getenv("AWS_REGION")
)

endpoint_name = 'jumpstart-dft-hf-textembedding-all-minilm-l6-v2-new'

sagemaker_runtime = session.client('runtime.sagemaker')
embeddings = []

payload = {"text_inputs": texts}
query_response = query_endpoint_with_json_payload(json.dumps(payload).encode('utf-8'))
embeddings = parse_response_multiple_texts(query_response)

# Initialize Pinecone environment
api_key = "19ff39f1-6252-456e-a6ca-a257faa2df8b"
pc = Pinecone(api_key=api_key)

# Connect to an existing Pinecone index
index = pc.Index(host="https://langchainpinecone-of7knlv.svc.apw5-4e34-81fa.pinecone.io")

# Update Pinecone index with embeddings
# Make sure you have unique IDs for each text/document  
for i, (text, embedding) in enumerate(zip(texts, embeddings)):
    vector_id = str(i)  # Unique identifier for the vector
    metadata = {"original_text": text}  # Store original text as metadata
    index.upsert(vectors=[(vector_id, embedding, metadata)])
