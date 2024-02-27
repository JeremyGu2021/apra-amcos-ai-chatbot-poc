import json
import os
from typing import Dict, List
import boto3


prompt_template = """Answer in more than 500 words to the following QUESTION based on the CONTEXT given.
If you do not know the answer and the CONTEXT doesn't contain the answer truthfully say "I don't know".

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

# session = boto3.Session(
#     aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
#     aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
#     aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
#     region_name=os.getenv("AWS_REGION")
# )


sagemaker_runtime = boto3.client('sagemaker-runtime')
# sagemaker_runtime = session.client('runtime.sagemaker')


text_embedding_endpoint_name = 'jumpstart-dft-hf-textembedding-all-minilm-l6-v2-new'
llm_endpoint_name = "jumpstart-dft-meta-textgeneration-llama-2-7b-f"


def get_llm_generation(payload):
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=llm_endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    response = response["Body"].read().decode("utf8")
    response = json.loads(response)
    return response

def get_text_embedding(encoded_json):
    response = sagemaker_runtime.invoke_endpoint(EndpointName=text_embedding_endpoint_name, ContentType='application/json', Body=encoded_json)
    return response

def parse_text_embedding_response(query_response):
    model_predictions = json.loads(query_response['Body'].read())
    embeddings = model_predictions['embedding']
    return embeddings

def parse_pinecone_response(query_response): 
    original_text_list = []
    
    for match in query_response.matches:
        original_text_list.append(match.metadata['original_text'])

    return original_text_list

def format_messages(messages: List[Dict[str, str]]) -> List[str]:
    """Format messages for Llama-2 chat models.
    
    The model only supports 'system', 'user' and 'assistant' roles, starting with 'system', then 'user' and 
    alternating (u/a/u/a/u...). The last message must be from 'user'.
    """
    prompt: List[str] = []

    if messages[0]["role"] == "system":
        content = "".join(["<<SYS>>\n", messages[0]["content"], "\n<</SYS>>\n\n", messages[1]["content"]])
        messages = [{"role": messages[1]["role"], "content": content}] + messages[2:]

    for user, answer in zip(messages[::2], messages[1::2]):
        prompt.extend(["<s>", "[INST] ", (user["content"]).strip(), " [/INST] ", (answer["content"]).strip(), "</s>"])

    prompt.extend(["<s>", "[INST] ", (messages[-1]["content"]).strip(), " [/INST] "])

    return "".join(prompt)

def print_messages(prompt: str, response: str) -> None:
    bold, unbold = '\033[1m', '\033[0m'
    print(f"{bold}> Input{unbold}\n{prompt}\n\n{bold}> Output{unbold}\n{response[0]['generated_text']}\n")