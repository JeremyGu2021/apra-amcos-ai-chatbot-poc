import json
import time
from pinecone import Pinecone
from sagemaker_utils import prompt_template, sagemaker_runtime, get_text_embedding, parse_text_embedding_response, parse_pinecone_response, query_openai_api, print_messages, format_messages, get_llm_generation


user_message = 'What is APRA AMCOS?'

print(f"The user message is: {user_message}")

start_time = time.time()

# Step 1 - convert user input into vector data, using sagemaker text_embedding model
payload = {"text_inputs": [user_message]}
query_response = get_text_embedding(json.dumps(payload).encode('utf-8'))
embeddings = parse_text_embedding_response(query_response)

# Step 2 - query Pinecore for matching
# Initialize Pinecone environment
api_key = "19ff39f1-6252-456e-a6ca-a257faa2df8b"
pc = Pinecone(api_key=api_key)

# Connect to an existing Pinecone index
index = pc.Index(host="https://langchainpinecone-of7knlv.svc.apw5-4e34-81fa.pinecone.io")

# Perform the query
# Specify the number of similar vectors to return by top_k
pinecone_query_response = index.query(
    vector=embeddings,
    top_k=2,
    include_values=True,
    include_metadata=True
)     

# Step 3 - query LLM with context
context_str = '\n'.join(parse_pinecone_response(pinecone_query_response)) 
text_input = prompt_template.replace("{context}", context_str).replace("{question}", user_message)

# dialog = [{"role": "user", "content": f"{text_input}"}]
# prompt = format_messages(dialog)
# payload = {"inputs": prompt, "parameters": {"max_new_tokens": 500, "top_p": 0.9, "temperature": 0.6}}
# response = get_llm_generation(payload)

dialog = [{"role": "user", "content": f"{text_input}"}]
openai_content = query_openai_api(dialog)
print('openai_content', openai_content)

end_time = time.time()
elapsed_time = end_time - start_time

# print_messages(prompt, response)
print(elapsed_time)
