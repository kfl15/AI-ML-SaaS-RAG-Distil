# from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings

from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel




def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    # embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

    # linux change for WSL
    embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://172.28.128.1:11434"
    )
    return embeddings
