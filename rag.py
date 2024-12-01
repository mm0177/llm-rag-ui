import os
import json, base64
from io import BytesIO

from PIL import Image
from diffusers import DiffusionPipeline
import torch

from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Qdrant
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter


from lm_studio import call_lm


# Use your Hugging Face token
hf_token = "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"

# Load the pipeline
pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=hf_token)

# Move the model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)


def get_index(): #creates and returns an in-memory vector store to be used in the application

    model_name = "all-MiniLM-L6-v2"
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

    embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs=encode_kwargs
        )
    
    # Debugging: Check if the file exists
    file_path = 'prompts_mobile_ui.csv'  # Or the correct relative path to your file
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")


    loader = CSVLoader(file_path=file_path, encoding='utf-8')    
    documents = loader.load()

    index_from_loader = Qdrant.from_documents(
            documents,
            embeddings,
            location=":memory:",  # Local mode with in-memory storage only
            collection_name="my_documents",
        )
        
    return index_from_loader #return the index to be cached by the client app

def semantic_search(index, original_prompt): #rag client function
        
    relevant_prompts = index.similarity_search(original_prompt)    

    list_prompts = []
    for i in range(len(relevant_prompts)):
        list_prompts.append(relevant_prompts[i].page_content)
    
    return list_prompts


def get_rag_response(original_prompt, selected_prompt): #rag client function

    return call_lm(original_prompt, selected_prompt)


###############################################################################
######################### Part 2: SD Model Stetup #############################


def get_image_response(prompt_content): #text-to-text client function
    # Generate an image
    with torch.no_grad():
        output = pipe(prompt_content)

    return output.images[0]