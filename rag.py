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
hf_token = ""

# Load the pipeline
pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=hf_token)

# Move the model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)


def get_index(): 

    model_name = "all-MiniLM-L6-v2"
    encode_kwargs = {'normalize_embeddings': True} 

    embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs=encode_kwargs
        )
    
    
    file_path = 'prompts_mobile_ui.csv'  
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")


    loader = CSVLoader(file_path=file_path, encoding='utf-8')    
    documents = loader.load()

    index_from_loader = Qdrant.from_documents(
            documents,
            embeddings,
            location=":memory:", 
            collection_name="my_documents",
        )
        
    return index_from_loader 

def semantic_search(index, original_prompt): 
        
    relevant_prompts = index.similarity_search(original_prompt)    

    list_prompts = []
    for i in range(len(relevant_prompts)):
        list_prompts.append(relevant_prompts[i].page_content)
    
    return list_prompts


def get_rag_response(original_prompt, selected_prompt): 

    return call_lm(original_prompt, selected_prompt)





def get_image_response(prompt_content): 
    # Generate an image
    with torch.no_grad():
        output = pipe(prompt_content)

    return output.images[0]
