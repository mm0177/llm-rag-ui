# llm-rag-ui

## THE PROJECT USES STABLE DIFFUSION WITH RAG TO GENRATE THE MOBILE UI IMAGES SEAMLESSLY
### HOW TO PROCEED WITH THE RUNNING OF THE APPLICATION
   * INSTALL LM STUDIO APPLICATION
   * IN LM STUDIO INSTALL llama-3.2-1b-instruct 
   * LOAD THE MODEL IN LM STUDIO SO THAT IT RUNS LOCALLY IN THE SYSTEM

To install the libraries , run the following command: `pip install requirements.txt`.

Add The Hugging Face Token,  `hf=` in the rag.py file

The prompt_creator.py file when runs generators the csv file ( Not needed to run as it has been included)

## Run The App

The Rag data is prepared in csv format, you're ready to build and run the application.
When the app runs it converts the datasets and also the prompts into embeddings and uses Qdrant to perform the search

To run the app , run the following command: `streamlit run app.py --server.port 8000`.

Note:
 * When the App First Runs it will take quite some time to run the Search to create the Vectors Please wait while it does it Do not cancel the process.



  
