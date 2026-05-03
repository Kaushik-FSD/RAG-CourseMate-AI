# This is a one time run file to create the database for the deeplearning.pdf file
# If you want to run again then delete the chroma_db folder and run the file again

#load pdf 
#split into chunks 
#create the embeddings 
#store into chroma 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma 

data = PyPDFLoader("document_loaders/deeplearning.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)


# Create an embeddings model
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    # If you're behind a restrictive proxy, set this to True after you've downloaded
    # the model once (or if you have it in the local HF cache already).
    model_kwargs={"local_files_only": False},
)

vectorstore = Chroma.from_documents(
    documents= chunks,
    embedding=embeddings_model,
    persist_directory="chroma_db"
)