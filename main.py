from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

model = ChatMistralAI(
    model="mistral-small-2603"
)

#loading data from file
# data_text = TextLoader("document_loaders/notes.txt")  
# docs_text = data_text.load()

# #loading data from pdf
# data_pdf = PyPDFLoader("document_loaders/GRU.pdf")
# docs_pdf = data_pdf.load()

data_deeplearning_pdf = PyPDFLoader("document_loaders/deeplearning.pdf")  #loading another big pdf
docs_deeplearning_pdf = data_deeplearning_pdf.load()

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs_deeplearning_pdf)

#create a prompt template
template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a AI that summarizes the text"),
        ("human", "{data}")
    ]
)

#create a final prompt
# For Text file
# prompt = template.format_messages(data = docs_text[0].page_content)

# For pdf file
# prompt = template.format_messages(data = docs_pdf)
# The above can break because of context window, we have to use chunking 

# For bigger pdf: (this will break, we will use chunking/splitting)
prompt = template.format_messages(data = docs_deeplearning_pdf)

response = model.invoke(prompt)

print(f"Response from Mistal:: {response.content}")
