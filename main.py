from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

#loading data from file
data_text = TextLoader("document_loaders/notes.txt")  
docs_text = data_text.load()

data_pdf = PyPDFLoader("document_loaders/GRU.pdf")
docs_pdf = data_pdf.load()

#create a prompt template
template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a AI that summarizes the text"),
        ("human", "{data}")
    ]
)


model = ChatMistralAI(
    model="mistral-small-2603"
)

#create a final prompt
# For Text file
# prompt = template.format_messages(data = docs_text[0].page_content)

# For pdf file
prompt = template.format_messages(data = docs_pdf)
# The above can break because of context window

response = model.invoke(prompt)

print(f"Response from Mistal:: {response.content}")
