from langchain_community.document_loaders import PyPDFLoader

# creating the pdf object
data = PyPDFLoader("document_loaders/GRU.pdf")

# This will crete the document object (metadata and content)
docs = data.load()

# Note: will create multiple document item in array
print(f"data :: {len(docs)}")
print(docs[14])