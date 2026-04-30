from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

# creating the pdf object
data = PyPDFLoader("document_loaders/GRU.pdf")
docs = data.load()

splitter = TokenTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 10
)

chunks = splitter.split_documents(docs)

print(len(chunks))

# chunks will have metadata and page_content
print(chunks[0].page_content, "\n\n", chunks[0].metadata)

