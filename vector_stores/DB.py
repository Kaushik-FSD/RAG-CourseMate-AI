# This file is for only to demonstrate the vector store and how to use it
# We have not used llm here, we have only used the vector store and the embeddings model
# how the embeddings model works is that it will convert the text into a vector of numbers
# and then it will store the vector in the vector store
# and then we can use the vector store to search for the most similar documents
# and then we can use the most similar documents to answer the question
# and then we can use the most similar documents to answer the question

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Create a list of documents
docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

# Create an embeddings model
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    # If you're behind a restrictive proxy, set this to True after you've downloaded
    # the model once (or if you have it in the local HF cache already).
    model_kwargs={"local_files_only": False},
)

# Create a vectorstore from the documents using Chroma 
vectorstore = Chroma.from_documents(
    documents = docs,
    embedding= embeddings_model,
    persist_directory= "chroma-db"
)

# This will search for the most similar documents to the question
# k=2 means it will return 2 most similar documents
results = vectorstore.similarity_search("what is used for data analysis?",k=2)

print("\n")
for result in results:
    print(result.page_content)
    print(result.metadata)

# This will return the retriver object
# Retriver is a class that will help us to retrieve the documents from the vector store
# but we can also use vectorstore.similarity_search() to get the most similar documents
# but the retriver object will help us to retrieve the documents from the vector store in a more efficient way
# also we can use the retriver object to chain with the llm -> which cant be done with the vectorstore.similarity_search()
retriver = vectorstore.as_retriever()

docs = retriver.invoke("Explain deep learning")

for d in docs:
    print(d.page_content)