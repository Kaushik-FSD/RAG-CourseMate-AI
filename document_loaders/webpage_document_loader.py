from langchain_community.document_loaders import WebBaseLoader

# creating the url object
url = "https://www.apple.com/macbook-air/"
data = WebBaseLoader(url)

# This will crete the document object (metadata and content)
docs = data.load()

print(f"data :: {docs}")