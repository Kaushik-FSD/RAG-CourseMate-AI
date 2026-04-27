from langchain_community.document_loaders import TextLoader

#This will create a object of the text file
# like this -> <langchain_community.document_loaders.text.TextLoader object at 0x1240d17f0>
data = TextLoader("document_loaders/notes.txt")  

# To create a structured data o/p with metadata and all for this document we have to load it
docs = data.load()

# [Document(metadata={'source': 'document_loaders/notes.txt'}, 
# page_content='Hello how are you \n\nI want to see what can I do and also I need your help \nplease help me')]
print(docs)