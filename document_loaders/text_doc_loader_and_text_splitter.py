from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=10,  # the number of word to take (so all chunks will have 10 characters including spaces)
    chunk_overlap=1,  #how many characters to overlap from each chunk
    separator=""  #by default it is set as "/n/n" means it will chunk based on new lines
)

#This will create a object of the text file
# like this -> <langchain_community.document_loaders.text.TextLoader object at 0x1240d17f0>
data = TextLoader("document_loaders/notes_chunk.txt")  

# To create a structured data o/p with metadata and all for this document we have to load it
docs = data.load()

chunks = splitter.split_documents(docs)

# [Document(metadata={'source': 'document_loaders/notes.txt'}, 
# page_content='Hello how are you \n\nI want to see what can I do and also I need your help \nplease help me')]
# print(docs)

print(chunks)