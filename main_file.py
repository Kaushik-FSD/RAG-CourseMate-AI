from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
load_dotenv()

model = ChatMistralAI(
    model="mistral-small-2603"
)

#create an embeddings model
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"local_files_only": False},
)

#locading vector store from the chroma_db folder
vectorstore = Chroma(
    persist_directory= "chroma_db",
    embedding_function=embeddings_model
)

# creating a retriver object from the vector store
retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 4,
        "fetch_k":10,
        "lambda_mult" :0.5
    }
)


#create a prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful AI assistant. Use ONLY the provided context to answer the question.
            If the answer is not present in the context,
            say: "I could not find the answer in the document."
            """
        ),
        (
            "human",
            """
            Context: {context}
            Question: {question}
            """
        )
    ]
)

print("Rag system created ")

print("press 0 to exit ")

while True:
    query = input("You : ")
    if query == "0":
        break 
    
    # This will return the most relevant documents based on the query
    docs = retriever.invoke(query)

    # Now based on the docs chunk we fetched we will create a context
    # The best part is we did not feed the entire document to the model, we just fed the relevant chunks
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    
    final_prompt = prompt.invoke({
        "context" :context,
        "question": query
    })
    
    response = model.invoke(final_prompt)

    print(f"\n AI: {response.content}")

