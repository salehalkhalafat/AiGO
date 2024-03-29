import os
import sys
import openai
from openai import OpenAI
import STT
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, WebBaseLoader, TextLoader, PyPDFLoader, GitLoader, CSVLoader, PythonLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.chains.summarize import load_summarize_chain, _load_stuff_chain
import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY
def retrive():
    # Enable to save to disk & reuse the model (for repeated queries on the same data)
    PERSIST = False
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")
    query = None
    if len(sys.argv) > 1:
        query = sys.argv[1]

    if PERSIST and os.path.exists("persist"):
        print("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        # loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
        loader = DirectoryLoader("Data/")
        if PERSIST:
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
        else:
            index = VectorstoreIndexCreator().from_loaders([loader])

    chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )

    chat_history = []
    while not query:
        text = STT.transcript
        query = text
        result = chain({"question": query, "chat_history": chat_history})
        chat_history.append((query, result['answer']))
        return result['answer']
