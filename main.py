import os
import logging
import json
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import PromptTemplate


def load_config(file_name):
    try:
        with open(file_name, 'r') as config_file:
            data = json.load(config_file)
            required_keys = ["context", "prompt_template", "embedding_model", "llm_name"]
            for key in required_keys:
                if key not in data:
                    raise KeyError(f"Missing required config key: {key}")
            return data
    except FileNotFoundError:
        logging.error(f"Configuration file '{file_name}' not found.")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from '{file_name}'.")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during config load: {str(e)}")
        raise


class ChatBot:
    def __init__(self, config_file_name="config.json"):
        load_dotenv()
        self.index_name = os.getenv("INDEX_NAME")
        if not self.index_name:
            logging.error("INDEX_NAME is not set in the environment variables.")
            raise ValueError("INDEX_NAME is required.")

        self.config = load_config(config_file_name)

        self.vectorstore = PineconeVectorStore(
            index_name=self.index_name,
            embedding=OllamaEmbeddings(model=self.config["embedding_model"])
        )

        self.llm = OllamaLLM(
            model=self.config["llm_name"],
            callbacks=[StreamingStdOutCallbackHandler()]
        )

        self.prompt = PromptTemplate(
            template=self.config['context'] + " " + self.config['prompt_template'],
            input_variables=["context", "question"]
        )

        self.rag_chain = (
                {"context": self.vectorstore.as_retriever(), "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
        )


if __name__ == '__main__':
    try:
        bot = ChatBot()
        while True:
            message = input(">>> ")
            if message == "/exit":
                break
            bot.rag_chain.invoke(message)
            print()
    except Exception as e:
        logging.error(f"Critical error during chatbot initialization: {str(e)}")

