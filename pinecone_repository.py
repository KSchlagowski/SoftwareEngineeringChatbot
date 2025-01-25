import os
import logging
import json
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore


def load_config(file_name):
    try:
        with open(file_name, 'r') as config_file:
            data = json.load(config_file)

            required_keys = ["embedding_model", "file_name"]
            for key in required_keys:
                if key not in data:
                    raise KeyError(f"Missing required key in config: {key}")

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


def load_and_split_document(file_name):
    try:
        document = TextLoader(file_name).load()
        text_splitter = CharacterTextSplitter(chunk_size=900, chunk_overlap=100)
        return text_splitter.split_documents(document)
    except FileNotFoundError:
        logging.error(f"Document file '{file_name}' not found.")
        raise
    except Exception as e:
        logging.error(f"Error loading or splitting document: {str(e)}")
        raise


class PineconeRepository:
    def __init__(self, config_file_name="config.json"):
        load_dotenv()
        self.config = load_config(config_file_name)
        self.index_name = os.getenv("INDEX_NAME")

        if not self.index_name:
            logging.error("INDEX_NAME is not set in the environment variables.")
            raise ValueError("INDEX_NAME is required.")

    def insert_data_into_pinecone(self):
        embedding_model = self.config["embedding_model"]
        file_name = self.config["file_name"]

        embeddings = OllamaEmbeddings(model=embedding_model)
        docs = load_and_split_document(file_name)

        PineconeVectorStore.from_documents(docs, embeddings, index_name=self.index_name)
        logging.info("Documents successfully inserted into Pinecone.")


if __name__ == '__main__':
    try:
        repository = PineconeRepository()
        repository.insert_data_into_pinecone()
    except Exception as e:
        logging.error(f"Critical error during data insertion: {str(e)}")
