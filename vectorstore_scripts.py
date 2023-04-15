import os
import csv
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader, UnstructuredURLLoader, UnstructuredFileLoader, PDFMinerLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv()  # Required to load .env

openai_api_key = os.getenv("OPENAI_API_KEY")


embeddings = OpenAIEmbeddings(max_retries=2)  # type: ignore


def load_all_documents_not_csv_from_folder(folder_path: str, glob_pattern: str = ""):
    """
    This function loads all documents from a specified folder path that are not in CSV format and
    returns them as a list.
    """
    loader = DirectoryLoader(folder_path, glob=glob_pattern or "**/*")
    loaded_documents = loader.load()

    print(f"Loaded {len(loaded_documents)} documents.")
    return loaded_documents


def load_documents_as_urls_from_csv_column(csv_path: str, column_name: str):
    """
    This function loads documents as URLs from a CSV file's column and returns the loaded documents.
    """
    with open(csv_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)

    print(reader)
    urls = []

    for row in reader:
        row_data = row[column_name]
        urls.append(row_data)

    loader = UnstructuredURLLoader(urls=urls)
    loaded_documents = loader.load()

    print(f"Loaded {len(loaded_documents)} documents.")
    return loaded_documents


def load_document_as_csv(csv_path: str):
    loader = CSVLoader(file_path=csv_path)
    loaded_documents = loader.load()

    print(f"Loaded {len(loaded_documents)} documents.")
    return loaded_documents


def load_document_as_txt(txt_path: str):
    loader = TextLoader(file_path=txt_path)
    loaded_documents = loader.load()

    print(f"Loaded {len(loaded_documents)} text documents.")
    return loaded_documents


def load_document_as_pdf(path: str):
    loader = PDFMinerLoader(file_path=path)
    loaded_documents = loader.load()

    print(f"Loaded {len(loaded_documents)} PDFs.")
    return loaded_documents


def load_documents_as_urls(urls: str):
    urls_list = urls.replace(" ", "").split(",")
    loader = UnstructuredURLLoader(urls=urls_list)
    loaded_documents = loader.load()

    print(f"Loaded {len(loaded_documents)} URL documents.")
    return loaded_documents


def initialize_vectorstore(input, vectorstore_name: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)

    texts = text_splitter.split_documents(input)

    vectorstore = FAISS.from_documents(texts, embeddings)
    print("Vectorstore created.")

    vectorstore_file_name = f"faiss_{vectorstore_name}"

    vectorstore.save_local(vectorstore_file_name)  # type: ignore

    print(f"New vectorstore saved locally: {vectorstore_file_name}")

    return


# TODO: Add tokens and costs logs to embedding new vectorstore
load_docs_type = input(
    "How do you want to load docs? (folder, txt, csv, urls, pdf): ")
knowledge = None
if (load_docs_type) == "folder":
    documents_folder = input(
        "What is the path to the folder holding the documents? ")
    glob_pattern = input(
        "What is the glob pattern for search for documents? Leave blank to feed everything. ")
    knowledge = load_all_documents_not_csv_from_folder(
        documents_folder, glob_pattern)
if (load_docs_type) == "pdf":
    knowledge = load_document_as_pdf(input("Path to .pdf file: "))
if (load_docs_type) == "txt":
    knowledge = load_document_as_txt(input("Path to .txt file: "))
if (load_docs_type) == "csv":
    knowledge = load_document_as_csv(input("Path to .csv file: "))
if (load_docs_type) == "urls":
    knowledge = load_documents_as_urls(
        input("Paste the URLs here (separated by commas): "))
# if (load_docs_type) == "urls_from_csv":
#     csv_path = input("Path to .csv file: ")
#     csv_column = input("What's the column that holds the URLs? ")
#     knowledge = load_documents_as_urls_from_csv_column(csv_path, csv_column)

initialize_vectorstore(knowledge, input("Vectorstore name: "))
