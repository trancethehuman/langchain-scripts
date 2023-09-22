import csv
import os
from typing import List

from bs4 import BeautifulSoup as Soup
from dotenv import load_dotenv
from langchain.document_loaders import (DirectoryLoader, PDFMinerLoader,
                                        TextLoader, UnstructuredFileLoader,
                                        UnstructuredURLLoader)
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

load_dotenv()  # Required to load .env

openai_api_key = os.getenv("OPENAI_API_KEY")


embeddings = OpenAIEmbeddings(max_retries=2)  # type: ignore

input_folder_path = "./input_data/"


def load_urls_recursively(url: str):
    loader = RecursiveUrlLoader(
        url=url, max_depth=7, extractor=lambda x: Soup(x, "html.parser").text)

    docs = loader.load()

    return docs


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


def save_faiss_locally(vectorstore, name: str):
    vectorstore.save_local(
        "./output_data/" + "faiss_" + name)  # type: ignore

    print(f"Vectorstore saved locally: {name}")


def initialize_vectorstore(input, vectorstore_name: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200)

    texts = ""
    if (input is not None):
        texts = text_splitter.split_documents(input)

    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore_file_name = vectorstore_name
    print("Vectorstore created.")

    save_faiss_locally(vectorstore, vectorstore_file_name)

    return


def merge_faiss(faiss_paths: str, new_faiss_name: str):
    faiss_paths_list = [input_folder_path +
                        path for path in faiss_paths.replace(" ", "").split(",")]

    print(faiss_paths_list)

    faiss_list = []

    for path in faiss_paths_list:
        faiss_list.append(FAISS.load_local(path, embeddings))

    print(faiss_list)

    first_faiss_from_list = faiss_list.pop(0)

    for other_faiss in faiss_list:
        first_faiss_from_list.merge_from(other_faiss)
    print("Vectorstores merged.")

    save_faiss_locally(first_faiss_from_list, new_faiss_name)

    return


def command_line():
    first_choice = input(
        "What would you like to do? Load docs (folder, txt, csv, urls, , urls_recursively, pdf) or merge FAISS vectorstores (merge_faiss): ")

    if (first_choice) == "merge_faiss":
        faiss_paths = input("List of FAISS vectorstores' paths: ")
        new_faiss_name = input("Name of the new vectorstore to be created: ")
        merge_faiss(faiss_paths, new_faiss_name)
        return

    knowledge = None
    if (first_choice) == "folder":
        documents_folder = input_folder_path + input(
            "What is the path to the folder holding the documents? ")
        glob_pattern = input(
            "What is the glob pattern for search for documents? Leave blank to feed everything. ")
        knowledge = load_all_documents_not_csv_from_folder(
            documents_folder, glob_pattern)
        initialize_vectorstore(knowledge,
                               input("Vectorstore name: "))
    if (first_choice) == "pdf":
        knowledge = load_document_as_pdf(
            input_folder_path + input("Path to .pdf file: "))
        initialize_vectorstore(knowledge,
                               input("Vectorstore name: "))
    if (first_choice) == "txt":
        knowledge = load_document_as_txt(
            input_folder_path + input("Path to .txt file: "))
        initialize_vectorstore(knowledge,
                               input("Vectorstore name: "))
    if (first_choice) == "csv":
        knowledge = load_document_as_csv(
            input_folder_path + input("Path to .csv file: "))
        initialize_vectorstore(knowledge,
                               input("Vectorstore name: "))
    if (first_choice) == "urls":
        knowledge = load_documents_as_urls(
            input("Paste the URLs here (separated by commas): "))
        initialize_vectorstore(knowledge,
                               input("Vectorstore name: "))

    if (first_choice) == "urls_recursively":
        documents = load_urls_recursively(
            input("Paste the URL here: "))
        initialize_vectorstore(documents,
                               input("Vectorstore name: "))

    # if (first_choice) == "urls_from_csv":
    #     csv_path = input("Path to .csv file: ")
    #     csv_column = input("What's the column that holds the URLs? ")
    #     knowledge = load_documents_as_urls_from_csv_column(csv_path, csv_column)

    return


command_line()
