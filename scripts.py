import os
import csv
from typing import List
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


def save_faiss_locally(vectorstore, name: str):
    vectorstore.save_local(
        "./output_data/" + "faiss_" + name)  # type: ignore

    print(f"Vectorstore saved locally: {name}")


def initialize_vectorstore(input, vectorstore_name: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)

    texts = text_splitter.split_documents(input)

    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore_file_name = f"faiss_{vectorstore_name}"
    print("Vectorstore created.")

    save_faiss_locally(vectorstore, vectorstore_file_name)

    return


def merge_faiss(main_faiss_path: str, other_faiss_paths: List[str], new_faiss_name: str):
    other_faiss_paths_list = other_faiss_paths.replace(" ", "").split(",")

    main_faiss = FAISS.load_local(main_faiss_path, embeddings)

    other_faiss_list = []

    for path in other_faiss_paths_list:
        other_faiss_list.append(FAISS.load_local(path, embeddings))

    for other_faiss in other_faiss_list:
        main_faiss.merge_from(other_faiss)
    print("Vectorstores merged.")

    save_faiss_locally(main_faiss, new_faiss_name)

    return


input_folder_path = "./input_data/"


def command_line():
    first_choice = input(
        "What would you like to do? Load docs (folder, txt, csv, urls, pdf) or merge FAISS vectorstores (merge_faiss): ")

    if (first_choice) == "merge_faiss":
        main_faiss = input_folder_path + input(
            "Path to the main FAISS vectorstore to be merged into: ")
        other_faiss_paths = input_folder_path + \
            input("List of other FAISS vectorstores' paths: ")
        new_faiss_name = input("Name of the new vectorstore to be created: ")
        merge_faiss(main_faiss, other_faiss_paths, new_faiss_name)
        return

    knowledge = None
    if (first_choice) == "folder":
        documents_folder = input_folder_path + input(
            "What is the path to the folder holding the documents? ")
        glob_pattern = input(
            "What is the glob pattern for search for documents? Leave blank to feed everything. ")
        knowledge = load_all_documents_not_csv_from_folder(
            documents_folder, glob_pattern)
    if (first_choice) == "pdf":
        knowledge = load_document_as_pdf(
            input_folder_path + input("Path to .pdf file: "))
    if (first_choice) == "txt":
        knowledge = load_document_as_txt(
            input_folder_path + input("Path to .txt file: "))
    if (first_choice) == "csv":
        knowledge = load_document_as_csv(
            input_folder_path + input("Path to .csv file: "))
    if (first_choice) == "urls":
        knowledge = load_documents_as_urls(
            input("Paste the URLs here (separated by commas): "))
    # if (first_choice) == "urls_from_csv":
    #     csv_path = input("Path to .csv file: ")
    #     csv_column = input("What's the column that holds the URLs? ")
    #     knowledge = load_documents_as_urls_from_csv_column(csv_path, csv_column)

    initialize_vectorstore(knowledge,
                           input("Vectorstore name: "))
    return


command_line()
