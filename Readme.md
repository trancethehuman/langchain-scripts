# LangChain Scripts

## What this is

- A collection of scripts to quickly load different types of docs into LangChain and create vector databases (FAISS)

## Getting Started

- Create New virtual env
  - `python -m venv virtual-env` (Mac)
  - `py -m venv virtual-env` (Windows)
- Start virtual env: `source virtual-env/bin/activate` (Mac)
- Create a new `.env` file and add these details
  ```
  OPENAI_API_KEY=key_here
  ```
- To enter the new virtual env every time you start your Powershell (Terminal) in this directory, add this to `.env` file

  Windows 11

  ```
  virtual-env\Scripts\Activate.ps1
  ```

- Install requirements: `pip install -r requirements.txt`
  - Note: the repo size can get large due to depedency packages

## Usage

- Create an `input_data` folder.
- Put documents into `/input_data` folder, at the root of this repo.
  - You can put them into a separate folder like `/input_data/my_docs` and later choose the folder loader in `main.py` for quick loading.
- Run `py main.py`
  - When asked for file paths, don't include `./input_data` or `./output_data`
- A new FAISS vector database should be outputed into `/output_data`
