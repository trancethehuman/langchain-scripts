# LangChain Scripts

## Getting Started

- Create New virtual env
  - `python -m venv langchain-scripts` (Mac)
  - `py -m venv langchain-scripts` (Windows)
- Start virtual env: `source langchain-scripts/bin/activate` (Mac)
- Create a new `.env` file and add these details
- ```
  OPENAI_API_KEY=key_here
  ```
- To enter the new virtual env every time you start your Powershell (Terminal) in this directory, add this to `.env` file

  Windows 11

  ```
  langchain-scripts\Scripts\Activate.ps1
  ```

- Install requirements: `pip install -r requirements.txt`
  - Note: the repo size can get large due to depedency packages

## Usage

- Put documents into `/input_data` folder, at the root of this repo
  - You can put them into a separate folder like `/input_data/my_docs` and use the folder loader for quick loading
- Run `py scripts.py`
  - When asked for file paths, don't include `./input_data` or `./output_data`
- A new FAISS vector database should be outputed into `/output_data`
