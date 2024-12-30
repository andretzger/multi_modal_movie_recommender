import os 
import requests
import zipfile

#downloading the dataset from the paper from github
def load_data():
    url = "https://github.com/Xia12121/MovieLens-dataset-with-poster/archive/main.zip"
    dataset_zip = "MovieLens-dataset-with-posters.zip"
    if not os.path.exists(dataset_zip):
        print("Downloading dataset...")
        response = requests.get(url)
        with open(dataset_zip, "wb") as f:
            f.write(response.content)
        
    # Unzip the dataset
    if not os.path.exists("MovieLens-dataset-with-posters-main"):
        print("Extracting dataset...")
        with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
            zip_ref.extractall(".")
