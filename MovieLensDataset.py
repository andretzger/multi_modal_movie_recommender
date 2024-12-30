import os 
import pandas as pd
from torchvision import transforms
from torch.utils.data import  Dataset
import torch
from PIL import Image

#class to process the unzipped csv of MovieLens Data into usablem, organized, pandas dataframe for the model 
class MovieLensDataset(Dataset):
    #initialize the data that is being inputed and how to handle it
    def __init__(self, MLData_file, posters_dir, tokenizer, transform=None):
        self.data = self.load_MovieLensData(MLData_file)
        self.posters_dir = posters_dir
        self.tokenizer = tokenizer
        self.transform = transform
    
    #determine the number of datapoints that are being used as the length
    def __len__(self):
        return len(self.data)

    #create the database for every movie review, organize it for which embedding model it will go into 
    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        #combine all of the unstructured text data into one large unstructured string 
        text_data = row['title'] + '. ' + row['genres'] + '. ' + row['intro']
        
        #pre-process the image for the model
        poster_path = os.path.join(self.posters_dir, row['poster'])
        image = Image.open(poster_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Structured data 
        structured_data = torch.tensor([row['userId'], row['movieId'], row['age'],row['occupation'],row['zip']], dtype=torch.float) 

        # Label
        label = torch.tensor(row['rating'], dtype=torch.long) #subtract 1 from the rating to make it span 0-4 for indexability

        return text_data, image, structured_data, label

    #a complete poster dataset was not provided. This will load the MovieLens data and will keep only movies that have posters along with reviews.
    #this was considered to be ok as training on a small local device would be made less compute-expensive with less data 
    def load_MovieLensData(self,MLData_file):
        raw_MovieLens_df = pd.read_csv(MLData_file)
        #remove all poster values less than 218 and those that are missing a value within that range 
        missing_posters = [33, 37, 51, 56, 59, 67, 84, 90, 91, 98, 109, 114, 115, 120, 124, 127, 130, 131, 133, 134, 136, 138, 139, 142, 143, 167, 182, 192, 197, 200]
        raw_MovieLens_df_edited = raw_MovieLens_df[raw_MovieLens_df["poster"].str.extract(r'(\d+)').astype(int)[0].apply(lambda poster_n: poster_n <=218 or poster_n ==294)]
        raw_MovieLens_df_edited = raw_MovieLens_df_edited[~raw_MovieLens_df_edited["poster"].str.extract(r'(\d+)').astype(int)[0].isin(missing_posters)]
        return raw_MovieLens_df_edited

