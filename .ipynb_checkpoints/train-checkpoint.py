from model import multi_modal_recommender
from MovieLensDataset import MovieLensDataset
import torch 
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

#load the data for training 
dev_set_file = './MovieLens-dataset-with-poster-main/dev_set.csv'
posters_dir = './MovieLens-dataset-with-poster-main/poster/'

def train_multi_modal_rec_model(dev_set_file, posters_dir, model=None, loss_func=None, optimizer=None, epochs=5):

    #set the variables for model, loss function, optimizer 
    if model == None: 
        model = multi_modal_recommender(structured_input_dim=5)  #set the number of columns that are in the structured input
    if loss_func == None:
        ce_loss = nn.CrossEntropyLoss()
    if optimizer == None: 
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    #initilaize the transform to preprocess the images 
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])
    
    #Create Dataset and DataLoader
    dataset = MovieLensDataset(dev_set_file, posters_dir,transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    #Train over epochs
    for epoch in range(epochs):
        model.train()
        for text_tokens, images, structured_data, labels in tqdm(dataloader):
            optimizer.zero_grad()
            outputs = model(text_tokens, images, structured_data)    
            loss = ce_loss(outputs, torch.sub(labels,1))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
    