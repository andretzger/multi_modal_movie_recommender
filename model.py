from transformers import BertModel, BertTokenizer, ViTModel
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch


#create the pretrained feature extractor for the unstructured text data 
class text_feature_extraction(nn.Module):
    #initialize the pretrained model and tokenizer 
    def __init__(self):
        super(text_feature_extraction, self).__init__()
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    #pass the input text through the BERT model to get the encoding representation of the text summaries 
    def forward(self, text_in):
        #print(text_in)
        encoded_text = self.bert_tokenizer(text_in, padding='max_length', truncation=True, return_tensors="pt")
        outputs = self.bert_model(**encoded_text)
        text_token = outputs.last_hidden_state[:, 0, :]  
        return text_token
    
#create the pretrained ViT to embed the poster image information 
class image_feature_extraction(nn.Module):
    #initialize the pretrain ViT model 
    def __init__(self):
        super(image_feature_extraction, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        
    # pass the input images through the ViT to get encodings 
    def forward(self, image_in):
        outputs = self.vit(pixel_values=image_in)
        image_token = outputs.last_hidden_state[:, 0, :]
        return image_token
    
#create the 1-layer network to embed the structured data 
class structured_data_extraction(nn.Module):
    #initilaized the embedding layer
    def __init__(self,input_dim, output_dim=768, intermediate_dim=512): 
        super(structured_data_extraction, self).__init__()
        self.embedding_layer = nn.Sequential(nn.Linear(input_dim, intermediate_dim), nn.ReLU(),
                                            nn.Linear(intermediate_dim, output_dim), nn.ReLU())
    
    #fpass the structured data through the initialized model
    def forward(self, structured_in):
        return self.embedding_layer(structured_in)
    
#create the transformer model for the "fused" features 
class feature_fuser(nn.Module):
    #initialize the transformer archetecture 
    def __init__(self, input_dim=768, num_heads=8, num_layers=2, feedforward_dim = 2048, intermediate_dim = 256): 
        super(feature_fuser, self).__init__()
        self.transformer = nn.Transformer( d_model = input_dim, 
                                          nhead = num_heads,
                                          num_encoder_layers = num_layers,
                                          num_decoder_layers = num_layers,
                                          dim_feedforward=2048 )
        self.classifier = nn.Sequential( 
            nn.Linear(input_dim, intermediate_dim), 
            nn.ReLU(), 
            nn.Linear(intermediate_dim, 5), #there is a 5 star rating system 
            nn.Softmax(dim=-1) ) 

    #pass the fused embeddings through the initialized model 
    def forward(self, fused_features): 
        fused_features.transpose(0,1)
        transformer_output = self.transformer(fused_features, fused_features)
        classifier_token = transformer_output[0,:,:]
        return self.classifier(classifier_token)
    

#put all of the pieces together into the multi modal model
class multi_modal_recommender(nn.Module): 
    #initialize all of the models that are part of the complete model
    def __init__(self, structured_input_dim):
        super(multi_modal_recommender, self).__init__()
        self.text_extractor = text_feature_extraction()
        self.image_extractor = image_feature_extraction()
        self.structured_extractor = structured_data_extraction(input_dim=structured_input_dim)
        self.feature_fusion = feature_fuser()

    #pass all of the data through its embedding models and then fuse them and pass into fused model
    def forward(self, text_in, image_in, structured_in): 
        text_embedding = self.text_extractor(text_in)
        image_embedding = self.image_extractor(image_in) 
        structured_embedding = self.structured_extractor(structured_in)
        fused_embedding = torch.stack([text_embedding, image_embedding, structured_embedding])
        return self.feature_fusion(fused_embedding)
        
    