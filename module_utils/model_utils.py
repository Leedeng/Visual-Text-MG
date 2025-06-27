import cv2
import numpy as np
import pandas as pd
import itertools
import os
import logging
import pandas as pd
from sklearn.utils import shuffle
import csv
from moviepy.editor import *
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from glob import glob
from decord import VideoReader,VideoLoader
from decord import cpu, gpu
from transformers import BertTokenizer
import matplotlib.pyplot as plt
from torchvision.io.video import read_video
from torchvision.models.video import r3d_18,R3D_18_Weights,r2plus1d_18,R2Plus1D_18_Weights 
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm.autonotebook import tqdm
from transformers import BertForSequenceClassification
from tqdm.autonotebook import tqdm
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
import sys
sys.path.append(os.path.relpath('./module_utils/'))
import data_utils, loss_utils, prompt_utils
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./log/R3D_original_PretrainedOn_F32_NormalSampling_resnet2p1d18_SMG_adaptiveprompting_finetuned_github')

class CFG:
    debug = False
    image_path = "../Flicker-8k/Images"
    captions_path = "../Flicker-8k"
    batch_size = 32
    num_workers = 8
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'resnet50'
    image_embedding = 512
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200

    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 0.05

    # image size
    frame = 32
    height = 224
    width = 224

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1

    # 
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    input_dim = tokenizer.vocab_size
    hidden_dim = 128
    nhead = 8
    num_layers = 3

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dataset_dir = './'
dataset_name = 'SMG/'

def sample_frame_indices(clip_len,frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices

    '''
    end_idx = seg_len
    start_idx = 0
    L = np.linspace(start_idx, end_idx-1, num=clip_len+1)
    indices = []
    for i in range(len(L)-1):
        idx = np.random.randint(L[i], L[i+1])
        indices.append(idx)
    indices = np.clip(indices, start_idx, end_idx).astype(np.int64)
    return indices

class CLIPDataset(Dataset):
    def __init__(self, file_dir,video_filenames, captions, label_df,frame, height, width, tokenizer, mode="training"):
    
        self.file_dir = file_dir
        self.video_filenames = list(video_filenames)
        self.captions = captions
        self.num_frames = frame
        self.height = height
        self.width = width
        self.mode = mode
        self.labels = list(label_df['name'].values)
        self.encoded_captions = tokenizer(
            list(self.captions), padding=True, truncation=True, max_length=CFG.max_length
        )
    def __len__(self):
        return len(self.video_filenames)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }
        video_filename = self.video_filenames[idx]
        vr = VideoReader(self.file_dir + video_filename, ctx=cpu(0),width=self.width, height=self.height)
        if len(vr) > self.num_frames + 1:
            indices = sample_frame_indices(clip_len=CFG.frame, frame_sample_rate=1, seg_len=len(vr))
            clip = vr.get_batch(indices).asnumpy()
            #clip = np.array(clip)
        else:
            clip = vr.get_batch([np.sort(np.random.choice(len(vr),self.num_frames, replace=True))]).asnumpy()
            #clip = np.array(clip)
        clip = torch.Tensor(clip).permute(0,3,1,2).float() # frames x channels x height x width
        item['clip'] = clip
        item['caption'] = self.captions[idx]
        label = np.zeros(17)
        label[self.labels.index(self.captions[idx])] = 1.
        item['label'] = label
        return item
    
def get_dataloader(mode):
    if mode == 'training':
        file_dir = dataset_name + 'training_clips/'
        clip_df = pd.read_csv(file_dir + 'clip.csv')
        label_df = pd.read_csv(dataset_name + 'Clip_label.csv')
        training_dataset = CLIPDataset(file_dir,clip_df['clip'].values, clip_df['caption'].values,label_df,CFG.frame,CFG.height,CFG.width,CFG.tokenizer, mode=mode)
        data_loader = torch.utils.data.DataLoader(
                    training_dataset,
                    batch_size=CFG.batch_size,
                    num_workers=CFG.num_workers,
                    shuffle=True,
                    drop_last=True)
    if mode == 'validation':
        file_dir = dataset_name + 'testing_clips/'
        clip_df = pd.read_csv(file_dir + 'clip.csv')
        label_df = pd.read_csv(dataset_name + 'Clip_label.csv')
        validation_dataset = CLIPDataset(file_dir,clip_df['clip'].values, clip_df['caption'].values,label_df,CFG.frame,CFG.height,CFG.width,CFG.tokenizer,mode=mode)
        data_loader = torch.utils.data.DataLoader(
                    validation_dataset,
                    batch_size=CFG.batch_size,
                    num_workers=CFG.num_workers,
                    shuffle=False,
                    drop_last=True)
        
    if mode == 'testing':
        file_dir = dataset_name + 'testing_clips/'
        clip_df = pd.read_csv(file_dir + 'clip.csv')
        label_df = pd.read_csv(dataset_name + 'Clip_label.csv')
        validation_dataset = CLIPDataset(file_dir,clip_df['clip'].values, clip_df['caption'].values,label_df,CFG.frame,CFG.height,CFG.width,CFG.tokenizer, mode=mode)
        data_loader = torch.utils.data.DataLoader(
                    validation_dataset,
                    batch_size=1,
                    num_workers=CFG.num_workers,
                    shuffle=False)
    return data_loader   

class ClipEncoder(torch.nn.Module):
    def __init__(self,output_layer = None,trainable = CFG.trainable):
        super().__init__()
        self.weights = R2Plus1D_18_Weights.DEFAULT
        self.preprocess = self.weights.transforms()
        self.pretrained = r2plus1d_18(pretrained=CFG.pretrained)
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1,len(self.layers)-self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])
        self.net = torch.nn.Sequential(self.pretrained._modules)
        for p in self.net.parameters():
            p.requires_grad = trainable

    def forward(self,x):
        x = self.preprocess(x)
        x = self.net(x)
        return x
    
class TextEncoder(torch.nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]
    
class ProjectionHead(torch.nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = torch.nn.Linear(embedding_dim, projection_dim)
        self.gelu = torch.nn.GELU()
        self.fc = torch.nn.Linear(projection_dim, projection_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x1 = self.gelu(projected)
        x1 = self.fc(x1)
        x1 = self.dropout(x1)
        x1 = x1 + projected
        x1 = self.layer_norm(x1)
        return x1
    
class ProjectionHead2(torch.nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = torch.nn.Linear(embedding_dim, projection_dim)
        self.gelu = torch.nn.GELU()
        self.fc = torch.nn.Linear(projection_dim, projection_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x1 = self.gelu(projected)
        x1 = self.fc(x1)
        x1 = self.dropout(x1)
        x1 = x1 + projected
        x1 = self.layer_norm(x1)
        x1 = self.fc(x1)
        return x1
    
class CLIPModel(torch.nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        self.image_encoder = ClipEncoder(output_layer = 'avgpool')
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.prompt_generator = prompt_utils.VisualPromptAdapter(num_blocks=1, feature_dim=256, initial_alpha=0.1)
        self.temperature = temperature
    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["clip"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features.squeeze(-1).squeeze(-1).squeeze(-1))
        text_embeddings = self.text_projection(text_features)
        return image_embeddings, text_embeddings