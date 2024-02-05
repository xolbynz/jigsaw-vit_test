import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import lightning as L

from torchvision.transforms import v2 as  transforms
from copy import deepcopy
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from torchvision.io import read_image
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

# modify vit-g (add pos bias, attn bias)
from timm.models.vision_transformer import Block, Attention, VisionTransformer

def attention_forward(self, x, attn_bias=None):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    if attn_bias is not None:
        attn + attn_bias
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

def block_forward(self, x_and_attn_bias):
    x, attn_bias = x_and_attn_bias
    x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_bias)))
    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    return (x, attn_bias)

def vision_transformer_forward_features(self, x, embed_bias=None, attn_bias=None):
    x = self.patch_embed(x)
    x = self._pos_embed(x)
    if embed_bias is not None:
        x = x + embed_bias
    x = self.patch_drop(x)
    x = self.norm_pre(x)
    x, _ = self.blocks((x,attn_bias))
    x = self.norm(x)
    return x

def vision_transformer_forward(self, x, embed_bias=None, attn_bias=None):
    x = self.forward_features(x, embed_bias, attn_bias)
    return x

class JigsawDataset(Dataset):
    def __init__(self, df, data_path, mode='train'):
        self.df = df
        self.data_path = data_path
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.mode == 'train':
            row = self.df.iloc[idx]
            image = read_image(os.path.join(self.data_path, row['img_path']))
            shuffle_order = row[[str(i) for i in range(1, 17)]].values-1
            image_src = self.reset_image(image, shuffle_order)
            image_reshuffle, reshuffle_order = self.shuffle_image(image_src)
            adjacency_matrix = self.get_adjacency_matrix(reshuffle_order)
            data = {
                'image_src':image_src,
                'image_reshuffle':image_reshuffle,
                'order':reshuffle_order,
                'adjacency_matrix':adjacency_matrix,
                'score': self.get_score(range(16), reshuffle_order),
            }
            return data
        elif self.mode == 'val':
            row = self.df.iloc[idx]
            image = read_image(os.path.join(self.data_path, row['img_path'])).numpy()
            shuffle_order = row[[str(i) for i in range(1, 17)]].values-1
            adjacency_matrix = self.get_adjacency_matrix(shuffle_order.tolist())
            data = {
                'image':image,
                'order':shuffle_order,
                'adjacency_matrix':adjacency_matrix,
            }
            return data
        elif self.mode == 'inference':
            row = self.df.iloc[idx]
            image = read_image(os.path.join(self.data_path, row['img_path'])).numpy()
            data = {
                'image':image
            }
            return data

    def reset_image(self, image, shuffle_order):
        c, h, w = image.shape
        block_h, block_w = h//4, w//4
        image_src = [[0 for _ in range(4)] for _ in range(4)]
        for idx, order in enumerate(shuffle_order):
            h_idx, w_idx = divmod(order,4)
            h_idx_shuffle, w_idx_shuffle = divmod(idx, 4)
            image_src[h_idx][w_idx] = image[:, block_h * h_idx_shuffle : block_h * (h_idx_shuffle+1), block_w * w_idx_shuffle : block_w * (w_idx_shuffle+1)]
        image_src = np.concatenate([np.concatenate(image_row, -1) for image_row in image_src], -2)
        return image_src

    def shuffle_image(self, image):
        c, h, w = image.shape
        block_h, block_w = h//4, w//4
        shuffle_order = list(range(0, 16))
        random.shuffle(shuffle_order)
        image_shuffle = [[0 for _ in range(4)] for _ in range(4)]
        for idx, order in enumerate(shuffle_order):
            h_idx, w_idx = divmod(order,4)
            h_idx_shuffle, w_idx_shuffle = divmod(idx, 4)
            image_shuffle[h_idx_shuffle][w_idx_shuffle] = image[:, block_h * h_idx : block_h * (h_idx+1), block_w * w_idx : block_w * (w_idx+1)]
        image_shuffle = np.concatenate([np.concatenate(image_row, -1) for image_row in image_shuffle], -2)
        return image_shuffle, shuffle_order

    def get_adjacency_matrix(self, order): # 패치에 대하여 연결된 패치 찾기
        order_matrix = [order[4*i:4*(i+1)]for i in range(4)]
        adj_matrix = np.zeros((16,16), dtype=int)
        for i in range(4):
            for j in range(4):
                o = order_matrix[i][j]
                i_o, j_o = divmod(o,4)
                for i_add,j_add in [(-1,0), (1,0), (0,1), (0,-1)]:
                    i_compare, j_compare = i_o+i_add, j_o+j_add
                    if i_compare<0 or i_compare>=4 or j_compare<0 or j_compare>=4 : continue
                    o_compare = order[i_compare*4+j_compare]
                    i_, j_ = i*4+j, order.index(i_compare*4+j_compare)
                    if (i_add,j_add) == (-1,0):
                        adj_matrix[i_][j_] = 1 # 상
                        adj_matrix[j_][i_] = 2 # 하
                    elif (i_add,j_add) == (-1,0):
                        adj_matrix[i_][j_] = 2
                        adj_matrix[j_][i_] = 1
                    elif  (i_add,j_add) == (0,-1):
                        adj_matrix[i_][j_] = 3 # 좌
                        adj_matrix[j_][i_] = 4 # 우
                    elif (i_add,j_add) == (0,1):
                        adj_matrix[i_][j_] = 4
                        adj_matrix[j_][i_] = 3
        return adj_matrix

    def get_score(self, order_true, order_pred): # regression task? 현재 아키텍처와 맞지 않을듯
        puzzle_a = np.array(order_true, dtype=int).reshape(4, 4)
        puzzle_s = np.array(order_pred, dtype=int).reshape(4, 4)

        accuracies = {}
        accuracies['1x1'] = np.mean(puzzle_a == puzzle_s)

        combinations_2x2 = [(i, j) for i in range(3) for j in range(3)]
        combinations_3x3 = [(i, j) for i in range(2) for j in range(2)]

        for size in range(2, 5):  # Loop through sizes 2, 3, 4
            correct_count = 0  # Initialize counter for correct full sub-puzzles
            total_subpuzzles = 0
            combinations = combinations_2x2 if size == 2 else combinations_3x3 if size == 3 else [(0, 0)]
            for start_row, start_col in combinations:
                rows = slice(start_row, start_row + size)
                cols = slice(start_col, start_col + size)
                if np.array_equal(puzzle_a[rows, cols], puzzle_s[rows, cols]):
                    correct_count += 1
                total_subpuzzles += 1

            accuracies[f'{size}x{size}'] = correct_count / total_subpuzzles

        score = (accuracies['1x1'] + accuracies['2x2'] + accuracies['3x3'] + accuracies['4x4']) / 4.
        return score

class JigsawCollateFn:
    def __init__(self, transform, mode):
        self.mode = mode
        self.transform = transform

    def __call__(self, batch):
        if self.mode=='train':
            pixel_values = torch.stack([self.transform(Image.fromarray(data['image_reshuffle'].astype(np.uint8).transpose(1,2,0))) for data in batch])
            order = torch.LongTensor([data['order'] for data in batch])
            adjacency_matrx = torch.LongTensor([data['adjacency_matrix'] for data in batch])
            return {
                'pixel_values':pixel_values,
                'order':order,
                'adjacency_matrx':adjacency_matrx
            }
        elif self.mode=='val':
            pixel_values = torch.stack([self.transform(Image.fromarray(data['image'].astype(np.uint8).transpose(1,2,0))) for data in batch])
            order = torch.LongTensor([data['order'] for data in batch])
            adjacency_matrx = torch.LongTensor([data['adjacency_matrix'] for data in batch])
            return {
                'pixel_values':pixel_values,
                'order':order,
                'adjacency_matrx':adjacency_matrx
            }
        elif self.mode=='inference':
            pixel_values = torch.stack([self.transform(Image.fromarray(data['image'].astype(np.uint8).transpose(1,2,0))) for data in batch])
            return {
                'pixel_values':pixel_values,
            }
            
class JigsawElectra(nn.Module):
    """
    1st Stage:
    In the initial stage, a transformer architecture is employed to discern optimal patch arrangements for each puzzle segment.
    This involves intricate spatial relationships, where the model dynamically identifies neighboring patches in cardinal directions(i.e., up, down, left, right).
    The foundation of this stage lies in the incorporation of attention matrices at the final layer, providing nuanced insights into patch interdependencies.
    
    2nd Stage:
    Subsequently, the second stage capitalizes on the predicted matrices from the initial stage to derive piece-type embeddings and connect-type embedding.
    These embeddings encapsulate diverse spatial configurations, such as cross shapes, left corners and right, and else.
    The innovation lies in the integration of piece-type embeddings as positional embedding biases, enhancing the model's contextual awareness.
    Furthermore, connect matrix embeddings serve as attention biases, enabling the model to capture intricate inter-piece relationships.
    The final objective of this stage is to predict an optimal reordering sequence, leveraging the acquired embeddings.
    
    The backbone model shares weights excluding head layers. And losses are jointly computed for gradient updates, aiming for efficient learning and high performance.
    """
    def __init__(self, model, config):
        super(JigsawElectra, self).__init__()
        for k,v in config.items():
            setattr(self,k,v)
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.num_patch_per_block = int(self.image_size/4/self.patch_size)
        self.model = model
        
        self.pos_emb = nn.Parameter(torch.randn(16, self.hidden_size))
        self.piece_type_emb = nn.Embedding(10, self.hidden_size, padding_idx=0)
        self.piece_type_emb.weight.data[0,:]=0
        self.piece_type_emb.weight.data = self.piece_type_emb.weight.data*0.1
        self.connect_type_emb = nn.Embedding(5, self.num_attention_heads, padding_idx=0)
        self.connect_type_emb.weight.data[0,:]=0
        self.connect_type_emb.weight.data = self.connect_type_emb.weight.data*0.1
        
        self.local_linear1 = nn.LazyLinear(self.hidden_size)
        self.local_linear2 = nn.LazyLinear(self.hidden_size)
        self.local_conv = nn.Conv2d(self.num_attention_heads, self.num_attention_heads, int(self.image_size/16), int(self.image_size/16))
        self.local_clf = nn.Sequential(
            nn.LazyLinear(self.num_attention_heads),
            nn.Tanh(),
            nn.LazyLinear(5),
        )

        self.global_conv = nn.Conv1d(self.hidden_size, self.hidden_size, int(self.image_size/16), int(self.image_size/16))
        self.global_clf = nn.Sequential(
            nn.LazyLinear(self.hidden_size),
            nn.Tanh(),
            nn.LazyLinear(16),
        )

    def _transpose(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        x = x.permute(0, 2, 1, 3)
        b, h, l, d = x.shape
        x = torch.cat(x.reshape(b, h, -1, self.num_patch_per_block, d).split(self.num_patch_per_block, 2), 3).reshape(b, h, l, d)
        return x
        
    def local_forward(self, x, label=None):
        pos_emb = self.pos_emb.reshape(4,4,-1)
        pos_emb = pos_emb.unsqueeze(-2).repeat(1,1,self.num_patch_per_block,1).reshape(4,-1,self.hidden_size)
        pos_emb = pos_emb.unsqueeze(1).repeat(1,self.num_patch_per_block, 1, 1).reshape(-1, 4*self.num_patch_per_block, self.hidden_size)
        pos_emb = pos_emb.reshape(-1, self.hidden_size)
        
        x = self.model(x, embed_bias=pos_emb)
        x1 = self._transpose(self.local_linear1(x))
        x2 = self._transpose(self.local_linear2(x))
        x = torch.matmul(x1,x2.transpose(-1, -2)).transpose(-1,-2)
        x = self.local_conv(x)
        x = x.permute(0,2,3,1)
        x = self.local_clf(x)
        probs = nn.Softmax(dim=-1)(x)
        loss = None
        if label is not None:
            loss = nn.CrossEntropyLoss()(x.reshape(-1, 5), label.reshape(-1))
        return x, probs, loss
        
    def global_forward(self, x, piece_type=None, connect_type=None, label=None):
        pos_emb = self.pos_emb.reshape(4,4,-1)
        pos_emb = pos_emb.unsqueeze(-2).repeat(1,1,self.num_patch_per_block,1).reshape(4,-1,self.hidden_size)
        pos_emb = pos_emb.unsqueeze(1).repeat(1,self.num_patch_per_block, 1, 1).reshape(-1, 4*self.num_patch_per_block, self.hidden_size)
        pos_emb = pos_emb.reshape(-1, self.hidden_size)
        
        if piece_type is not None:
            b = piece_type.shape[0]
            piece_emb = self.piece_type_emb(piece_type).reshape(b, 4, 4, -1)
            piece_emb = piece_emb.unsqueeze(-2).repeat(1,1,1,self.num_patch_per_block,1).reshape(b, 4,-1,self.hidden_size)
            piece_emb = piece_emb.unsqueeze(2).repeat(1,1,self.num_patch_per_block, 1, 1).reshape(b,-1, 4*self.num_patch_per_block, self.hidden_size)
            piece_emb = piece_emb.reshape(b,-1, self.hidden_size)
            pos_emb = piece_emb+pos_emb
            
        attn_bias = None
        if connect_type is not None:
            b = connect_type.shape[0]
            attn_bias = self.connect_type_emb(connect_type) # B 16,16,8
            attn_bias = attn_bias.unsqueeze(-2).repeat(1,1,1,int(self.image_size/16),1).reshape(b,16,-1,self.num_attention_heads)
            attn_bias = attn_bias.unsqueeze(2).repeat(1,1,int(self.image_size/16), 1, 1).reshape(b,-1, self.image_size, self.num_attention_heads)
            attn_bias = attn_bias.permute(0,3,1,2)
            
        x = self.model(
            x,
            embed_bias=pos_emb,
            attn_bias=attn_bias,
        )
        x = self._transpose(x)
        b, h, l, d = x.shape
        x = x.permute(0,1,3,2).reshape(b,h*d,l)
        x = self.global_conv(x)
        x = x.permute(0,2,1)
        x = self.global_clf(x)
        probs = nn.Softmax(dim=-1)(x)
        
        loss = None
        if label is not None:
            loss = nn.CrossEntropyLoss()(x.reshape(-1, 16), label.reshape(-1))
        return x, probs, loss 

class LitJigsawElectra(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.config = config
        self.jigsaw_electra = JigsawElectra(model, config)
        self.inference_iter = 1
        self.validation_step_outputs = []

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-5)
        return opt
        
    def training_step(self, batch):
        x_local, x_local_probs, loss_local = self.jigsaw_electra.local_forward(batch['pixel_values'], batch['adjacency_matrx'])        
        connect_type = x_local_probs.argmax(-1).detach()
        piece_type = self.connect_to_piece(connect_type).detach()
        x_global, x_global_probs, loss_global = self.jigsaw_electra.global_forward(batch['pixel_values'], piece_type=piece_type, connect_type=connect_type, label=batch['order'])
        loss = loss_local*0.2 + loss_global
        self.log("train_loss_local", loss_local, on_step=True, on_epoch=False)
        self.log("train_loss_global", loss_global, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch):
        x_local, x_local_probs, loss_local = self.jigsaw_electra.local_forward(batch['pixel_values'], batch['adjacency_matrx'])
        self.log("val_loss_local", loss_local)
        connect_type = x_local_probs.argmax(-1).detach()
        piece_type = self.connect_to_piece(connect_type).detach()
        local_accuracy = torch.mean(1*(connect_type == batch['adjacency_matrx']), dtype=torch.float32)
        self.log("val_acc_local", local_accuracy)
        x_global, x_global_probs, loss_global = self.jigsaw_electra.global_forward(batch['pixel_values'], piece_type=piece_type, connect_type=connect_type, label=batch['order'])
        self.log("val_loss_global", loss_global)
        self.validation_step_outputs.append((x_global_probs, batch['order']))
        return
    
    def predict_step(self, batch):
        pixel_values = batch['pixel_values']
        label = batch.get('order', None)
        for i in range(self.inference_iter):
            x_local, x_local_probs, _ = self.jigsaw_electra.local_forward(pixel_values)        
            connect_type = x_local_probs.argmax(-1).detach()
            piece_type = self.connect_to_piece(connect_type).detach()
            x_global, x_global_probs, _ = self.jigsaw_electra.global_forward(batch['pixel_values'], piece_type=piece_type, connect_type=connect_type)
            reorder = self._probs_to_order(x_global_probs)
            pixel_values = self._reorder_image(pixel_values, reorder)
        return x_global_probs, reorder, label
    
    def connect_to_piece(self, connect_types):
        device = connect_types.device
        connect_types = connect_types.detach().cpu()
        piece_types = []
        for connect_type in connect_types:
            piece_type = []
            for connect_type_row in connect_type:
                connect_bins = torch.bincount(connect_type_row)
                if torch.equal(connect_bins[1:5], torch.LongTensor([0,1,0,1])): #  ┌
                    piece_type.append(1)
                elif torch.equal(connect_bins[1:5], torch.LongTensor([0,1,1,1])): # ㅜ
                    piece_type.append(2)
                elif torch.equal(connect_bins[1:5], torch.LongTensor([0,1,1,0])): # ㄱ
                    piece_type.append(3)
                elif torch.equal(connect_bins[1:5], torch.LongTensor([1,1,0,1])): # ㅏ
                    piece_type.append(4)
                elif torch.equal(connect_bins[1:5], torch.LongTensor([1,1,1,0])): # ㅓ
                    piece_type.append(5)
                elif torch.equal(connect_bins[1:5], torch.LongTensor([1,0,0,1])): # ㄴ
                    piece_type.append(6)
                elif torch.equal(connect_bins[1:5], torch.LongTensor([1,0,1,1])): # ㅗ
                    piece_type.append(7)
                elif torch.equal(connect_bins[1:5], torch.LongTensor([1,0,1,0])): # ┘
                    piece_type.append(8)
                elif torch.equal(connect_bins[1:5], torch.LongTensor([1,1,1,1])): # +
                    piece_type.append(9)
                else: # unknown
                    piece_type.append(0)
            piece_types.append(piece_type)
        piece_types = torch.LongTensor(piece_types).to(device)
        return piece_types
        
    def on_validation_epoch_end(self):
        order_pred = []
        order_true = []
        for probs, order in self.validation_step_outputs:
            order_pred.append(self._probs_to_order(probs))
            order_true.append(order)
        order_pred = torch.cat(order_pred).detach().cpu().numpy()
        order_true = torch.cat(order_true).detach().cpu().numpy()
        
        score, accuracies = self._get_score(order_true, order_pred)

        self.log("val_score_1x1", accuracies['1x1'])
        self.log("val_score", score)
        self.validation_step_outputs.clear()
        return
    
    def _get_score(self, order_true, order_pred):
        combinations_2x2 = [(i, j) for i in range(3) for j in range(3)]
        combinations_3x3 = [(i, j) for i in range(2) for j in range(2)]
        accuracies = {}
        accuracies['1x1'] = np.mean(order_true == order_pred)
        
        for size in range(2, 5): 
            correct_count = 0  
            total_subpuzzles = 0
            for i in range(len(order_true)):
                puzzle_a = order_true[i].reshape(4, 4)
                puzzle_s = order_pred[i].reshape(4, 4)
                combinations = combinations_2x2 if size == 2 else combinations_3x3 if size == 3 else [(0, 0)]
                for start_row, start_col in combinations:
                    rows = slice(start_row, start_row + size)
                    cols = slice(start_col, start_col + size)
                    if np.array_equal(puzzle_a[rows, cols], puzzle_s[rows, cols]):
                        correct_count += 1
                    total_subpuzzles += 1
            accuracies[f'{size}x{size}'] = correct_count / total_subpuzzles
        score = (accuracies['1x1'] + accuracies['2x2'] + accuracies['3x3'] + accuracies['4x4']) / 4.
        return score, accuracies
        
    def _probs_to_order(self, probs): # Greedily arrange the jigsaw puzzle pieces based on maximum probability.
        order = []
        for prob in probs:
            prob = prob.reshape(16,16).clone()
            indices = [-1 for _ in range(16)]
            for _ in range(16):
                i, j = divmod(int(prob.argmax()),16)
                indices[i]=j
                prob[i, :] = float('-inf')
                prob[:, j] = float('-inf')
            order.append(indices)
        order = torch.LongTensor(order)
        return order
    
    def _reorder_image(self, images, reorders):
        device = images.device
        images_reordered = []
        for image, reorder in zip(images, reorders):
            image = image.cpu().clone().numpy()
            reorder = reorder.cpu().clone().numpy()
            c, h, w = image.shape
            block_h, block_w = h//4, w//4
            image_src = [[0 for _ in range(4)] for _ in range(4)]
            for idx, order in enumerate(reorder):
                h_idx, w_idx = divmod(order,4)
                h_idx_shuffle, w_idx_shuffle = divmod(idx, 4)
                image_src[h_idx][w_idx] = image[:, block_h * h_idx_shuffle : block_h * (h_idx_shuffle+1), block_w * w_idx_shuffle : block_w * (w_idx_shuffle+1)]
            image_reordered = np.concatenate([np.concatenate(image_row, -1) for image_row in image_src], -2)
            image_reordered = torch.from_numpy(image_reordered)
            images_reordered.append(image_reordered)
        images_reordered = torch.stack(images_reordered).to(device)
        return images_reordered
    

Attention.forward = attention_forward

Block.forward = block_forward

VisionTransformer.forward_features = vision_transformer_forward_features

VisionTransformer.forward = vision_transformer_forward

model = timm.create_model('vit_medium_patch16_gap_256', pretrained=True, num_classes=0)

model_config = {
    'image_size':256,
    'patch_size':16,
    'hidden_size':512,
    'num_attention_heads':8,
}

transform_config = timm.data.resolve_data_config(model.pretrained_cfg)
transform_config.pop('crop_pct')
transform_config.pop('crop_mode')


transform = transforms.Compose([
    transforms.Resize(size=(256,256), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
])
# transform = timm.data.create_transform(
#     **transform_config
# )

config = {}
config['seed']=42
config['batch_size']=24

L.seed_everything(config['seed'])

train_df = pd.read_csv('./DATA/train.csv')
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=config['seed'])
test_df = pd.read_csv('./DATA/test.csv')


train_dataset = JigsawDataset(
    df=train_df,
    data_path='./DATA',
    mode='train'
)
val_dataset = JigsawDataset(
    df=val_df,
    data_path='./DATA',
    mode='val'
)

train_dataloader = DataLoader(train_dataset, collate_fn=JigsawCollateFn(transform, 'train'), batch_size=config['batch_size'])
val_dataloader = DataLoader(val_dataset, collate_fn=JigsawCollateFn(transform, 'val'), batch_size=config['batch_size'])

checkpoint_callback = ModelCheckpoint(
    monitor='val_score',
    mode='max',
    dirpath='./checkpoints/',
    filename='jigsawelectra-vitgap-{epoch:02d}-{val_score:.4f}',
    save_top_k=3,
    save_weights_only=True
)
earlystopping_callback = EarlyStopping(monitor="val_score", mode="max", patience=10)

lit_jigsaw_electra = LitJigsawElectra(model, model_config)

trainer = L.Trainer(
    max_epochs=100, 
    precision='bf16-mixed', 
    callbacks=[checkpoint_callback, earlystopping_callback]
)

trainer.fit(lit_jigsaw_electra, train_dataloader, val_dataloader) 

val_dataset = JigsawDataset(
    df=val_df,
    data_path='./DATA',
    mode='val'
)

pred_dataset = JigsawDataset(
    df=test_df,
    data_path='./DATA',
    mode='inference'
)

val_dataloader = DataLoader(val_dataset, collate_fn=JigsawCollateFn(transform, 'val'), batch_size=config['batch_size'])
pred_dataloader = DataLoader(pred_dataset, collate_fn=JigsawCollateFn(transform, 'inference'), batch_size=config['batch_size'])

lit_jigsaw_electra = LitJigsawElectra.load_from_checkpoint('./checkpoints/jigsawelectra-vitgap-epoch=48-val_score=0.8538.ckpt',model=model, config=model_config)
lit_jigsaw_electra.inference_iter=1

trainer = L.Trainer()

val_preds = trainer.predict(lit_jigsaw_electra, val_dataloader)

val_order_pred = torch.cat([order_pred for pixel_values, order_pred, order_true in val_preds]).cpu().numpy()
val_order_true = torch.cat([order_true for pixel_values, order_pred, order_true in val_preds]).cpu().numpy()

lit_jigsaw_electra._get_score(val_order_true, val_order_pred) # inference_iter=1 늘린다고 좋아지지 않음. pretrained image clf 를 이용하여 선별적으로 iterative하게 하면


preds = trainer.predict(lit_jigsaw_electra, pred_dataloader)



order_pred = torch.cat([order_pred for pixel_values, order_pred, _ in preds]).cpu().numpy()

submission = pd.read_csv('./DATA/sample_submission.csv')

submission.iloc[:,1:] = order_pred+1

submission.to_csv('./submissions/test_submission.csv', index=False)



