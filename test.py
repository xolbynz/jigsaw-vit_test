# from google.colab import drive
# drive.mount('/content/drive')

# # !mkdir DATA
# !unzip -qq {'/content/drive/MyDrive/DACON/open.zip'} -d /content/DATA

# pip install timm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from torch.nn import DataParallel
import timm
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import pandas as pd
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import os
import time



def calc_puzzle(answer_df, submission_df):
    # Check for missing values in submission_df
    if submission_df.isnull().values.any():
        raise ValueError("The submission dataframe contains missing values.")

    # Public or Private answer Sample and Sorting by 'ID'
    submission_df = submission_df[submission_df.iloc[:, 0].isin(answer_df.iloc[:, 0])]
    submission_df = submission_df.sort_values(by='ID').reset_index(drop=True)

    # Check for length in submission_df
    if len(submission_df) != len(answer_df):
        raise ValueError("The submission dataframe wrong length.")

    # Convert position data to numpy arrays for efficient computation
    answer_positions = answer_df.iloc[:, 2:].to_numpy()  # Excluding ID, img_path, and type columns
    submission_positions = submission_df.iloc[:, 1:].to_numpy()  # Excluding ID column

    # Initialize the dictionary to hold accuracies
    accuracies = {}

    # Define combinations for 2x2 and 3x3 puzzles
    combinations_2x2 = [(i, j) for i in range(3) for j in range(3)]
    combinations_3x3 = [(i, j) for i in range(2) for j in range(2)]

    # 1x1 Puzzle Accuracy
    accuracies['1x1'] = np.mean(answer_positions == submission_positions)

    # Calculate accuracies for 2x2, 3x3, and 4x4 puzzles
    for size in range(2, 5):  # Loop through sizes 2, 3, 4
        correct_count = 0  # Initialize counter for correct full sub-puzzles
        total_subpuzzles = 0

        # Iterate through each sample's puzzle
        for i in range(len(answer_df)):
            puzzle_a = answer_positions[i].reshape(4, 4)
            puzzle_s = submission_positions[i].reshape(4, 4)
            combinations = combinations_2x2 if size == 2 else combinations_3x3 if size == 3 else [(0, 0)]

            # Calculate the number of correct sub-puzzles for this size within a 4x4
            for start_row, start_col in combinations:
                rows = slice(start_row, start_row + size)
                cols = slice(start_col, start_col + size)
                if np.array_equal(puzzle_a[rows, cols], puzzle_s[rows, cols]):
                    correct_count += 1
                total_subpuzzles += 1

        accuracies[f'{size}x{size}'] = correct_count / total_subpuzzles

    score = (accuracies['1x1'] + accuracies['2x2'] + accuracies['3x3'] + accuracies['4x4']) / 4.
    return score

class Model(nn.Module):
    def __init__(self, mask_ratio = 0.0, pretrained = True):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.pretrained = pretrained


        deit3 = timm.create_model('deit3_base_patch16_384', pretrained = pretrained)

        self.patch_embed = deit3.patch_embed
        self.cls_token = deit3.cls_token
        self.blocks = deit3.blocks
        self.norm = deit3.norm

        self.jigsaw = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 24*24)
        )

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # target = einops.repeat(self.target, 'L -> N L', N=N)
        # target = target.to(x.device)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep] # N, len_keep
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        target_masked = ids_keep

        return x_masked, target_masked

    def forward(self, x):
        x = self.patch_embed(x)
        x, target = self.random_masking(x, self.mask_ratio)

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        x = self.jigsaw(x[:, 1:])
        return x.reshape(-1, 24*24), target.reshape(-1)

    def forward_test(self, x):
        x = self.patch_embed(x)

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        x = self.jigsaw(x[:, 1:])
        return x

class JigsawDataset(Dataset):
    def __init__(self, df, data_path, mode='train', transform=None):
        self.df = df
        self.data_path = data_path
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.mode == 'train':
            row = self.df.iloc[idx]
            image = read_image(os.path.join(self.data_path, row['img_path']))
            shuffle_order = row[[str(i) for i in range(1, 17)]].values-1
            image = self.reset_image(image, shuffle_order)
            image = Image.fromarray(image)
            if self.transform:
                image = self.transform(image)
            return image
        elif self.mode == 'test':
            row = self.df.iloc[idx]
            image = Image.open(os.path.join(self.data_path, row['img_path']))
            if self.transform:
                image = self.transform(image)
            return image

    def reset_image(self, image, shuffle_order):
        c, h, w = image.shape
        block_h, block_w = h//4, w//4
        image_src = [[0 for _ in range(4)] for _ in range(4)]
        for idx, order in enumerate(shuffle_order):
            h_idx, w_idx = divmod(order,4)
            h_idx_shuffle, w_idx_shuffle = divmod(idx, 4)
            image_src[h_idx][w_idx] = image[:, block_h * h_idx_shuffle : block_h * (h_idx_shuffle+1), block_w * w_idx_shuffle : block_w * (w_idx_shuffle+1)]
        image_src = np.concatenate([np.concatenate(image_row, -1) for image_row in image_src], -2)
        return image_src.transpose(1, 2, 0)

def build_transform(is_train):
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size = (384, 384),
            is_training = True,
            color_jitter = 0.3,
            auto_augment = 'rand-m9-mstd0.5-inc1',
            interpolation= 'bicubic',
            re_prob= 0.25,
            re_mode= 'pixel',
            re_count= 1,
        )
        return transform

    t = []
    t.append(transforms.Resize((384,384), interpolation=3))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df = pd.read_csv('./DATA/train.csv')
    train_df = df.iloc[:-6000]
    valid_df = df.iloc[-6000:]

    train_transform = build_transform(is_train = True)
    valid_transform = build_transform(is_train = False)

    train_dataset = JigsawDataset(df = train_df,
                                data_path = './DATA',
                                mode = 'train',
                                transform = train_transform)
    valid_dataset = JigsawDataset(df = valid_df,
                                data_path = './DATA',
                                mode = 'test',
                                transform = valid_transform)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size = 64,
        shuffle = True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size = 64,
        shuffle = False
    )

    model = Model(mask_ratio = 0.5)
    # model = DataParallel(model)  # 멀티 GPU 사용을 위해 모델을 DataParallel로 래핑합니다.
    model.to(device)
    optimizer = optim.AdamW(model.parameters(),
                            lr=3e-5,
                            weight_decay = 0.05)

    device = 'cuda'
    for epoch in range(1, 11):
        print('Epoch ', epoch)
        st = time.time()
        model.train()
        for i, x in enumerate(train_dataloader):
            x = x.to(device)

            optimizer.zero_grad()

            preds, targets = model(x)

            loss = F.cross_entropy(preds, targets)

            loss.backward()
            optimizer.step()

            if i % 400 == 0:
                print(f'[{i} / {len(train_dataloader)}] loss:', loss.item())
        et = time.time()
        print('Time elapsed: ', et-st)
    # 모델 저장
    torch.save(model.state_dict(), 'your_model.pth')

    # 모델 불러오기
    model.load_state_dict(torch.load('your_model.pth'))
    outs = []
    model.eval()
    with torch.no_grad():
        for x in tqdm(valid_dataloader):
            x = x.to('cuda')
            out = model.forward_test(x)
            out = out.argmax(dim=2).cpu().numpy()
            outs.append(out)

    outs = np.vstack(outs)
    valid_pred_df = valid_df.copy().drop(columns=['img_path'])

    for I, (idx, row) in enumerate(tqdm(valid_pred_df.iterrows(), total=len(valid_df))):
        w = outs[I].reshape(24,24)
        CNT_ROW = np.zeros((4,4,4), dtype=np.int32)
        CNT_COL = np.zeros((4,4,4), dtype=np.int32)
        for i in range(24):
            for j in range(24):
                ROW = i // 6
                COL = j // 6
                v = w[i][j]
                CNT_ROW[ROW][COL][v // 24 // 6] += 1
                CNT_COL[ROW][COL][v % 24 // 6] += 1
        ans = CNT_ROW.argmax(2) * 4 + CNT_COL.argmax(2) + 1
        ans = ans.reshape(16)
        ans = list(map(str, ans))
        valid_pred_df.loc[idx, '1':'16'] = ans
    score = calc_puzzle(valid_df, valid_pred_df)
    print(score)
    
if __name__ == "__main__":
    main()