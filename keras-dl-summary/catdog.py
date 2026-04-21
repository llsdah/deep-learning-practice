from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL.Image as Image
import torch

class CatDogDataset:
    def __init__(self, 폴더, transform=None):
        self.폴더 = Path(폴더)
        self.파일들 = list(self.폴더.glob('**/*.jpg'))
        self.labels = {'cat': 0, 'dog': 1}
        self.transform = transform
    
    def __len__(self):
        return len(self.파일들)
    
    def __getitem__(self, 색인):
        파일 = self.파일들[색인]
        target = self.labels['cat' if 'cat' in 파일.stem else 'dog']
        with Image.open(파일) as img:
            sample = img.copy()
        if self.transform:
            sample = self.transform(sample)
        return sample, target
    

def get_dataloaders(datasets, **설정):
    print('# 데이터로더')
    dataloaders = {}
    for split in datasets.keys():
        dataloaders[split] = torch.utils.data.DataLoader(
            datasets[split], shuffle=(split=='train'),
            **설정
        )
        print(f'{split:<10}: {len(dataloaders[split]):,} batches')
    return dataloaders

def get_train_results(history, plot=True):
    train_results = pd.DataFrame(history.history)
    if plot:
        plt.figure(figsize=(10, 4))
        왼쪽 = plt.subplot(1, 2, 1) # 1행 2열의 1번째 위치
        train_results.plot(y=['loss', 'val_loss'], ax=왼쪽)
        오른쪽 = plt.subplot(1, 2, 2) # 1행 2열의 2번째 위치
        train_results.plot(y=['accuracy', 'val_accuracy'], ax=오른쪽)
        plt.tight_layout()
        plt.show()
    return train_results

def evaluate_model(model, dataloader):
    results = model.evaluate(dataloader, return_dict=True)
    return pd.Series(results)