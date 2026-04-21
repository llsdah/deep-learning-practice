from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL.Image as Image
import torch

class CatDogDataset:
    def __init__(self, folder, transform=None):
        self.folder = Path(folder)
        self.files = list(self.folder.glob('**/*.jpg'))
        self.labels = {'cat':0, 'dog':1}
        self.transform = transform

    def __len__(self):
        return len(self.files)
    

    def __getitem__(self, index):
        file = self.files[index]
        target = self.labels[ 'cat' if 'cat' in file.stem else 'dog']

        with Image.open(file) as img:
            sample = img.copy()
        if self.transform:
            sample = self.transform(sample)
        return sample, target



def get_dataloaders(datasets, **config):
    dataloaders = {}
    for split in datasets.keys():
        dataloaders[split] = torch.utils.data.DataLoader(
            datasets[split], 
            **config    
        )
        print(f'{split:<10}: {len(dataloaders[split]):,} batches')
    return dataloaders


def get_train_results(history, plot=True):
    train_results = pd.DataFrame(history.history)
    if plot:
        plt.figure(figsize=(10,4))
        lest = plt.subplot(1, 2, 1) # 1행 2열의 1번째 위치
        train_results.plot(lest)
        right = plt.subplot(1, 2, 2) # 1행 2열의 2번째 위치
        train_results.plot(y=['accuracy', 'val_accuracy'], ax=right)
        plt.tight_layout()
        plt.show()
        
    return train_results

def eveluate_model(model, dataloader):
    results = model.evaluate(dataloader, return_dict=True)
    return pd.Series(results)