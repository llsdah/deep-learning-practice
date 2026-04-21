from pathlib import Path
import torchvision.transforms as transforms
from catdog_me import  CatDogDataset
import torch


folder = Path('/workspace/cats_dogs_small')
assert folder.is_dir(), f'file is not'


preprocessing = transforms.Compose({
    transforms.Resize((180,180)),
    transforms.ToTensor()
})


dataset = {}

for sub in folder.iterdir():
    split = sub.name
    dataset[split] = CatDogDataset(sub,  transform=preprocessing) # transforms.ToTensor()
    print(f'{split:<10}: {len(dataset[split]):,} samples')

dataloaders = {}

for split in dataset.keys():
    dataloaders[split] = torch.utils.data.DataLoader(dataset[split], batch_size=32, shuffle=(split=='train'),
    num_workers=2, # 병렬 동작
    # OS memory page copy
    pin_memory=True,  
                                                     
    )
    print(f'{split:<10}: {len(dataloaders[split]):,} batches')