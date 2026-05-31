from pathlib import Path
import pandas as pd
import torch

import sentencepiece as spm
from text import TextDataset, 배치병합

폴더 = Path('data/koen')

tokenizer = spm.SentencePieceProcessor()
tokenizer.load('spm.model')

def 전처리(text, 최대길이):
    정수시퀀스 = tokenizer.encode(text, out_type=int)    
    return 정수시퀀스[:최대길이]

datasets = {}
for split, 샘플수 in [('train', 10000), ('validation', 1000), ('test', 1000)]:
    frame = pd.read_csv(폴더 / f'{split}.csv')
    frame.dropna(inplace=True) # 결측값 제거
    # 각 클래스에서 샘플링
    구어체 = frame[frame['source'] == 71265].sample(샘플수, random_state=0)
    특허 = frame[frame['source'] == 563].sample(샘플수, random_state=0)

    samples = pd.concat([구어체, 특허], ignore_index=True)
    
    datasets[split] = TextDataset(
        texts=samples['ko'].tolist(),
        # 텍스트 분류를 위해 클래스 레이블도 함께 제공해야 합니다.
        labels=samples['source'].map({71265: 0, 563: 1}).tolist(),
        transform=lambda text: 전처리(text, 최대길이=300)
    )
    print(f'{split:<10}: {len(datasets[split]):,}')

dataloaders = {}
for split in datasets.keys():
    dataloaders[split] = torch.utils.data.DataLoader(
        datasets[split], shuffle=(split=='train'),
        batch_size=32, num_workers=2, collate_fn=배치병합
    )
    print(f'{split:<10} 배치 수: {len(dataloaders[split])}')