import os
os.environ['KERAS_BACKEND'] = 'torch'

import argparse
import numpy as np
import pandas as pd
import torch
import keras
from keras import layers
import keras.ops as ops
from pathlib import Path
import datasets
import sentencepiece as spm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

COPYRIGHT_NOTICE = """코드베이직(codebasic) 2026. 모든 권리 보유. 
https://codebasic.co/"""

# ============================================================================
# 1. 데이터 로더 및 데이터셋
# ============================================================================

def get_data_loader(datasets, **설정):
    """DataLoader 생성"""
    data_loader = {}
    for split in datasets.keys():
        shuffle = True if split == 'train' else False
        data_loader[split] = DataLoader(datasets[split], shuffle=shuffle, **설정)
    return data_loader

class TextDataset:
    """텍스트 데이터셋 (GPT 스타일: 언어 모델링)"""
    def __init__(self, texts, transforms=None):
        self.texts = texts
        self.transforms = transforms

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, 번호):
        text = self.texts[번호]
        if self.transforms:
            text = self.transforms(text)
        return text

def 배치정규화(batch):
    """배치 데이터 정규화 및 패딩 (GPT 스타일)"""
    # 개별 값들을 각각 torch.tensor 형식으로 변환
    sequences = [torch.as_tensor(seq, dtype=torch.long) for seq in batch]
    
    # 입력: 전체 시퀀스에서 마지막 토큰 제외
    xs = [seq[:-1] for seq in sequences]
    # 타겟: 전체 시퀀스에서 첫 토큰 제외 (다음 토큰 예측)
    ys = [seq[1:] for seq in sequences]
    
    xs = pad_sequence(xs, batch_first=True, padding_value=0)
    ys = pad_sequence(ys, batch_first=True, padding_value=0)
    
    return xs, ys

# ============================================================================
# 2. GPT 컴포넌트
# ============================================================================

@keras.saving.register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    """위치 임베딩 레이어"""
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        length = ops.shape(inputs)[-1]
        positions = ops.arange(0, length, 1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions
    
    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        else:
            return ops.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config

@keras.saving.register_keras_serializable()
class GPTBlock(layers.Layer):
    """GPT 트랜스포머 블록 (Decoder-only)"""
    def __init__(self, embed_dim, num_heads, ff_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout_1 = layers.Dropout(0.1)
        self.dropout_2 = layers.Dropout(0.1)
        self.supports_masking = True

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=False):
        # Causal attention mask 생성
        causal_mask = self.get_causal_attention_mask(inputs)
        
        # Self-Attention with causal mask
        attn_output = self.attention(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=causal_mask,
            training=training
        )
        attn_output = self.dropout_1(attn_output, training=training)
        out1 = self.layernorm_1(inputs + attn_output)
        
        # Feed Forward Network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout_2(ffn_output, training=training)
        return self.layernorm_2(out1 + ffn_output)
    
    def get_causal_attention_mask(self, inputs):
        """인과적 주의 마스크 생성 (미래 토큰 마스킹)"""
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        sequence_length = input_shape[1]
        
        # 하삼각 행렬 생성
        i = ops.arange(sequence_length)[:, None]
        j = ops.arange(sequence_length)
        mask = ops.cast(i >= j, dtype="int32")
        
        # (1, seq_len, seq_len)으로 변형 후 (batch, seq_len, seq_len)으로 확대
        mask = ops.reshape(mask, (1, sequence_length, sequence_length))
        broadcast_shape = (batch_size, sequence_length, sequence_length)
        mask = ops.broadcast_to(mask, broadcast_shape)
        
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
        })
        return config

# ============================================================================
# 3. 모델 구축
# ============================================================================

@keras.saving.register_keras_serializable(name='perplexity')
def perplexity_metric(y_true, y_pred):
    """Perplexity 메트릭 계산"""
    # SparseCategoricalCrossentropy 계산
    cross_entropy = keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=True
    )
    # Perplexity = exp(cross_entropy)
    return ops.exp(ops.mean(cross_entropy))

# 훈련 로그에서 표시될 메트릭 이름 설정
perplexity_metric.__name__ = 'perplexity'

def build_model(vocab_size, embed_dim, num_heads, ff_dim, num_blocks, sequence_length):
    """GPT 스타일 언어 모델 구축"""
    inputs = keras.Input(shape=(None,), dtype='int32', name='input_tokens')
    
    # 위치 임베딩
    x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs)
    x = layers.Dropout(0.1)(x)
    
    # GPT 블록 스택
    for i in range(num_blocks):
        x = GPTBlock(embed_dim, num_heads, ff_dim, name=f'gpt_block_{i}')(x)
    
    # 출력층
    outputs = layers.Dense(vocab_size, name='output')(x)
    
    return keras.Model(inputs=inputs, outputs=outputs, name='gpt_model')

# ============================================================================
# 4. 텍스트 생성 함수
# ============================================================================

def generate_text(prompt, model, tokenizer, max_length=100, temperature=1.0):
    """텍스트 생성"""
    # 프롬프트 토큰화
    tokens = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    tokens = np.array([tokens], dtype=np.int64)
    
    for _ in range(max_length):
        # 모델 예측
        predictions = model.predict(tokens, verbose=0)
        
        # 마지막 토큰의 로짓 가져오기
        logits = predictions[0, -1, :] / temperature
        
        # 확률 분포로 변환 및 샘플링 (softmax)
        logits_exp = np.exp(logits - np.max(logits, keepdims=True))
        probs = logits_exp / np.sum(logits_exp, axis=-1, keepdims=True)
        next_token = np.random.choice(len(probs), p=probs)
        
        # EOS 토큰이면 종료
        if next_token == tokenizer.eos_id():
            break
        
        # 다음 토큰 추가
        tokens = np.concatenate([tokens, np.array([[next_token]], dtype=np.int64)], axis=1)
    
    # 디코딩
    generated_text = tokenizer.decode(tokens[0].tolist())
    return generated_text

# ============================================================================
# 5. 메인 함수
# ============================================================================

def print_program_header():
    """프로그램 시작 헤더 출력"""
    print("=" * 70)
    print("GPT 스타일 언어 모델 훈련")
    print(COPYRIGHT_NOTICE)
    print("=" * 70)

def main():
    print_program_header()

    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='GPT 스타일 언어 모델 훈련')
    parser.add_argument('--batch_size', type=int, default=128, help='배치 크기 (기본값: 128)')
    parser.add_argument('--num_workers', type=int, default=0, help='데이터 로더 워커 수 (기본값: 0)')
    parser.add_argument('--embed_dim', type=int, default=256, help='임베딩 차원 (기본값: 256)')
    parser.add_argument('--num_heads', type=int, default=8, help='어텐션 헤드 수 (기본값: 8)')
    parser.add_argument('--ff_dim', type=int, default=1024, help='잠재차원 차원 (기본값: 1024)')
    parser.add_argument('--stacks', type=int, default=6, help='GPT 블록 수 (기본값: 6)')
    parser.add_argument('--sequence_length', type=int, default=200, help='최대 시퀀스 길이 (기본값: 200)')
    parser.add_argument('--epochs', type=int, default=20, help='훈련 에포크 수 (기본값: 20)')
    parser.add_argument('--load', type=str, metavar='MODEL_FILE', default=None, 
                        help='기존 모델 파일 경로 (지정 시 훈련 생략)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='데이터셋 로컬 경로 또는 HuggingFace ID')
    args = parser.parse_args()

    print(f"\n설정: batch_size={args.batch_size}, num_workers={args.num_workers}, stacks={args.stacks}, epochs={args.epochs}, load={args.load}")
    print(f"모델: embed_dim={args.embed_dim}, num_heads={args.num_heads}, ff_dim={args.ff_dim}, sequence_length={args.sequence_length}")
    
    # 1. SentencePiece 토크나이저 로드
    print("\n[1] 토크나이저 로드 중...")
    model_file = Path('spm.model')
    vocab_file = Path('spm.vocab')
    assert model_file.exists() and vocab_file.exists(), "SentencePiece 모델 파일 필요"
    
    tokenizer = spm.SentencePieceProcessor(model_file=str(model_file))
    print(f"어휘 크기: {tokenizer.vocab_size()}")
    
    # 2. 데이터셋 로드
    print("\n[2] 데이터셋 로드 중...")
    dataset_path = Path(args.dataset)
    csv_files = {f.stem: str(f) for f in sorted(dataset_path.glob('*.csv'))}
    assert csv_files, f"CSV 파일을 찾을 수 없음: {args.dataset}"
    print(f"CSV 파일: {list(csv_files.keys())}")
    한영데이터 = datasets.load_dataset('csv', data_files=csv_files)
    
    시퀀스최대길이 = args.sequence_length
    # 구어체 선택
    한영데이터 = 한영데이터.filter(lambda example: example['source'] in [71265])
    
    # 한국어와 영어를 결합하여 단일 텍스트로 구성 (언어 모델링용)
    def combine_texts(example):
        combined = f"{example['ko']} {example['en']}"
        length = len(tokenizer.encode(combined))
        return {'text': combined, 'length': length}
    
    한영데이터 = 한영데이터.map(combine_texts)
    한영데이터 = 한영데이터.filter(lambda example: example['length'] <= 시퀀스최대길이)
    한영데이터 = 한영데이터.remove_columns(['source', 'ko', 'en', 'length'])
    
    for split in 한영데이터.keys():
        print(f"{split} 데이터셋 크기: {len(한영데이터[split]):,}")
    
    # 3. 데이터로더 생성
    print("\n[3] 데이터로더 생성 중...")
    전처리 = lambda text: tokenizer.encode(text, add_bos=True, add_eos=True)
    datasets_dict = {}
    for split in 한영데이터.keys():
        datasets_dict[split] = TextDataset(
            한영데이터[split]['text'],
            transforms=전처리
        )
        print(f'{split:>12}: {len(datasets_dict[split]):,}')
    
    dataloaders = get_data_loader(
        datasets_dict, batch_size=args.batch_size, collate_fn=배치정규화, num_workers=args.num_workers
    )
    
    checkpoint_filepath = args.load if args.load else 'gpt_model.keras'
    
    # 기존 모델 로드 또는 새로 훈련
    if args.load:
        # 기존 모델 로드
        print(f"\n[4] 기존 모델 로드 중: {args.load}")
        model = keras.models.load_model(checkpoint_filepath)
        model.summary()
        print("\n훈련을 생략하고 평가만 수행합니다.")
    else:
        # 4. 모델 생성 및 컴파일
        print("\n[4] 모델 생성 중...")
        keras.backend.clear_session()
        model = build_model(
            vocab_size=tokenizer.vocab_size(),
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            ff_dim=args.ff_dim,
            num_blocks=args.stacks,
            sequence_length=args.sequence_length
        )
        model.summary()
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy', perplexity_metric]
        )
        
        # 5. 모델 훈련
        print("\n[5] 모델 훈련 중...")
        history = model.fit(
            dataloaders['train'],
            validation_data=dataloaders['validation'],
            epochs=args.epochs,
            callbacks=[
                keras.callbacks.ModelCheckpoint(checkpoint_filepath, save_best_only=True),
                keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
            ]
        )
        
        # 6. 최적 모델 로드
        print("\n[6] 최적 모델 로드 중...")
        model = keras.models.load_model(checkpoint_filepath)
        
        # 7. 훈련 결과 출력
        print("\n[7] 훈련 결과:")
        results = pd.DataFrame(history.history).round(4)
        print(results.tail())
        
        # 8. 결과 저장
        print("\n[8] 결과 저장 중...")
        results.to_csv('gpt_model_history.csv', index=False)
        model.save('gpt_model_final.keras')
        print(f"훈련 기록: gpt_model_history.csv")
        print(f"최종 모델: gpt_model_final.keras")
    
    # 공통: 모델 평가
    step = 5 if args.load else 9
    print(f"\n[{step}] 모델 평가 중...")
    test_metrics = model.evaluate(dataloaders['test'], return_dict=True)
    test_metrics = pd.DataFrame({'test': test_metrics}).round(4)
    print(test_metrics)
    
    # 공통: 텍스트 생성 예제
    step += 1
    print(f"\n[{step}] 텍스트 생성 예제:")
    prompts = [
        "안녕하세요",
        "좋은 아침",
        "감사합니다"
    ]
    for prompt in prompts:
        generated = generate_text(prompt, model, tokenizer, max_length=50, temperature=0.8)
        print(f"\n프롬프트: {prompt}")
        print(f"생성: {generated}")
    
    print("\n" + "="*70)
    print(f"{'평가' if args.load else '훈련'} 완료!")
    if not args.load:
        print(f"최적 모델: {checkpoint_filepath}")
    print("="*70)

if __name__ == '__main__':
    main()
