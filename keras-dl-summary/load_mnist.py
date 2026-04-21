from torchvision.datasets import MNIST
import torchvision.transforms as transforms

전처리 = transforms.ToTensor()

mnist = {}
for split in ['train', 'test']:
    mnist[split] = MNIST(root='data', train=(split=='train'), download=True, transform=전처리)
    print(f'{split:<5}: {len(mnist[split]):,}')