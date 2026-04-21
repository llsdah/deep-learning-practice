from torchvision.datasets import MNIST
import torchvision.transforms as transform

preprocessing = transform.ToTensor()

mnist = {}

for split in ['train','test']:
    mnist[split] = MNIST(root='data', train=(split=='train'), download=True, transform=preprocessing)
    print(f'(split:<5): {len(mnist[split]):,}')
    