from torchvision import transforms
import os
import torch
import torchvision
import torch.nn as nn
from quantize import quantize

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    print('cuda')
model = torchvision.models.resnet18()

batch_size = 100
bit_precision = 8


mnist_folder = os.path.join(os.path.dirname(__file__), 'data')
train_dataset = torchvision.datasets.CIFAR10(root=mnist_folder,
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root=mnist_folder,
                                            train=False,
                                            transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

criterion = nn.CrossEntropyLoss()

quantized = [p for n,p in model.named_parameters() if 'bias' not in n and 'bn' not in n and 'downsample' not in n]
optimizer_quantized = torch.optim.Adam(quantized, lr=0.001)

full_precision = quantized.copy()
optimizer_full_precision = torch.optim.Adam(full_precision, lr=0.001)

powers_of_2 = torch.arange(bit_precision)
powers_of_2 = 2**powers_of_2
powers_of_2 = powers_of_2.to(torch.float)
constants = [25600,256000,25600,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
constants = [
        25600,
        256000,
        25600,
        25600,
        25600,
        25600,
        25600,
        25600,
        25600,
        25600,
        25600,
        25600,
        25600,
        25600,
        25600,
        25600,
        25600,
        25600,
        25600,
        25600,
        25600,
        25600,
        25600,
        25600,
        25600,
        25600,
        25600,
        25600,
        25600,
        25600,
        ]
weights = [torch.Tensor(powers_of_2.to(torch.float))/constants[i//3]
           for i,(name,_) in enumerate(model.named_parameters()) if 'bias' not in name and 'bn' not in name and 'downsample' not in name]

num_epochs = 5


def train():
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for index, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # TODO, convert full precision to quantized
            # list of 8-long bool list describing each layer
            binary_representation = [torch.Tensor([])
                                     for _ in range(len(quantized))]
            for i, f in enumerate(full_precision):
                flat_quantized, binary_representation[i] = quantize(
                    bit_precision, weights[i], f.flatten())
                quantized[i].data = flat_quantized.reshape(quantized[i].size())
                # binr = torch.where(
                #     binary_representation[i] > 0, 1, 0).to(torch.float)
                binr = (torch.clone(binary_representation[i]))
                b = torch.t(binr)
                bbt = torch.matmul(b,torch.t(b))
                inv = torch.inverse(bbt)
                bbtb = torch.matmul(inv,b)
                intermediate = torch.matmul(bbtb,f.flatten())
                weights[i] = intermediate
                print(i,end='')
                pass
            print()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer_quantized.zero_grad()
            optimizer_full_precision.zero_grad()

            loss.backward()
            for q, f in zip(quantized, full_precision):
                f.grad = torch.Tensor(q.grad)

            optimizer_quantized.zero_grad()
            # TODO convert quantized_grad to weights and full precision grad

            optimizer_full_precision.step()

            if (index+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, index+1, total_step, loss.item()))



if __name__ == "__main__":
    train()
