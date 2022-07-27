from torchvision import transforms
import os
import torch
import math
import torchvision
import torch.nn as nn
from quantize import quantize

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    print('cuda')
model_q = torchvision.models.resnet18(num_classes = 10)
model_q = model_q.to(device)

batch_size = 100
bit_precision = 4


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

quantized = [p for n,p in model_q.named_parameters() if 'bias' not in n and 'bn' not in n and 'downsample' not in n]
lr = 0.01
optimizer_quantized = torch.optim.Adam(quantized, lr=lr)
rest_layers = [p for name,p in model_q.named_parameters()if 'bias'  in name or 'bn' in name or 'downsample' in name ]
optimizer_rest = torch.optim.Adam(rest_layers,lr=lr)

full_precision = quantized.copy()
optimizer_full_precision = torch.optim.Adam(full_precision, lr=lr)

powers_of_2 = torch.arange(bit_precision)
powers_of_2 = 2**powers_of_2
powers_of_2 = powers_of_2.to(torch.float)
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
weights = []
index = 0
for i,(name,_) in enumerate(model_q.named_parameters()):
    if 'bias' not in name and 'bn' not in name and 'downsample' not in name:
        weights.append( torch.Tensor(powers_of_2) / constants[index])
        index+=1
num_epochs = 5

binary_representation = [torch.Tensor([])
        for _ in range(len(quantized))]
PERIOD = 1
def train():
    total_step = len(train_loader)
    print(' ' * 50,']','\b'*52,flush=True,end='')
    for epoch in range(num_epochs):
        for index, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer_quantized.zero_grad()
            optimizer_full_precision.zero_grad()
            # TODO, convert full precision to quantized
            # list of 8-long bool list describing each layer
            for i, f in enumerate(full_precision):
                for _ in range(PERIOD):
                    flat_quantized, binary_representation[i] = quantize(
                        bit_precision, weights[i], f)
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

            optimizer_quantized.zero_grad()
            optimizer_full_precision.zero_grad()
            # Forward pass
            outputs = model_q(images)
            #  print(next(model_q.named_parameters()))
            loss = criterion(outputs, labels)

            #  # Backward and optimize
            #  optimizer_quantized.zero_grad()
            #  optimizer_full_precision.zero_grad()

            loss.backward()
            for q, f in zip(quantized, full_precision):
                f.grad = torch.clone(q.grad)

            optimizer_quantized.zero_grad()
            # TODO convert quantized_grad to weights and full precision grad

            optimizer_full_precision.step()
            optimizer_rest.step()

            if (index+1) %2 == 0:
                print('=',end='',flush=True)
            if (index+1) % 100 == 0:
                print()
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, index+1, total_step, loss.item()))
                print(test())
                print(' ' * 50,']','\b'*52,flush=True,end='')
#  def train():
    #  total_step = len(train_loader)
    #  for epoch in range(num_epochs):
        #  for index, (images, labels) in enumerate(train_loader):
            #  images = images.to(device)
            #  labels = labels.to(device)
            #  optimizer_quantized.zero_grad()

            #  # Forward pass
            #  outputs = model_q(images)
            #  #  print(next(model_q.named_parameters()))
            #  loss = criterion(outputs, labels)

            #  #  # Backward and optimize
            #  #  optimizer_quantized.zero_grad()
            #  #  optimizer_full_precision.zero_grad()

            #  loss.backward()
            #  optimizer_quantized.step()
            #  _, predicted = outputs.max(1)
            #  total = labels.size(0)
            #  correct = predicted.eq(labels).sum().item()
            #  if (index+1) % 100 == 0:
                #  print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      #  .format(epoch+1, num_epochs, index+1, total_step, loss.item()))
                #  print(test())
class AverageMeter:
    def __init__(self):
        self.sum = 0
        self.avg = 0
        self.num = 0
    def update(self,val,cnt):
        self.sum += val * cnt
        self.num += cnt
        self.avg = self.sum/self.num

def test():
    global best_acc
    global test_loader

    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model_q.eval()

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        # measure data loading time
        with torch.no_grad():
            inputs = inputs.to(device)
            targets = targets.to(device)
            #inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            outputs = model_q(inputs)
            loss = criterion(outputs, targets)

        # measure accuracy and record loss
        _,predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        prec1  = correct/inputs.size(0)
        losses.update(loss.data, inputs.size(0))
        top1.update(prec1, inputs.size(0))

        # measure elapsed time

    return (losses.avg, top1.avg)

def accuracy(output, target):
    o = torch.topk(output,1,1)[1]
    return torch.sum(torch.where(o==target,1,0))/target.size(0)


if __name__ == "__main__":
    train()
