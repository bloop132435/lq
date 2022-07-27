from typing import Tuple
from bisect import bisect_left
import math
import torch
import numpy as np
import itertools

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    print('cuda')

def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest index to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return 0
    if pos == len(myList):
        return len(myList)-1
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return pos
    else:
        return pos-1


def quantize( num_bits: int, weights: torch.Tensor, full_precision: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #  all_binary_values = torch.from_numpy(np.fromiter((itertools.product([-1,1],repeat=num_bits)),np.float32))
    all_binary_values = torch.cartesian_prod(*[torch.Tensor([-1,1]) for _ in range(num_bits)])
    all_possible_values = torch.matmul(
        all_binary_values.to(torch.float), weights.to(torch.float))
    flat_precision = torch.clone(full_precision.flatten())
    min_mse = torch.full_like(flat_precision,10000)
    indicies = torch.zeros_like(flat_precision)
    for i in range(2 ** num_bits):
        value = torch.full_like(flat_precision,all_possible_values[i].item())
        mse = (value - flat_precision) ** 2
        new_mse = torch.minimum(mse,min_mse)
        mse_delta = min_mse - new_mse
        min_mse = new_mse
        indicies = torch.where(mse_delta!=0,i,indicies)
    quantized_value = all_possible_values[indicies.long()]
    quantized_binary = all_binary_values[indicies.long()]
    return quantized_value, quantized_binary

def old_quantize(num_bits: int, weights: torch.Tensor, full_precision: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    all_binary_values = torch.Tensor(
        list(itertools.product([-1, 1], repeat=num_bits)))
    all_possible_values = torch.matmul(
        all_binary_values.to(torch.float), weights.to(torch.float))
    combo = []
    for i in range(len(all_binary_values)):
        item = list([x.item() for x in all_binary_values[i]])
        item.append(all_possible_values[i].item())
        combo.append(item)
    combo.sort(key=lambda x: x[-1])
    combo = torch.Tensor(combo)
    all_binary_values = combo[:, :-1]
    all_possible_values = combo[:, -1]

    quantized_value = torch.zeros_like(full_precision)
    size = list(full_precision.size())[0]
    quantized_binary = torch.zeros((size, num_bits))
    for i, x in enumerate(full_precision.flatten()):
        idx = torch.argmin(torch.abs(all_possible_values-x.item()))
        quantized_value[i] = all_possible_values[idx]
        quantized_binary[i] = all_binary_values[idx]
    return quantized_value, quantized_binary

if __name__ == "__main__":
    import timeit
    powers_of_2 = torch.arange(3)
    powers_of_2 = 2**powers_of_2
    arr = torch.randint(-10, 10, [8000])
    print(arr)
    print(quantize(3, powers_of_2, arr))
    #  print(timeit.timeit('old_quantize(3,powers_of_2,arr)',number=100,globals={'old_quantize':old_quantize,'powers_of_2':powers_of_2,'arr':arr}))
    print(timeit.timeit('quantize(3,powers_of_2,arr)',number=10000,globals={'quantize':quantize,'powers_of_2':powers_of_2,'arr':arr}))
# def quantize_grad()
