from typing import Tuple
from bisect import bisect_left
import math
import torch
import itertools


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


def quantize(num_bits: int, weights: torch.Tensor, full_precision: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
    all_possible_values = torch.Tensor(combo[:, -1])

    size = list(full_precision.size())[0]
    flat_precision = torch.clone(full_precision.flatten())
    min_mse = torch.ones_like(flat_precision) * 10000
    indicies = torch.zeros_like(flat_precision)
    for i in range(2 ** num_bits):
        value = all_possible_values[i] * torch.ones_like(flat_precision)
        mse = (value - flat_precision) ** 2
        new_mse = torch.minimum(mse,min_mse)
        mse_delta = min_mse - new_mse
        min_mse = new_mse
        new_mask = torch.where(mse_delta==0,0,1)
        orig_mask = torch.where(mse_delta==0,1,0)
        new_indicies = torch.ones_like(flat_precision) * i * new_mask + indicies * orig_mask
        indicies = new_indicies
    quantized_value = all_possible_values[indicies.long()]
    quantized_binary = all_binary_values[indicies.long()]
    return quantized_value, quantized_binary


if __name__ == "__main__":
    powers_of_2 = torch.arange(3)
    powers_of_2 = 2**powers_of_2
    arr = torch.randint(-10, 10, [8])
    print(arr)
    print(quantize(3, powers_of_2, arr))
# def quantize_grad()
