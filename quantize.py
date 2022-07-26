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
    powers_of_2 = torch.arange(8)
    powers_of_2 = 2**powers_of_2
    arr = torch.randint(-10, 10, (2, 3, 4))
    print(arr)
    print(quantize(8, powers_of_2, arr))
# def quantize_grad()
