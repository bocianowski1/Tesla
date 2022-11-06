import re
import numpy as np

def extract_numbers(s):
    number_extract_pattern = "\\d+"
    r = re.findall(number_extract_pattern, s)
    if len(r) > 0:
        return int(''.join(r))

def flatten(arr):
    return [item for sub_arr in arr for item in sub_arr]

def accuracy(y_pred, y_test):
    n = len(y_pred)
    return np.sqrt((1/n) * np.sum(y_pred.detach().numpy() - y_test.numpy())**2)

def equal_lengths(arr):
    for i in range(1, len(arr)):
        if len(arr[i-1]) != len(arr[i]):
            return False
    return True

def print_lengths(arr):
    for a in arr:
        print(len(a))