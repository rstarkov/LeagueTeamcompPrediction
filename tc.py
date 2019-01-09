import math, time, random, gzip, csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque

np.random.seed(1)
tf.set_random_seed(1)
random.seed(1)

def read_shr_uint8(filename, cols):
    arrs = []
    blocks = 0
    with gzip.open(filename,'rb') as f:
        while True:
            arr = np.fromstring(f.read(10000 * cols), dtype=np.uint8)
            if len(arr) > 0:
                arrs.append(arr)
            if len(arr) < 10000 * cols:
                break
            block_id = int.from_bytes(f.read(4), byteorder='little')
            if block_id - 0x76543210 != blocks:
                raise ValueError('Invalid block number: %d' % block_id)
            blocks = blocks + 1

    arr = np.concatenate(arrs)
    arr.shape = (len(arr) // cols, cols)
    return arr

def read_shr_uint16(filename, cols):
    arrs = []
    blocks = 0
    with gzip.open(filename,'rb') as f:
        while True:
            arr = np.fromstring(f.read(2 * 10000 * cols), dtype=np.uint16)
            if len(arr) > 0:
                arrs.append(arr)
            if len(arr) < 10000 * cols:
                break
            block_id = int.from_bytes(f.read(4), byteorder='little')
            if block_id - 0x76543210 != blocks:
                raise ValueError('Invalid block number: %d' % block_id)
            blocks = blocks + 1

    arr = np.concatenate(arrs)
    arr.shape = (len(arr) // cols, cols)
    return arr

def read_shr_categories(filename):
    d = {}
    with open(filename, 'r') as f:
      reader = csv.reader(f)
      for k, v in reader: d[k] = v
    return d

print("Loading match data...")
champLabels = read_shr_categories('Data/teamcomp.plr.champ.classes.csv')
champCount = len(champLabels)
timestamps = read_shr_uint16('Data/teamcomp.timestamp.shr.gz', 1)
champs = read_shr_uint8('Data/teamcomp.plr.champ.shr.gz', 10) # 5 winners, followed by 5 losers
print("Loaded {:,} matches".format(timestamps.shape[0]))

def construct_batch(first, count):
    # main batch
    last = first + count
    inputs = np.zeros((count*2, champCount*2 + 1))
    outputs = np.ones((count*2, 1)) # Did first team win
    for ch in range(5):
        inputs[range(count), champs[first:last, ch]] = 1
    for ch in range(5):
        inputs[range(count), champCount + champs[first:last, 5+ch]] = 1
    inputs[:count, -1] = timestamps[first:last, 0]

    # mirrored
    inputs[count:, :champCount] = inputs[:count, champCount:(2*champCount)]
    inputs[count:, champCount:(2*champCount)] = inputs[:count, :champCount]
    inputs[count:, -1] = inputs[:count, -1]
    outputs[count:, 0] = 0

    return inputs, outputs

test, outputs = construct_batch(0, 10)
np.savetxt("dump.csv", test, delimiter=",")
print(outputs)
