import math, time, random, gzip, csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque

np.random.seed(1)
tf.set_random_seed(1)
random.seed(1)

def log(val):
    print("LOG:", val)
    print(val, file=logfile)
    logfile.flush()

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

def construct_chunk(first, count):
    log("Constructing chunk...")
    # main chunk
    last = first + count
    inputs = np.zeros((count*3, champ_count*2 + 1))
    outputs = np.ones((count*3, 1)) # Did first team win
    for ch in range(5):
        inputs[range(count), champs[first:last, ch]] = 1
    for ch in range(5):
        inputs[range(count), champ_count + champs[first:last, 5+ch]] = 1
    inputs[:count, -1] = timestamps[first:last, 0]

    # mirrored
    inputs[count:2*count, :champ_count] = inputs[:count, champ_count:2*champ_count]
    inputs[count:2*count, champ_count:2*champ_count] = inputs[:count, :champ_count]
    inputs[count:2*count, -1] = inputs[:count, -1]
    outputs[count:2*count, 0] = 0

    # equalized (not all of the games)
    inputs[2*count:, :] = inputs[:count, :]
    inputs[2*count:, champ_count:2*champ_count] = inputs[2*count:, :champ_count]
    outputs[2*count:, 0] = 0.5

    log("Chunk constructed")
    return inputs, outputs

logfile = open('log.txt', 'a')
log("==============================")

log("Loading match data...")
champ_labels = read_shr_categories('data/teamcomp.plr.champ.classes.csv')
champ_count = len(champ_labels)
timestamps = read_shr_uint16('data/teamcomp.timestamp.shr.gz', 1)
champs = read_shr_uint8('data/teamcomp.plr.champ.shr.gz', 10) # 5 winners, followed by 5 losers
matchCount = timestamps.shape[0]
shuf = np.random.permutation(match_count)
timestamps = timestamps[shuf]
champs = champs[shuf]
log("Loaded {:,} matches".format(match_count))

chunk_size = 1000000 # 1 million size = 3 million training samples = 7 GB of RAM
validation_size = 100000
inputs, outputs = construct_chunk(0, 10)

model = keras.Sequential()
model.add(keras.layers.Dense(256, activation=tf.nn.elu, input_shape=(inputs.shape[1],)))
model.add(keras.layers.Dense(128, activation=tf.nn.elu))
model.add(keras.layers.Dense(64, activation=tf.nn.elu))
model.add(keras.layers.Dense(64, activation=tf.nn.elu))
model.add(keras.layers.Dense(32, activation=tf.nn.elu))
model.add(keras.layers.Dense(32, activation=tf.nn.elu))
model.add(keras.layers.Dense(32, activation=tf.nn.elu))
model.add(keras.layers.Dense(32, activation=tf.nn.elu))
model.add(keras.layers.Dense(1))

model.summary()

learn_rate = 0.0001

start_time = time.time()
epoch = 1
chunks = match_count // chunk_size
chunk_size = match_count // chunks
log('Using chunk size {:,}'.format(chunk_size))
while True:
    model.compile(loss = 'mse', optimizer = keras.optimizers.RMSprop(learn_rate), metrics = ['mse', 'mae'])
    for c in range(chunks):
        log("Training on chunk %d of %d" % (c+1, chunks))
        inputs, outputs = construct_chunk(c*chunk_size, chunk_size)
        hist = model.fit(inputs, outputs, epochs=1, batch_size=32, shuffle=True)
        log(abs(hist.history["loss"][-1] - 0.25))
        model.save('models/tc-%04d-%03d.h5' % (epoch, ))

    epoch = epoch + 1
    learn_rate = 0.8 * learn_rate
