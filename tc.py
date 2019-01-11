import math, time, random, gzip, csv, tabulate
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque


np.random.seed(1)
tf.set_random_seed(1)
random.seed(1)


def log(val):
    print(val)
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


def read_categories(filename):
    with open(filename, 'r') as f:
        return np.array([line.strip() for line in f])


def construct_chunk(first, count):
    log("Constructing chunk...")
    # main chunk
    last = first + count
    inputs = np.zeros((count*3, champ_count*2 + 1), dtype=np.float64)
    outputs = np.ones((count*3, 1), dtype=np.float64) # Did first team win
    for ch in range(5):
        inputs[range(count), champs[first:last, ch]] = 1
    for ch in range(5):
        inputs[range(count), champ_count + champs[first:last, 5+ch].astype(np.int32)] = 1
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
    verify_chunk(inputs)
    return inputs, outputs


def verify_chunk(chunk):
    log("Verifying chunk...")
    for r in range(chunk.shape[0]):
        if len(np.nonzero(chunk[r, :champ_count])[0]) != 5:
            raise ValueError('Champion count in team 1 is not 5')
        if len(np.nonzero(chunk[r, champ_count:2*champ_count])[0]) != 5:
            raise ValueError('Champion count in team 2 is not 5')
    log("Chunk verified")


def validate(model):
    predicted = model.predict(val_inputs)
    bins = np.zeros((101, 2))
    for r in range(validation_size*2): # the rest are mirror matchups with an expected output of 0.5
        bin = max(0, min(100, int(predicted[r]*100 + 0.5)))
        bins[bin, 0] += 1
        bins[bin, 1] += val_outputs[r, 0]
    sum = 0
    total = 0
    for bin in range(101):
        if bins[bin, 0] > 0:
            actual_wr = bins[bin, 1] / bins[bin, 0] * 100
            total += bins[bin, 0]
            sum += bins[bin, 0] * ((bin - actual_wr) ** 2)
            log('Bin %d%% (+/- 0.5%%): actual w/r = %2.1f (%s games)' % (bin, actual_wr, '{:,}'.format(int(bins[bin, 0]))))
    log('Validation MSE %%: %.4f' % (sum / total))
    tbl = []
    for r in range(10):
        tbl.append([
            ', '.join(champ_labels[np.nonzero(val_inputs[r, :])[0][0:5]]),
            ', '.join(champ_labels[np.nonzero(val_inputs[r, champ_count:])[0][0:5]]),
            predicted[r]])
    log(tabulate.tabulate(tbl, headers=['Team 1', 'Team 2', 'P(Team 1 Win)'], tablefmt='orgtbl'))


logfile = open('log.txt', 'a')
log("==============================")

log("Loading match data...")
champ_labels = read_categories('data/teamcomp.plr.champ.classes.csv')
champ_count = len(champ_labels)
timestamps = read_shr_uint16('data/teamcomp.timestamp.shr.gz', 1)
champs = read_shr_uint8('data/teamcomp.plr.champ.shr.gz', 10) # 5 winners, followed by 5 losers
match_count = timestamps.shape[0]
shuf = np.random.permutation(match_count)
timestamps = timestamps[shuf]
champs = champs[shuf]
log("Loaded {:,} matches".format(match_count))

chunk_size = 100000 # 1 million size = 3 million training samples = 7 GB of RAM
validation_size = 100000
val_inputs, val_outputs = construct_chunk(0, validation_size)

model = keras.Sequential()
model.add(keras.layers.Dense(128, activation=tf.nn.elu, input_shape=(val_inputs.shape[1],)))
model.add(keras.layers.Dense(128, activation=tf.nn.elu))
model.add(keras.layers.Dense(128, activation=tf.nn.elu))
model.add(keras.layers.Dense(128, activation=tf.nn.elu))
model.add(keras.layers.Dense(1))

model.summary()

learn_rate = 0.0001

start_time = time.time()
epoch = 1
chunks = (match_count-validation_size) // chunk_size
chunk_size = match_count // chunks
log('Using chunk size {:,}'.format(chunk_size))
while True:
    model.compile(loss = 'mse', optimizer = keras.optimizers.Adam(learn_rate), metrics = ['mse', 'mae'])
    for c in range(chunks):
        log("Training on chunk %d of %d" % (c+1, chunks))
        inputs, outputs = construct_chunk(validation_size+c*chunk_size, chunk_size)
        hist = model.fit(inputs, outputs, epochs=1, batch_size=32, shuffle=True)
        log(abs(hist.history["loss"][-1] - 0.25))
        model.save('models/tc-%04d-%03d.h5' % (epoch, c+1))
        validate(model)

    epoch = epoch + 1
    learn_rate = 0.8 * learn_rate
