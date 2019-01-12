import math, time, random, gzip, csv, tabulate
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque


class TrainingState:
    chunk_size = 100_000 # 1 million size = 3 million training samples = 7 GB of RAM
    validation_size = 100_000
    batch_size = 32
    floatx = 'float32'
    random_seed = 1
    learn_rate = 0.00001
    cur_epoch = 1
    cur_chunk = 1

state = TrainingState()

keras.backend.set_floatx(state.floatx)
np.random.seed(state.random_seed)
tf.set_random_seed(state.random_seed)
random.seed(state.random_seed)


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
                raise ValueError(f'Invalid block number: {block_id}')
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
                raise ValueError(f'Invalid block number: {block_id}')
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
            raise ValueError(f'Champion count in team 1 is not 5 (row {r})')
        if len(np.nonzero(chunk[r, champ_count:2*champ_count])[0]) != 5:
            raise ValueError(f'Champion count in team 2 is not 5 (row {r})')
    log("Chunk verified")


def validate(model):
    predicted = model.predict(val_inputs)
    bins = np.zeros((101, 2))
    for r in range(val_inputs.shape[0]):
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
            log(f'Bin {bin}% (+/- 0.5%): actual w/r = {actual_wr:2.1f} ({int(bins[bin, 0]):,} games)')
    log(f'Validation MSE %: {sum / total:.4f}')
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
log(f"Loaded {match_count:,} matches")

val_inputs, val_outputs = construct_chunk(0, state.validation_size)
val_inputs = val_inputs[:2*state.validation_size, :]
val_outputs = val_outputs[:2*state.validation_size, :]

model = keras.Sequential()
model.add(keras.layers.Dense(128, activation=tf.nn.elu, input_shape=(val_inputs.shape[1],)))
model.add(keras.layers.Dense(128, activation=tf.nn.elu))
model.add(keras.layers.Dense(128, activation=tf.nn.elu))
model.add(keras.layers.Dense(128, activation=tf.nn.elu))
model.add(keras.layers.Dense(1))

model.summary()

start_time = time.time()
chunk_count = (match_count - state.validation_size) // state.chunk_size
state.chunk_size = (match_count - state.validation_size) // chunk_count
compiled_learn_rate = -1
log(f'Using chunk size {state.chunk_size:,}')
while True:
    if state.learn_rate != compiled_learn_rate:
        model.compile(loss = 'mse', optimizer = keras.optimizers.Adam(state.learn_rate), metrics = ['mse', 'mae'])
        compiled_learn_rate = state.learn_rate

    log(f"Training on chunk {state.cur_chunk} of {chunk_count}")
    inputs, outputs = construct_chunk(state.validation_size + (state.cur_chunk-1)*state.chunk_size, state.chunk_size)
    if state.cur_epoch == 1 and state.cur_chunk == 1:
        epochs = 10
    elif state.cur_epoch == 1 and state.cur_chunk == 2:
        epochs = 5
    else:
        epochs = 2
    hist = model.fit(inputs, outputs, epochs=epochs, batch_size=state.batch_size, shuffle=True)
    log(abs(hist.history["loss"][-1] - 0.25))
    model.save('models/tc-%04d-%03d.h5' % (state.cur_epoch, state.cur_chunk))
    validate(model)

    state.cur_chunk += 1
    if state.cur_chunk > chunk_count:
        state.cur_chunk = 0
        state.cur_epoch += 1
        state.learn_rate *= 0.1
