import math, time, random, gzip, csv, tabulate, json, os
import numpy as np
import tensorflow as tf
from tensorflow import keras


save_path = 'models/test'  # currently meant to be modified in code


class TrainingState:
    def __init__(self, js):
        self.chunk_size = 100_000 # 1 million size = 3 million training samples = 7 GB of RAM
        self.validation_size = 100_000
        self.batch_size = 32
        self.floatx = 'float32'
        self.random_seed = 1
        self.learn_rate = 0.00001
        self.cur_epoch = 1
        self.cur_chunk = 1
        self.total_time_sec = 0
        if js is not None:
            self.__dict__ = {**self.__dict__, **json.loads(js)}

    def to_json(self):
        return json.dumps(self.__dict__, indent=4)

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
    print("Constructing chunk...")
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

    print("Chunk constructed")
    verify_chunk(inputs)
    return inputs, outputs


def verify_chunk(chunk):
    print("Verifying chunk...")
    for r in range(chunk.shape[0]):
        if len(np.nonzero(chunk[r, :champ_count])[0]) != 5:
            raise ValueError(f'Champion count in team 1 is not 5 (row {r})')
        if len(np.nonzero(chunk[r, champ_count:2*champ_count])[0]) != 5:
            raise ValueError(f'Champion count in team 2 is not 5 (row {r})')
    print("Chunk verified")


def validate(model):
    print("Validate: predicting...")
    predicted = model.predict(val_inputs)
    print("Validate: computing stats...")
    bins = np.zeros((101, 2))
    stat_count = 0
    stat_mean = 0
    stat_m2 = 0
    for r in range(val_inputs.shape[0]):
        bin = max(0, min(100, int(predicted[r]*100 + 0.5)))
        bins[bin, 0] += 1
        bins[bin, 1] += val_outputs[r, 0]
        stat_count += 1
        delta = predicted[r]*100 - stat_mean
        stat_mean += delta / stat_count
        delta2 = predicted[r]*100 - stat_mean
        stat_m2 += delta * delta2
    stdev = math.sqrt(stat_m2 / stat_count)

    sum = 0
    total = 0
    for bin in range(101):
        if bins[bin, 0] > 0:
            actual_wr = bins[bin, 1] / bins[bin, 0] * 100
            total += bins[bin, 0]
            sum += bins[bin, 0] * ((bin - actual_wr) ** 2)
            log(f'Bin {bin}% (+/- 0.5%): actual w/r = {actual_wr:2.1f} ({int(bins[bin, 0]):,} games)')
    mse = sum / total

    log(f'Validation MSE %: {mse:.4f}, stdev: {stdev:.4f}')

    tbl = []
    for r in range(10):
        tbl.append([
            ', '.join(champ_labels[np.nonzero(val_inputs[r, :])[0][0:5]]),
            ', '.join(champ_labels[np.nonzero(val_inputs[r, champ_count:])[0][0:5]]),
            predicted[r]])
    log(tabulate.tabulate(tbl, headers=['Team 1', 'Team 2', 'P(Team 1 Win)'], tablefmt='orgtbl'))

    return (mse, stdev)


def revalidate_old(epoch, chunk, end_epoch, end_chunk):
    while True:
        start = time.time()
        print("Loading model...")
        model = keras.models.load_model(get_checkpoint_path(epoch, chunk))
        (val_mse, val_stdev) = validate(model)
        with open(f'{save_path}/progress_old.csv', 'a') as f:
            print(f"-1,,{epoch},{chunk},,-1,,{val_mse},{val_stdev}", file=f)
        chunk += 1
        if chunk > chunk_count:
            chunk = 1
            epoch += 1
        if epoch == end_epoch and chunk == end_chunk:
            exit()
        del model
        keras.backend.clear_session()
        print (time.time() - start)


def get_checkpoint_path(epoch, chunk):
    return f'{save_path}/checkpoints/tc-{epoch:04d}-{chunk:03d}.h5'


state_file_path = f'{save_path}/state.json'
if not os.path.isfile(state_file_path):
    state = TrainingState(None)
    os.mkdir(save_path)
    print(f'Initialised a blank state; please edit {state_file_path} and restart')
    exit()

with open(f'{save_path}/state.json', 'r') as f:
    state = TrainingState(f.read())

logfile = open(f'{save_path}/log.txt', 'a')
log("==============================")
log("Loaded state:")
log(state.to_json())

keras.backend.set_floatx(state.floatx)
np.random.seed(state.random_seed)
tf.set_random_seed(state.random_seed)
random.seed(state.random_seed)

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

chunk_count = (match_count - state.validation_size) // state.chunk_size
state.chunk_size = (match_count - state.validation_size) // chunk_count
log(f'Using chunk size {state.chunk_size:,}')

val_inputs, val_outputs = construct_chunk(0, state.validation_size)
val_inputs = val_inputs[:2*state.validation_size, :]
val_outputs = val_outputs[:2*state.validation_size, :]

#revalidate_old(1, 119, 5, 237)

if state.cur_epoch == 1 and state.cur_chunk == 1:
    with open(f'{save_path}/model.py', 'r') as f:
        l = locals().copy()
        g = globals().copy()
        exec(f.read(), l, g)
        if not 'build_model' in g:
            raise ValueError('model.py should def a function named build_model accepting an input_shape')
        model = g['build_model'](val_inputs[0].shape)
else:
    last_epoch = state.cur_epoch
    last_chunk = state.cur_chunk - 1
    if last_chunk < 1:
        last_epoch -= 1
        last_chunk = chunk_count
    print("Loading model...")
    model = keras.models.load_model(get_checkpoint_path(last_epoch, last_chunk))

model.summary()

start_time = time.time()
compiled_learn_rate = -1
while True:
    if state.learn_rate != compiled_learn_rate:
        print("Compiling model...")
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
    print("Saving model...")
    model.save(get_checkpoint_path(state.cur_epoch, state.cur_chunk))
    (val_mse, val_stdev) = validate(model)

    t = time.time()
    state.total_time_sec += t - start_time
    start_time = t

    with open(f'{save_path}/progress.csv', 'a') as f:
        print(f"{state.total_time_sec:.1f},,{state.cur_epoch},{state.cur_chunk},,{hist.history['loss'][-1]},,{val_mse},{val_stdev}", file=f)

    state.cur_chunk += 1
    if state.cur_chunk > chunk_count:
        state.cur_chunk = 1
        state.cur_epoch += 1
        state.learn_rate *= 0.1

    with open(state_file_path, 'w') as f:
        print(state.to_json(), file=f)
