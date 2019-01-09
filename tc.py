import numpy as np
import gzip

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


arr = read_shr_uint16('Data/teamcomp.duration.shr.gz', 1)
print(arr.shape)
print(arr[0:10,:])

arr = read_shr_uint16('Data/teamcomp.timestamp.shr.gz', 1)
print(arr.shape)
print(arr[0:10,:])

arr = read_shr_uint8('Data/teamcomp.queue.shr.gz', 1)
print(arr.shape)
print(arr[0:10,:])

arr = read_shr_uint8('Data/teamcomp.plr.champ.shr.gz', 10)
print(arr.shape)
print(arr[0:10,:])

arr = read_shr_uint8('Data/teamcomp.plr.spell.shr.gz', 20)
print(arr.shape)
print(arr[0:10,:])

arr = read_shr_uint8('Data/teamcomp.plr.rank.shr.gz', 10)
print(arr.shape)
print(arr[0:10,:])
