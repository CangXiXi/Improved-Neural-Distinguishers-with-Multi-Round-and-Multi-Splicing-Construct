import numpy as np
from os import urandom
#Speck64/128
def WORD_SIZE():
    return (32)

def ALPHA():
    return (8)

def BETA():
    return (3)

MASK_VAL = 2 ** WORD_SIZE() - 1

def left_round(value, shiftBits):
	t1 = (value >> (WORD_SIZE() - shiftBits)) ^ (value << shiftBits)
	t2 = ((2 ** WORD_SIZE()) - 1)
	return t1 & t2

def right_round(value, shiftBits):
	t1 = (value << (WORD_SIZE() - shiftBits)) ^ (value >> shiftBits)
	t2 = ((2 ** WORD_SIZE()) - 1)
	return t1 & t2

def enc_one_round(p, k):
    c0, c1 = p[0], p[1]
    c0 = right_round(c0, ALPHA())
    c0 = (c0 + c1) & MASK_VAL
    c0 = c0 ^ k
    c1 = left_round(c1, BETA())
    c1 = c1 ^ c0
    return(c0,c1)

def dec_one_round(c,k):
    c0, c1 = c[0], c[1]
    c1 = c1 ^ c0
    c1 = right_round(c1, BETA())
    c0 = c0 ^ k
    c0 = (c0 - c1) & MASK_VAL
    c0 = left_round(c0, ALPHA())
    return(c0, c1)

def expand_key(k, t):
    ks = [0 for i in range(t)]
    ks[0] = k[len(k)-1]
    l = list(reversed(k[:len(k)-1]))
    for i in range(t-1):
        l[i%3], ks[i+1] = enc_one_round((l[i%3], ks[i]), i)
    return(ks)

def encrypt(p, ks):
    x, y = p[0], p[1]
    for k in ks:
        x,y = enc_one_round((x,y), k)
    return(x, y)

def decrypt(c, ks):
    x, y = c[0], c[1]
    for k in reversed(ks):
        x, y = dec_one_round((x,y), k)
    return (x,y)

def check_testvector():
    key = (0x1b1a1918,0x13121110, 0x0b0a0908, 0x03020100)
    pt = (0x3b726574, 0x7475432d)
    ks = expand_key(key, 27)
    ct = encrypt(pt, ks)
    if (ct == (0x8c6fa548,0x454e028b)):  
        print("Testvector verified.")
        return(True)
    else:
        print("Testvector not verified.")
        return(False)


def convert_to_binary(arr):
    X = np.zeros((256 * WORD_SIZE(),len(arr[0])),dtype=np.uint8)
    for i in range(256 * WORD_SIZE()):
        index = i // WORD_SIZE();
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
        X[i] = (arr[index] >> offset) & 1
    X = X.transpose()
    return (X)


def make_train_data(n, nr, diff):
    k = 1
    num =(64*64*k)/128
    X = []
    Y = np.frombuffer(urandom(n), dtype=np.uint8) 
    Y = Y & 1
    keys = np.frombuffer(urandom(16 * n), dtype=np.uint32).reshape(4, -1)
    ks = expand_key(keys, nr)
    ks_back = np.frombuffer(urandom(16), dtype=np.uint16).reshape(4, -1)
    ks_back = expand_key(ks_back, 1)
    ks_use_back = np.broadcast_to(ks_back, (1, n))
    for i in range(int(num)):
        plain0l = np.frombuffer(urandom(4*n),dtype=np.uint32)
        plain0r = np.frombuffer(urandom(4*n),dtype=np.uint32)
        plain1l = plain0l ^ diff[0]
        plain1r = plain0r ^ diff[1]
        num_rand_samples = np.sum(Y==0)
        plain1l[Y==0] = np.frombuffer(urandom(4*num_rand_samples),dtype=np.uint32)
        plain1r[Y==0] = np.frombuffer(urandom(4*num_rand_samples),dtype=np.uint32)
        ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
        ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks)
        ctdata0l2, ctdata0r2 = dec_one_round((ctdata0l, ctdata0r), ks_use_back)
        ctdata1l2, ctdata1r2 = dec_one_round((ctdata1l, ctdata1r), ks_use_back)
        X += [ctdata0l, ctdata0r , ctdata1l, ctdata1r, ctdata0l2, ctdata0r2, ctdata1l2, ctdata1r2]
        i += 1
    X = convert_to_binary(X)

    return (X,Y,ks_back)

def make_train_val_data(n, nr, diff,ks_back):
    k = 1
    num =(64*64*k)/128
    X = []
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    keys = np.frombuffer(urandom(16 * n), dtype=np.uint32).reshape(4, -1)
    ks = expand_key(keys, nr)
    ks_use_back = np.broadcast_to(ks_back, (1, n))
    for i in range(int(num)):
        plain0l = np.frombuffer(urandom(4*n),dtype=np.uint32)
        plain0r = np.frombuffer(urandom(4*n),dtype=np.uint32)
        plain1l = plain0l ^ diff[0]
        plain1r = plain0r ^ diff[1]
        num_rand_samples = np.sum(Y==0)
        plain1l[Y==0] = np.frombuffer(urandom(4*num_rand_samples),dtype=np.uint32)
        plain1r[Y==0] = np.frombuffer(urandom(4*num_rand_samples),dtype=np.uint32)
        ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
        ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks)
        ctdata0l2, ctdata0r2 = dec_one_round((ctdata0l, ctdata0r), ks_use_back)
        ctdata1l2, ctdata1r2 = dec_one_round((ctdata1l, ctdata1r), ks_use_back)
        X += [ctdata0l, ctdata0r , ctdata1l, ctdata1r, ctdata0l2, ctdata0r2, ctdata1l2, ctdata1r2]
        i += 1
    X = convert_to_binary(X)

    return (X,Y)