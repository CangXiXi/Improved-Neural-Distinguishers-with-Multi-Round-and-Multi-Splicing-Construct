#Simon64/128
import numpy as np
from os import urandom

def block_size():
    return(64)

def WORD_SIZE():
    return(32)

def key_words():
    return(4)

def const_seq():
    return(8)

def all_round():
    return(44)

Z0 = [1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0]
Z1 = [1,0,0,0,1,1,1,0,1,1,1,1,1,0,0,1,0,0,1,1,0,0,0,0,1,0,1,1,0,1,0,1,0,0,0,1,1,1,0,1,1,1,1,1,0,0,1,0,0,1,1,0,0,0,0,1,0,1,1,0,1,0]
Z3 = [1,1,0,1,1,0,1,1,1,0,1,0,1,1,0,0,0,1,1,0,0,1,0,1,1,1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,1,0,0,1,1,1,0,0,1,1,0,1,0,0,0,0,1,1,1,1]

def left_round(value, shiftBits):
	t1 = (value >> (WORD_SIZE() - shiftBits)) ^ (value << shiftBits)
	t2 = ((2 ** WORD_SIZE()) - 1)
	return t1 & t2

def right_round(value, shiftBits):
	t1 = (value << (WORD_SIZE() - shiftBits)) ^ (value >> shiftBits)
	t2 = ((2 ** WORD_SIZE()) - 1)
	return t1 & t2

def enc_one_round(p,k):
    x,y=p[0],p[1]
    tmp=x
    x=y^((left_round(x,1))&(left_round(x,8)))^(left_round(x,2))^k
    y=tmp
    return (x,y)

def dec_one_round(c, k):
    x,y=c[0],c[1]
    tmp=y
    y=x^k^(left_round(y,2))^((left_round(y,1))&(left_round(y,8)))
    x=tmp
    return (x,y)

def expand_key(k,t):
    subkey= [0 for i in range(t)]
    c=pow(2,WORD_SIZE())-4
    m=key_words()
    for i in range(0,m):
        subkey[i]=k[m-1-i]
    for i in range(m,t):
        tmp=right_round(subkey[i-1], 3)
        if (m==4):
            tmp=tmp^subkey[i-3]
        tmp=tmp^(right_round(tmp,1))
        subkey[i]=c^Z3[(i-m)%62]^tmp^subkey[i-m]
    return(subkey)

def expand_key_1(k):
    subkey= [0]
    m=key_words()
    subkey[0]=k[m-1]
    return(subkey)

def encrypt(p,ks):
    x,y= p[0],p[1]
    for k in ks:
        x,y=enc_one_round((x,y), k)
    return(x,y)

def decrypt(c, ks):
    x, y = c[0], c[1]
    
    for k in reversed(ks):
        x, y = dec_one_round((x,y), k)
    return(x,y)

def check_testvector():
    key = (0x1b1a1918,0x13121110, 0x0b0a0908, 0x03020100)
    pt = (0x656b696c, 0x20646e75)
    ks = expand_key(key, 44)
    ct = encrypt(pt, ks)
    if (ct == (0x44c8fc20, 0xb9dfa07a)):
        print("Testvector verified.")
        return(True)
    else:
        print("Testvector not verified.")
        return(False)

        
def convert_to_binary(arr):
    X = np.zeros((128 * WORD_SIZE(),len(arr[0])),dtype=np.uint8)
    for i in range(128 * WORD_SIZE()):
        index = i // WORD_SIZE();
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
        X[i] = (arr[index] >> offset) & 1
    X = X.transpose()
    return (X)


def make_train_data(n, nr, diff):
    k = 1
    num =(64*64*k)/256
    X = []
    Y = np.frombuffer(urandom(n), dtype=np.uint8) 
    Y = Y & 1
    ks_back = np.frombuffer(urandom(8), dtype=np.uint16).reshape(4, -1)
    ks_back = expand_key_1(ks_back)
    keys = np.frombuffer(urandom(16 * n), dtype=np.uint32).reshape(4, -1)
    keys = keys % pow(2, 24)
    ks_use_back = np.broadcast_to(ks_back, (1, n))
    ks = expand_key(keys, nr)
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
        ctdatal2, ctdatar2 = ctdata0l2 ^ ctdata1l2, ctdata0r2 ^ ctdata1r2
        ctdatal, ctdatar = ctdata0l^ctdata1l, ctdata0r^ctdata1r
        X += [ctdata0l, ctdata0r, ctdata1l, ctdata1r, ctdata0l2, ctdata0r2, ctdata1l2, ctdata1r2]
        i += 1
    X = convert_to_binary(X)
    return (X,Y,ks_back)

def make_train_val_data(n, nr, diff,ks_back):
    k = 1
    num =(64*64*k)/256
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
        X += [ctdata0l, ctdata0r,ctdata1l, ctdata1r,ctdata0l2, ctdata0r2 , ctdata1l2, ctdata1r2]
        i += 1
    X = convert_to_binary(X)
    return (X,Y)

