import speck as sp
import numpy as np
from os import urandom
from tensorflow.keras.models import load_model
wdir = './'
n = 10**6
rounds = [5,6]
for i in rounds:
    model = load_model('best'+str(i)+'depth5.h5')
    x,y= sp.make_test_data(n, nr=i, diff=(0x0040,0))
    scores = model.evaluate(x,y)
    np.save(wdir + 'evaluate_scores_r'+str(i)+'.npy',scores);