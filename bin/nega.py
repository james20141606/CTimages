import numpy as np
import pickle
import h5py
name = np.loadtxt('a.txt',dtype = 'S')

a = {}
for i in range(name.shape[0]):
    f =  open(name[i],'rb')
    a[i] = pickle.load(f)

b = np.array([val for (key,val) in a.items()])
with h5py.File('image_nega','w') as t:
    t.create_dataset('image',data = b)
