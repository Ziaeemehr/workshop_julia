import h5py
f = h5py.File('data/test1.h5', 'r')
print(f.keys())
dset = f['mygroup/A']
print(dset.shape, type(dset))
f.close()


f = h5py.File("data/mydata.h5", "r")
print(f.keys())
A = f["A"]
B = f["B"]
print(type(A), type(B))
print(A.shape, B.shape)
f.close()