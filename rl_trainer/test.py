import numpy as np

a=np.array([1,2,30,5])
b=a.reshape(2,2)
b=np.array(b[np.newaxis,:])
print(b.shape)