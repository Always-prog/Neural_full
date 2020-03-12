# Neural_full
__init__:
Network([array your network])
commands:
trening(array inputs, must outputs, trening number),  
think(array inputs),  
GetLayer(index),  
GetWeight(index),  
SaveWeight(want name your weight),  
LoadWeight(name your saves),  
<code>  
from UpPyReLu import *
import numpy as np
MyNet = Network([2,10,50,50,10,2])

inputs = np.full((2,2),0.0)
outputs = np.full((2,2),0.0)

for i in range(99999):
    if i / 2 == 0:
        inputs = np.full((2,2),0.9)
        outputs = np.full((2,2),0.1)
    else:
        inputs = np.full((2,2),0.1)
        outputs = np.full((2,2),0.9)
    MyNet.trening(inputs,outputs,0.3)

inputs = np.full((2,2),0.1)
print("inputs = 0.1 "+MyNet.think(inputs))


inputs = np.full((2,2),0.9)
print("inputs = 0.9 "+MyNet.think(inputs))
</code>
