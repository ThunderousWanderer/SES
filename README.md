# SES
Jupyter Notebooks of coding exercise and mini project

```python
import numpy as np
import pandas as pd
import sklearn

def generateDataset(size, ndim=4):
    bias=1
    x = np.linspace(-2*np.pi, +2*np.pi, size)
    timeSeries=4*np.sin(x)+bias
    features=np.zeros((size, ndim))
    labels=np.zeros((size, 1))
    for i in range(size):
        for j in range(ndim):
            features[i,j]=np.random.random_sample()*10
        if np.linalg.norm(features[i])>8:
            labels[i]=1
    return timeSeries, features, labels


timeSeries, features, labels = generateDataset(1000)
```


################## Your Code Here ##########################



################## First question ##########################

#Visualize the data, either using pandas ploting capability 
#or matplotlib.
#TBC

################## Second question ########################

#Preprocess the data so it is suited for ML analysis.
#Specifically it is desired to have Train and Validation
#Datasets with 75/25 weight.


################## Third question ##########################

#Using Principal Analysis Decomposition, reduce the 
#dimensionality of the features to ndim=2. Then apply an
#MLP classifier on the new features and the original labels
#MLP characteristics are 3 hidden layers of 25 cells and 
#sigmoid activation. Evaluate the model


################## Fourth question #########################

#Create a model to predict future steps of the timeSeries
#Please only use scikitLearn methods even though sub-optimal
#Justify the model selection and accuracy
