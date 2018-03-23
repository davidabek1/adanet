# ADANET

AdaNet: Adaptive Structural Learning of Artificial Neural Networks

Reference: Cortes, C., Gonzalvo, X., Kuznetsov, V., Mohri, M. & Yang, S.. (2017). AdaNet: Adaptive Structural Learning of Artificial Neural Networks. Proceedings of the 34th International Conference on Machine Learning, in PMLR 70:874-883

This Python project is aimed to implement an API of AdaNet, using algorithm based on the article: http://proceedings.mlr.press/v70/cortes17a.html
In short this model is building from scratch a neural network according to the data complexity it fits,  
This is why it named as adaptive model. For this implementation the problem at hand is always a binary classification.  
During fit operation it will build the hidden layers and number of neurons in each layer . 
The decision if to go deeper (add hidden layer) or to go wider (add neuron to existing layer),  
Or update an existing neuron weight is done in a closed form of calculations  
(By using Banach space duality) shown in the article.  
Lastly it will optimize the weight of the best neuron (added or existing), update parameters and iterate.  
The article talks about several variants of AdaNet, this is the AdaNet.CVX implementation,  
Explained on Appendix C - that solves a convex sub problem in each step in a closed form. 
Further detailed explanations of this variant is shown in a previous version of the article [v.1]:  
All versions: https://arxiv.org/abs/1607.01097  
v.1: https://arxiv.org/abs/1607.01097v1

or directly here on GitHub: [AdaNet-v.1](https://github.com/davidabek1/adanet/blob/master/AdaNet-%20Adaptive%20Structural%20Learning%20of%20Artificial%20Neural%20Networks___1607.01097v1.pdf)


## Article Abstract

We present a new theoretical framework for analyzing and learning artificial neural
networks. Our approach simultaneously and adaptively learns both the structure
of the network as well as its weights. The methodology is based upon and accompanied
by strong data-dependent theoretical learning guarantees. We present
some preliminary results to show that the final network architecture adapts to the
complexity of a given problem.

## Getting Started

Clone the full project folder, it will let you be ready and up and running the test and API files.

### Prerequisites

The Python code is using this packages, you need to install it in the environment for it to be imported:

```
import csv
import itertools
import collections
import time
import os
import numpy
import matplotlib
import scipy

import sklearn
import tensorflow
```

### Installing

Put the Python code files in the same root folder of the project: 
AdaNet_CVX.py - API main file  
test_adanet.py - test code to run several datasets and results of the model  
AdaNet_CIFAR_10_feature_extraction.py - CIFAR-10 dataset and features extraction  
twospirals.py - toy dataset creation as part of testing 

This next lines of code will enable the use of the API  

```
from AdaNet_CVX import AdaNetCVX             # first import the module class 
adanet_clf = AdaNetCVX(**params_adanet)  # Second initialize the class with relevant parameters, all parameters have default values.
adanet_clf.fit(X_train, y_train)  # Third, ones can start using the fit and predict standard methods, just like with sklearn package.
adanet_clf.predict(X_test)
adanet_clf.adaParams  # After fit operation the classifier provides attributes, this parameters of the model
adanet_clf.history    # After fit operation the classifier provides attributes, this history of the model fitting
adanet_clf.predict(X_test)
```

## Running the tests

Running test_adanet.py will create 4 csv files: 3 of models (AdaNet, FFNN, LR) evaluation hyper-parameters, 
and a csv having the test run comparing all of them, with t-test significance.

## Built With

* [VScode](https://code.visualstudio.com/) - Python 3.6 coding
* [Tensorflow](https://www.tensorflow.org/) - model implementation, article based

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

I use VScode builtin Git Source control with remote to this github adanet repository 

## Authors

* **David Abekasis**


## License

This project has no license

## Acknowledgments

* I couldn't get this far without the MATLAB code of: https://github.com/lw394/adanet

