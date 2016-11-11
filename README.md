# ffnn_implementation

Build for course assignment at university of Freiburg: [GitHub Page](https://github.com/mllfreiburg/dl_lab_2016)

## Installation

1. Clone repository
3. install requirements
```
pip install -r requirements.txt
```

## Run on MNIST dataset

1. Open python editor (e.x. jupyter notebook)
2. Import run_mnist.py
```
import run_mnist as r
```
3. Train neural network
```
nn = r.train_mnist(0.7,100,100)
```
4. Compute test error and show images that are classified right and wrong
```
r.test_mnist(nn)
```

## Check gradients of neural network

1. Open python editor (e.x. jupyter notebook)
2. Import run_gradient_checking.py
```
import run_gradient_checking
```
