# billboard100_predict
Artificial Neural Network (ANN) built from scratch to predict deciles of popular songs based on their chart histories. Uses one hidden layer and trained using backpropagation and gradient descent.

## Getting Started

These instructions will allow you to build the dataset, train the network and use it to predict the Billboard charts locally. Start by cloning the repo - `git clone https://github.com/rsriv/billboard100_predict.git`

### Prerequisites

The main dependencies are NumPy and the Unofficial Python Billboard API. See instructions below.

```
pip install numpy
```

```
pip install billboard.py
```



## Running Locally

Takes command line options indicated by a hyphen (-) as input to do one or more of the functions in the table below. See examples for details on how to use options. Note: order does not matter when inputting options.
>-v &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; verbose  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  output raw prediction after each training iteration

>-t &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  train &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;       train neural network and write parameters to file

>-p &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; predict &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    predict current chart's performance next week and this week's chart (to test accuracy)

>-d &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; get data &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   download up-to-date Billboard Chart history data

>-help &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; help &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;       display this

### Example Syntax 1

Below is an example of how to fetch and save an updated dataset, train the network and output predictions in verbose mode. (Recommended after installation)

```
python billboard_predict.py -dtpv
```

### Example Syntax 2

Below is an example of how to get the help menu shown above.

```
python billboard_predict.py -help
```

## Next Steps

* Create large feature sets for more accurate predictions - including a rank for an artist's popularity/chartablility
* Implement more advanced neural network architectures

## Built With

* [billboard.py](https://github.com/guoguo12/billboard-charts) - API for collecting chart data
* [NumPy](http://www.numpy.org) - Framework for facilitating advanced computations

##Notes
### Training Algorithm - How it Works

Feedforward feature set through 3 layers (single hidden layer) with nodes using a sigmoid activation function to compute a prediction h. Compute cost then backpropogate to get gradients for each parameter Theta. Perform gradient descent by updating parameters using gradients from backpropagation. Repeat until convergence is roughly achieved.

###Experimental Notes
Through 100 training iterations, achieved ~16% top-100 accuracy and 60% top-10
accuracy.

Through 1000 training iterations, achieved ~25 top-100 accuracy and 60% top-10
accuracy.

Through 10000 training iterations, achieved ~27 top-100 accuracy and 60% top-10
accuracy.

Through 20000 training iterations, achieved ~26 top-100 accuracy and 50% top-10
accuracy.
