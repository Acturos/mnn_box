updated by 2020年10月7日  
# Implementation of Mnn activation fuction with pytorch  
## Usage
```python
from mnn_pytorch import *
import torch
# Never forget to set the default data type of torch as Double
torch.set_default_tensor_type(torch.DoubleTensor)

input_mean = torch.randn(1, 10)
input_std = torch.randn(1, 10)
# Use customed Linear layer
mnn_linear1 = Mnn_Linear(10, 10)
output_mean, output_std = mnn_linear1.forward(input_mean, input_std)
# Use another variable name to make sure the input will not change
activated_mean = Mnn_Activate_Mean.apply(output_mean, output_std)
activated_std = Mnn_Activate_Std.apply(output_mean, output_std)
# Next step ...
```
# Math details
## Notation 
$g(x) = e^{x^2}\int^{x}_{-\infty}e^{-u^2}du$  
$G(x) = \int^{x}_{0}g(t)dt$  
$h(x) = e^{x^2}\int^{x}_{-\infty}e^{-u^2}[g(t)]^2du$  
$H(x) = \int^{x}_{infty}h(t)dt$  
$T_{ref}$: refractory time  
$V_{rest}$: the rest voltage of a neuron
$V_{th}$: the fire threshold of a neuron
$L$: the conductance of a neuron's membrane
$I^{up} = \frac{V_{th}L - \mu}{\sigma}$: the upbound of integral  
$I^{low} = \frac{V_{rest}L - \mu}{\sigma}$ : the lowbound of integral  


## The forward funtion of Mnn_Activate_Mean($\mu_{in}, \sigma_{in}$):

$\mu_{out} = 1/(T_{ref}+(G(I^{up})-G(I^{low}))*2/L)$

## The backward function of Mnn_Activate_Mean:

$\frac{\partial \mu_{out}}{\partial \mu_{in}} = 2*\mu_{out}^{2}*(g(I^{up})-g(I^{low}))/(L*\sigma_{in})$

$\frac{\partial \mu_{out}}{\partial \sigma_{in}} = 2*\mu_{out}^{2}*(I^{up}g(I^{up})-I^{low}g(I^{low})) / (\sigma_{in}L)$

## The forward function of Mnn_Activate_Std($\mu_{in}, \sigma_{in}$)

$\sigma_{out} = (\frac{5}{L^2}*(H(I^{up}-I^{low})))^{\frac{1}{2}}*\mu_{out}^{\frac{3}{2}}$ 

## The backward function of Mnn_Activate_Std
$s3 = \sigma_{out}/\mu^{\frac{3}{2}}_{out}$  
$\frac{\partial \sigma_{out}}{\partial\mu_{in}} = \frac{-4\mu_{out}^{\frac{3}{2}}(h(I^{up}) - h(I^{low}))}{s3\sigma_{in}L^2} + \frac{3}{2}s3\sqrt{\mu_{out}}\frac{\partial\mu_{out}}{\partial\mu_{in}}$  
$\frac{\partial\sigma_{out}}{\partial\sigma_{in}} = \frac{-4\mu_{out}^{\frac{3}{2}}I^{up}h(I^{up})-I^{low}h(I^{low})}{s3\sigma_{in}L^2}+\frac{3}{2}s3*\sqrt{\mu_{out}}\frac{\partial\mu_{out}}{\partial\sigma_{in}}$

# Fast Dawson integral usage
Here we provide a solution based on asymptotic expansions of these integrals (g(x), h(x), and their integrals G(x) and H(x) from [1](https://www.researchgate.net/publication/7069579_Dynamics_of_moment_neuronal_networks))

The result is the reliable evaluation of these functions over arbitrary input values, and orders-of-magnitude speed improvement over brute force methods.

Usage:
```Python
from fast_dawson import *

ds1 = Dawson1()
ds2 = Dawson2()

x = np.arange(-3,3,0.01)

g = ds1.dawson1(x)  #g(x)

G = ds1.int_fast(x) #integral of g(x)

h = ds2.dawson2(x)	#h(x)

H = ds2.int_fast(x) #integral of h(x)

plt.semilogy(x,g,x,G,x,h,x,H)
plt.xlabel('x')
plt.legend(['g(x)','G(x)','h(x)','H(x)'])
plt.show()
```



***  
# mnn_box
A python package used for reproducing the moment mapping neural network on cpus using `numpy` functionality.

Moment mapping nn is a spiking neural nn which can implement feedforward process based  concentrating on 
some statistics of spike pattern. Inspired from deep learning, we aim to construct a complete mnn framework and 
compare the effects with mlp based on `pytorch`.

> Feng, Jianfeng, Yingchun Deng, and Enrico Rossoni. "Dynamics of moment neuronal networks." Physical Review E 73.4 (2006): 041906.

>Lu, Wenlian, Enrico Rossoni, and Jianfeng Feng. "On a Gaussian neuronal field model." NeuroImage 52.3 (2010): 913-933.

## Requirement
- python3.7
- mnn_box package

# Getting started
we choose the mnist classification task to test the learning power. It's analogous 
to the `tensorflow`:
+ import mnn_box as mnn
+ build the computational graph
+ run the nn and test the accuracy

# results
The MNIST database of handwritten digits, has a training set of 60,000 examples,
and a test set of 10,000 examples. we test the 1 hidden layer mnn's performance, and the following 
picture show the loss and accuracy during training and predicting.
![loss](https://github.com/lepodl/mnn_box/raw/master/image/loss.png)
![accuracy](https://github.com/lepodl/mnn_box/raw/master/image/accuracy.png)

