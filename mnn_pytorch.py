# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 14:49:43 2020

@author: zzc14
"""
import torch
#import numpy as np
from torch import Tensor
import torch.optim as optim
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init
from fast_dawson import *

func_dawson = Dawson1()
func_ddawson = Dawson2()

"""
args:
    _vol_rest: the rest voltage of a neuron
    _vol_th: the fire threshold of a neuron
    _t_ref: the refractory time of a neuoron after it fired
    _conductance: the conductance of a neuron's membrane
    _ratio: num Excitation neurons : num Inhibition neurons 
"""
_vol_rest = 0
_vol_th = 20
_t_ref = 5
_conductance = 0.05
_ratio = 0.8


class Mnn_Linear(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super(Mnn_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input1: Tensor, input2: Tensor):
        out1 = F.linear(input1, self.weight, self.bias)*(1-_ratio)
        out2 = F.linear(torch.pow(input2, 2), torch.pow(self.weight, 2), self.bias)*(1+np.power(_ratio, 2))
        out2 = torch.sqrt(out2)
        return out1, out2

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Mnn_Activate_Mean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mean_in, std_in):
        clone_mean = mean_in.clone().detach().numpy()
        clone_std = std_in.clone().detach().numpy()
        shape = clone_mean.shape
        clone_mean = clone_mean.flatten()
        clone_std = clone_std.flatten()
        up_bound = (_vol_th * _conductance - clone_mean) / clone_std
        low_bound = (_vol_rest * _conductance - clone_mean) / clone_std
        mean_out = 1 / (_t_ref + (func_dawson.int_fast(up_bound) - func_dawson.int_fast(low_bound)) * 2 / _conductance)
        """
        Perform that f = input + (f-input)' 
        This is to make the least op on the input tensor
        """
        mean_out -= clone_mean
        mean_out = torch.from_numpy(mean_out.reshape(shape))
        mean_out = torch.add(mean_in, mean_out)
        ctx.save_for_backward(mean_in, std_in, mean_out)
        return mean_out

    @staticmethod
    def backward(ctx, grad_output):
        mean_in, std_in, mean_out = ctx.saved_tensors
        clone_mean_in = mean_in.clone().detach().numpy()
        clone_std_in = std_in.clone().detach().numpy()
        clone_mean_out = mean_out.clone().detach().numpy()
        shape = clone_mean_in.shape
        clone_mean_in = clone_mean_in.flatten()
        clone_std_in = clone_std_in.flatten()
        clone_mean_out = clone_mean_out.flatten()
        up_bound = (_vol_th * _conductance - clone_mean_in) / clone_std_in
        low_bound = (_vol_rest * _conductance - clone_mean_in) / clone_std_in
        temp_value = func_dawson.dawson1(up_bound) - func_dawson.dawson1(low_bound)
        grad_mean = 2 * np.power(clone_mean_out, 2) * temp_value / (_conductance * clone_std_in)
        temp_value = up_bound * func_dawson.dawson1(up_bound) - low_bound * func_dawson.dawson1(low_bound)
        grad_std = 2 * np.power(clone_mean_out, 2) * temp_value / (clone_std_in * _conductance)

        grad_mean = torch.from_numpy(grad_mean.reshape(shape))
        grad_std = torch.from_numpy(grad_std.reshape(shape))
        grad_mean = torch.mul(grad_output, grad_mean)
        grad_std = torch.mul(grad_output, grad_std)
        return grad_mean, grad_std


class Mnn_Activate_Std(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mean_in, std_in):
        clone_mean = mean_in.clone().detach().numpy()
        clone_std = std_in.clone().detach().numpy()
        shape = clone_mean.shape
        clone_mean = clone_mean.flatten()
        clone_std = clone_std.flatten()
        up_bound = (_vol_th * _conductance - clone_mean) / clone_std
        low_bound = (_vol_rest * _conductance - clone_mean) / clone_std
        mean_out = 1 / (_t_ref + (func_dawson.int_fast(up_bound) - func_dawson.int_fast(low_bound)) * 2 / _conductance)
        temp_out_mean = np.power(mean_out, 3 / 2)
        """
        Perform that f = input + (f-input)' 
        This is to make the least op on the input tensor
        """
        std_out = (8 / np.power(_conductance, 2)) * (func_ddawson.int_fast(up_bound) - func_ddawson.int_fast(low_bound))
        std_out = np.sqrt(std_out) * temp_out_mean
        std_out -= clone_std
        std_out = torch.from_numpy(std_out.reshape(shape))
        std_out = torch.add(std_out, std_in)

        mean_out -= clone_mean
        mean_out = torch.from_numpy(mean_out.reshape(shape))
        mean_out = torch.add(mean_in, mean_out)
        ctx.save_for_backward(mean_in, std_in, mean_out, std_out)
        return std_out

    @staticmethod
    def backward(ctx, grad_output):
        mean_in, std_in, mean_out, std_out = ctx.saved_tensors
        clone_mean_in = mean_in.clone().detach().numpy()
        clone_std_in = std_in.clone().detach().numpy()
        clone_mean_out = mean_out.clone().detach().numpy()
        clone_std_out = std_out.clone().detach().numpy()
        shape = clone_mean_in.shape
        clone_mean_in = clone_mean_in.flatten()
        clone_std_in = clone_std_in.flatten()
        clone_mean_out = clone_mean_out.flatten()
        clone_std_out = clone_std_out.flatten()

        up_bound = (_vol_th * _conductance - clone_mean_in) / clone_std_in
        low_bound = (_vol_rest * _conductance - clone_mean_in) / clone_std_in
        temp_s3 = clone_std_out / (np.power(clone_mean_out, 3 / 2))

        temp_value = func_dawson.dawson1(up_bound) - func_dawson.dawson1(low_bound)
        mean_grad_mean = 2 * np.power(clone_mean_out, 2) * temp_value / (_conductance * clone_std_in)
        temp_value = up_bound * func_dawson.dawson1(up_bound) - low_bound * func_dawson.dawson1(low_bound)
        mean_grad_std = 2 * np.power(clone_mean_out, 2) * temp_value / (clone_std_in * _conductance)

        temp_value = func_ddawson.dawson2(up_bound) - func_ddawson.dawson2(low_bound)
        temp_variable1 = -4*np.power(clone_mean_out, 3/2)/(temp_s3*clone_std_in*np.power(_conductance, 2))
        temp_variable2 = 3/2*temp_s3*np.sqrt(clone_mean_out)
        std_grad_mean = temp_variable1 * temp_value + temp_variable2 * mean_grad_mean

        temp_value = up_bound*func_ddawson.dawson2(up_bound) - low_bound*func_ddawson.dawson2(low_bound)
        std_grad_std = temp_variable1*temp_value + temp_variable2 * mean_grad_std

        std_grad_mean = torch.from_numpy(std_grad_mean.reshape(shape))
        std_grad_std = torch.from_numpy(std_grad_std.reshape(shape))
        std_grad_mean = torch.mul(grad_output, std_grad_mean)
        std_grad_std = torch.mul(grad_output, std_grad_std)
        return std_grad_mean, std_grad_std


def loss_function(pred_mean, pred_std, target_mean, target_std):
    loss1 = F.mse_loss(pred_mean, target_mean)
    loss2 = F.mse_loss(pred_std, target_std)
    return loss1+loss2


if __name__ == "__main__":

    # make sure each run the model has the same inited weights.
    seed = 5
    torch.manual_seed(seed)
    torch.set_default_tensor_type(torch.DoubleTensor)
    mnn_linear1 = Mnn_Linear(10, 10)
    print("===============================")
    print("Weight of mnn_linear1:", mnn_linear1.weight)
    print("===============================")
    input_mean = torch.randn(1, 10)
    input_std = torch.randn(1, 10)

    optimizer = optim.SGD(mnn_linear1.parameters(), lr=1)
    target_mean = torch.ones(1, 10)
    target_std = torch.ones(1, 10)

    for epoch in range(1):
        output_mean, output_std = mnn_linear1.forward(input_mean, input_std)
        activated_mean = Mnn_Activate_Mean.apply(output_mean, output_std)
        activated_std = Mnn_Activate_Std.apply(output_mean, output_std)
        loss = loss_function(activated_mean, activated_std, target_mean, target_std)
        mnn_linear1.zero_grad()
        loss.backward()
        optimizer.step()
        print("===============================")
        print("Weight of mnn_linear1:", mnn_linear1.weight)
        print("===============================")

