# -*- coding: utf-8 -*-
import copy
import math
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, ticker
from matplotlib.colors import LogNorm


from scipy.spatial import ConvexHull

################################################################################
#
#  Bilevel Optimization Toy Example
#
#  min_{x,w} f(x, w)
#  s.t. x = argmin_x g(x, w)
#
#  f_x = df/dx
#  f_w = df/dw
#  g_x = dg/dx
#  g_w = dg/dw
#
#  xhat = x^{(k)}, where k is steps of inner optimization
#  inner_opt means optimizer of xhat
#  opt means optimizer of [x, w]
#
################################################################################

UPPER=10
LOWER=-10

def bilevel_descent_bome(x, w, x_lr, w_lr, xhat_lr, k, maxIter, eta):
    xs, ws, fs, gs, xhats = [], [], [], [], []

    xhat = copy.deepcopy(x)
    x_opt = torch.optim.SGD([x], lr=x_lr)
    w_opt = torch.optim.SGD([w], lr=w_lr)
    xhat_opt = torch.optim.SGD([xhat], lr=xhat_lr)

    xs.append(x.data.clone().view(-1))
    ws.append(w.data.clone().view(-1))

    n_params_x = x.numel()
    n_params_w = w.numel()
    df = torch.zeros(n_params_x+n_params_w).to(x.device)
    dg = torch.zeros(n_params_x+n_params_w).to(x.device)

    for i in range(maxIter):

        xhat.data = x.data.clone()
        for j in range(k):
            xhat_opt.zero_grad()
            xhat.grad = g_x(xhat, w).data
            xhat_opt.step()
            xhat.data.clamp_(LOWER, UPPER)

        xhats.append(xhat.data.clone().view(-1).cpu())

        g_gap = (g(x, w) - g(xhat, w)).data.clone()
        
        # prepare gradients 
        fx = f_x(x, w)
        fw = f_w(x, w)
        loss, gx, gw_minus_gw_k = g_x_xhat_w(x, xhat, w)

        df[:n_params_x].copy_(fx.view(-1).clone())
        dg[:n_params_x].copy_(gx.view(-1).clone())
        df[n_params_x:].copy_(fw.view(-1).clone())
        dg[n_params_x:].copy_(gw_minus_gw_k.clone())

        norm_dq = dg.norm().pow(2)
        dot = df.dot(dg)

        d = df + F.relu(eta - dot/(norm_dq+1e-8)) * dg

        w_opt.zero_grad()
        x_opt.zero_grad()
        x.grad = d[:n_params_x].data.view(x.shape).clone()
        w.grad = d[n_params_x:].data.view(w.shape).clone()
        w_opt.step()
        x_opt.step()
        x.data.clamp_(LOWER, UPPER)
        w.data.clamp_(LOWER, UPPER)

        xs.append(x.data.clone().view(-1).cpu())
        ws.append(w.data.clone().view(-1).cpu())
        fs.append(f(x,w).data.clone().view(-1).cpu())
        gs.append(g_gap.clone().view(-1).cpu())

    res = { 
        'x'   : torch.vstack(xs),
        'xhat': torch.vstack(xhats),
        'w'   : torch.vstack(ws),
        'f'   : torch.vstack(fs).view(-1),
        'g'   : torch.vstack(gs).view(-1),
    }
    return res 


def optimistic(x, w, x_lr, w_lr, xhat_lr, k, maxIter):
    xs, ws, fs, gs = [], [], [], []

    x_opt = torch.optim.SGD([x], lr=x_lr)
    w_opt = torch.optim.SGD([w], lr=w_lr)

    xs.append(x.data.clone().view(-1))
    ws.append(w.data.clone().view(-1))

    from copy import deepcopy

    for i in range(maxIter):

        if i == 0:
            x_prev = torch.Tensor(xs[-1]*2).requires_grad_()
            w_prev = torch.Tensor(ws[-1]*2).requires_grad_()
        else:
            x_prev = torch.Tensor(xs[-1]).requires_grad_()
            w_prev = torch.Tensor(ws[-1]).requires_grad_()

        g_gap = calculate_g_gap(x, w, xhat_lr, k)

        w.data = (w.data - 2 * w_lr * x.data + w_lr * x_prev.data).clone()
        x.data = (x.data + 2 * x_lr * w.data - x_lr * w_prev.data).clone()

        if (i+1) % k == 0 or i == 0:
            xs.append(x.data.clone().view(-1).cpu())
            ws.append(w.data.clone().view(-1).cpu())
            fs.append(f(x,w).data.clone().view(-1).cpu())
            gs.append(g_gap.clone().view(-1).cpu())

    res = { 
        'x'   : torch.vstack(xs),
        'w'   : torch.vstack(ws),
        'f'   : torch.vstack(fs).view(-1),
        'g'   : torch.vstack(gs).view(-1),
    }
    return res 


def BSG_1(x, w, x_lr, w_lr, xhat_lr, k, maxIter=500):
    xs, ws, fs, gs = [], [], [], []
    xs.append(x.data.clone().view(-1))
    ws.append(w.data.clone().view(-1))

    w_opt = torch.optim.SGD([w], lr=w_lr)
    x_opt = torch.optim.SGD([x], lr=x_lr)

    for i in range(maxIter):

        g0 = g(x,w).data.clone()
        for it in range(k):
            x_opt.zero_grad()
            x.grad = g_x(x, w).data
            x_opt.step()
            x.data.clamp_(LOWER, UPPER)
        g_gap = g0 - g(x,w).data.clone()

        # prepare gradients 
        fx = f_x(x, w)
        fw = f_w(x, w)
        gx = g_x(x, w)
        gw = g_w(x, w)

        w_opt.zero_grad()
        w.grad = (fw - fx.view(-1).dot(gx.view(-1)) / (gx.view(-1).dot(gx.view(-1))+1e-50) * gw).data
        w_opt.step()
        w.data.clamp_(LOWER, UPPER)

        xs.append(x.data.clone().view(-1).cpu())
        ws.append(w.data.clone().view(-1).cpu())
        fs.append(f(x,w).data.clone().view(-1).cpu())
        gs.append(g_gap.clone().view(-1).cpu())

    res = { 
        'x'   : torch.vstack(xs),
        'w'   : torch.vstack(ws),
        'f'   : torch.vstack(fs).view(-1),
        'g'   : torch.vstack(gs).view(-1),
    }
    return res 


def calculate_g_gap(x, w, lr, k):
    x_ = copy.deepcopy(x)
    g0 = g(x, w).data.clone()
    for j in range(k):
        x_.data = x_.data - lr * g_x(x_, w).data.clone()
    gnow = g(x_, w).data.clone()
    return g0 - gnow


def BVFSM(x, w, x_lr, w_lr, xhat_lr, k, maxIter=500, l2_reg=0.1, ln_reg=1):
    xs, ws, fs, gs, gns = [], [], [], [], []

    xs.append(x.data.clone().view(-1))
    ws.append(w.data.clone().view(-1))

    z_l2_reg_coef = l2_reg
    y_l2_reg_coef = l2_reg
    y_ln_reg_coef = ln_reg
    decay_rate = 1.1

    z = copy.deepcopy(x)
    w_opt = torch.optim.SGD([w], lr=w_lr)
    z_opt = torch.optim.SGD([z], lr=xhat_lr)
    x_opt = torch.optim.SGD([x], lr=x_lr)

    for i in range(maxIter):
        reg_decay_rate = 1 / (math.log(decay_rate * (maxIter+1)))

        for it in range(k):
            z_opt.zero_grad()
            loss_z = g(z, w) + z_l2_reg_coef * reg_decay_rate * z.norm(2).pow(2)
            loss_z.backward()
            z_opt.step()
            z.data.clamp_(LOWER, UPPER)

        g_gap = calculate_g_gap(x, w, xhat_lr, k)

        for it in range(k):
            x_opt.zero_grad()
            loss_x = g(x, w)
            loss_z = g(z, w) + z_l2_reg_coef * reg_decay_rate * z.norm(2).pow(2)
            log_barrier = -y_ln_reg_coef * reg_decay_rate * torch.log(loss_x.detach() + loss_z.detach() - loss_x + 1e-4)
            loss_x = f(x, w) + log_barrier + y_l2_reg_coef * reg_decay_rate * x.norm(2).pow(2)
            loss_x.backward()
            x_opt.step()
            x.data.clamp_(LOWER, UPPER)

        w_opt.zero_grad()
        loss_x = g(x, w)
        loss_z = g(z, w) + z_l2_reg_coef * reg_decay_rate * z.norm(2).pow(2)
        log_barrier = -y_ln_reg_coef * reg_decay_rate * torch.log(loss_x.detach() + loss_z - loss_x + 1e-4)
        loss_w = f(x, w) + log_barrier + y_l2_reg_coef * reg_decay_rate * x.norm(2).pow(2)
        loss_w.backward()
        w_opt.step()
        w.data.clamp_(LOWER, UPPER)

        xs.append(x.data.clone().view(-1).cpu())
        ws.append(w.data.clone().view(-1).cpu())
        fs.append(f(x,w).data.clone().view(-1).cpu())
        gs.append(g_gap.clone().view(-1).cpu())

    res = { 
        'x'   : torch.vstack(xs),
        'w'   : torch.vstack(ws),
        'f'   : torch.vstack(fs).view(-1),
        'g'   : torch.vstack(gs).view(-1),
    }
    return res 


def penalty(x, w, x_lr, w_lr, xhat_lr, k, maxIter=500, lmbd_g=0.1, eps=0.1, gamma=0.1):
    xs, ws, fs, gs, gns = [], [], [], [], []
    xs.append(x.data.clone().view(-1))
    ws.append(w.data.clone().view(-1))

    def penalty_gx(x, w, gamma_k, nu_k):
        gx = torch.autograd.grad(g(x,w), x, create_graph=True, allow_unused=True)[0]
        loss = f(x, w) + (nu_k * gx).mean() + lmbd_g * g(x, w) + 0.5 * gamma_k * gx.norm(2).pow(2)
        grad_x = torch.autograd.grad(loss, x, allow_unused=True)[0]
        return grad_x, grad_x.norm().detach().cpu().item()

    def penalty_gw(x, w, gamma_k, nu_k):
        gx = torch.autograd.grad(g(x,w), x, create_graph=True, allow_unused=True)[0]
        loss = f(x, w) + (nu_k * gx).mean() + 0.5 * gamma_k * gx.norm(2).pow(2)
        grad_w = torch.autograd.grad(loss, w, allow_unused=True)[0]
        return grad_w, grad_w.norm().detach().cpu().item(), gx


    lmbd_g = lmbd_g
    eps    = eps
    gamma  = gamma
    nu     = torch.ones_like(x) * 1e-4

    c_gamma = 1.1
    c_eps = 0.9
    c_lmbd = 0.9

    x_opt = torch.optim.SGD([x], lr=x_lr)
    w_opt = torch.optim.SGD([w], lr=w_lr)

    for i in range(maxIter):

        g_gap = calculate_g_gap(x, w, xhat_lr, k)
        for j in range(k):
            x_opt.zero_grad()
            grad, gx_norm = penalty_gx(x, w, gamma, nu)
            x.grad = grad.data
            x_opt.step()
            x.data.clamp_(LOWER, UPPER)

        grad, gw_norm, gx = penalty_gw(x, w, gamma, nu)
        w_opt.zero_grad()
        w.grad = grad.data
        w_opt.step()
        w.data.clamp_(LOWER, UPPER)

        if gx_norm**2 + gw_norm**2 < eps**2:
            gamma *= c_gamma
            eps *= c_eps
            lmbd_g *= c_lmbd
            nu += gx.detach().data * gamma
            w_lr *= 0.9
            x_lr *= 0.9
            w_opt = torch.optim.SGD([w], lr=w_lr)
            x_opt = torch.optim.SGD([x], lr=x_lr)
            print("update gamma and eps", gamma, eps)

        xs.append(x.data.clone().view(-1).cpu())
        ws.append(w.data.clone().view(-1).cpu())
        fs.append(f(x,w).data.clone().view(-1).cpu())
        gs.append(g_gap.clone().view(-1).cpu())
        gns.append((f_x(x,w).data.norm().cpu(), f_w(x,w).data.norm().cpu(), g_x(x,w).data.norm().cpu(), g_w(x,w).data.norm().cpu()))

    fx_, fw_, gx_, gw_ = map(torch.vstack, zip(*gns))
    res = { 
        'x'   : torch.vstack(xs),
        'w'   : torch.vstack(ws),
        'f'   : torch.vstack(fs).view(-1),
        'g'   : torch.vstack(gs).view(-1),
        'gns' : (fx_.view(-1), fw_.view(-1), gx_.view(-1), gw_.view(-1)),
    }
    return res


def control_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def plot3d(F, xl=10, name="none"):
    n = 500
    x = np.linspace(-xl, xl, n)
    y = np.linspace(-xl, xl, n)
    X, Y = np.meshgrid(x, y)

    Xs = torch.Tensor(np.transpose(np.array([list(X.flat), list(Y.flat)]))).double()
    Ys = F(Xs[:,0], Xs[:,1]) 
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)
    Yv = Ys.view(n,n)
    surf = ax.plot_surface(X, Y, Yv.numpy(), cmap=cm.viridis)
    #print(Ys.mean(1).min(), Ys.mean(1).max())

    #ax.set_zticks([-16, -8, 0, 8])
    #ax.set_zlim(-20, 10)
    #ax.set_xticks([-10, 0, 10])
    #ax.set_yticks([-10, 0, 10])
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in ax.zaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    ax.view_init(25)
    plt.tight_layout()
    plt.savefig(f"imgs/3d-{name}.png", dpi=1000)

def plot_fnc(F, xl=5, name="none"):
    n = 200
    x = np.linspace(-xl, xl, n)
    y = np.linspace(-xl, xl, n)
    X, Y = np.meshgrid(x, y)

    Xs = torch.Tensor(np.transpose(np.array([list(X.flat), list(Y.flat)]))).double()
    Ys = F(Xs[:,0], Xs[:,1]) 
    plt.figure()
    c = plt.contour(X, Y, Ys.view(n,n), 40, cmap='RdGy')
    plt.tight_layout()
    plt.savefig(f"imgs/fnc-{name}.png", dpi=1000)
    plt.close()

### original f, g and their partial derivatives

A = torch.ones(1)

def f(x, w): 
    return x * A * w
    #return x.view(1,2).mm(A).mm(w.view(2,1)).view(-1)

def g(x, w):
    return - x * A * w
    #return -x.view(1,2).mm(A).mm(w.view(2,1)).view(-1)

def g_x(x, w):
    grad = torch.autograd.grad(g(x,w), x, allow_unused=True)[0]
    return grad if grad is not None else torch.zeros_like(x)

def g_w(x, w):
    grad = torch.autograd.grad(g(x,w), w, allow_unused=True)[0]
    return grad if grad is not None else torch.zeros_like(w)

def f_x(x, w):
    grad = torch.autograd.grad(f(x,w), x, allow_unused=True)[0]
    return grad if grad is not None else torch.zeros_like(x)

def f_w(x, w):
    grad = torch.autograd.grad(f(x,w), w, allow_unused=True)[0]
    return grad if grad is not None else torch.zeros_like(w)

def g_x_xhat_w(x, xhat, w):
    loss = g(x, w) - g(xhat.detach(), w)
    grad = torch.autograd.grad(loss, [x, w], allow_unused=True)
    return loss.detach().cpu().item(), grad[0], grad[1]


def plot_contour(res, method):
    w = res['w']
    x = res['x']

    n = 200
    x_range = np.linspace(LOWER, UPPER, n)
    w_range = np.linspace(LOWER, UPPER, n)
    X, Y = np.meshgrid(x_range, w_range)

    Xs = torch.Tensor(np.transpose(np.array([list(X.flat), list(Y.flat)]))).double()

    Fs = f(Xs[:,0], Xs[:,1])
    fs = g(Xs[:,0], Xs[:,1])

    plt.figure(figsize=(12, 5))

    m = w.shape[0]
    color_list = np.zeros((m,3))
    color_list[:,0] = 1.
    color_list[:,1] = np.linspace(0, 1, m)

    color_list2 = np.zeros((m-1,3))
    color_list2[:,2] = 1.
    color_list2[:,1] = np.linspace(0, 1, m-1)

    plt.subplot(121)
    c = plt.contour(X, Y, Fs.view(n,n), 100, cmap='RdGy')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.scatter(x, w, color=color_list, s=3, zorder=4)
    plt.scatter(x[0], w[0], s=5, color='r')
    plt.text(x=x[0], y=w[0], s="start", color='r')
    plt.title("Outer f")

    plt.subplot(122)
    c = plt.contour(X, Y, fs.view(n,n), 100, cmap='RdGy')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.scatter(x, w, color=color_list, s=3, zorder=4)
    plt.scatter(x[0], w[0], s=5, color='r')
    plt.text(x=x[0], y=w[0], s="start", color='r')
    plt.title("Inner g")
    plt.savefig(f"imgs/toy_adv_{method}.png")
    plt.close()



### toy coreset
seed = 0
control_seed(seed)

x = torch.FloatTensor([4]).requires_grad_()
w = torch.FloatTensor([4]).requires_grad_()

x_data = x.data.clone()
w_data = w.data.clone()

k = 10

x_lr = xhat_lr = w_lr = 0.05

fn_maps = {
    "ogd": optimistic,
    "bome": bilevel_descent_bome,
    "BSG-1": BSG_1,
    "BVFSM": BVFSM,
    "penalty": penalty,
}

results = {
    'method': {},
    'iter': {},
    'eta': {},
}

method = "bome"
k = 10
maxIter = 200
eta = 0.5
x.data = x_data.clone()
w.data = w_data.clone()
res = fn_maps[method](x, w, x_lr, w_lr, xhat_lr, k, maxIter, eta)
results['method'][method] = res

method = "ogd"
maxIter = 2000
x.data = x_data.clone()
w.data = w_data.clone()
res = fn_maps[method](x, w, x_lr, w_lr, xhat_lr, k, maxIter)
results['method'][method] = res

maxIter = 200
method = "BSG-1"
x.data = x_data.clone()
w.data = w_data.clone()
res = fn_maps[method](x, w, x_lr, w_lr, xhat_lr, k, maxIter)
results['method'][method] = res

method = "BVFSM"
l2reg = 0.1; lnreg = 1.0
x.data = x_data.clone()
w.data = w_data.clone()
res = fn_maps[method](x, w, x_lr, w_lr, xhat_lr, k, maxIter, l2reg, lnreg)
results['method'][method] = res

method = "penalty"
lmbd_g = 1.0; eps = 0.1; gamma = 0.01
x.data = x_data.clone()
w.data = w_data.clone()
res = fn_maps[method](x, w, x_lr, w_lr, xhat_lr, k, maxIter, lmbd_g, eps, gamma)
results['method'][method] = res


method = "bome"
eta = 0.5
for (k, maxIter) in zip([1, 10, 100], [2000, 500, 500]):
    x.data = x_data.clone()
    w.data = w_data.clone()
    res = fn_maps[method](x, w, x_lr, w_lr, xhat_lr, k, maxIter, eta)
    results['iter'][k] = res
    plot_contour(res, method+str(k))

method = "bome"
maxIter=500
k = 10
for eta in [0.1, 0.5, 0.9]:
    x.data = x_data.clone()
    w.data = w_data.clone()
    res = fn_maps[method](x, w, x_lr, w_lr, xhat_lr, k, maxIter, eta)
    results['eta'][eta] = res
    plot_contour(res, method+str(k))

torch.save(results, "result_adv.pt")
