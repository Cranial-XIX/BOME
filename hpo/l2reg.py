import argparse
import copy
import hypergrad as hg # hypergrad package
import math
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_20newsgroups_vectorized

from torchvision import datasets


################################################################################
#
#  Bilevel Optimization Toy Example
#
#  min_{x,w} f(x, w)
#  s.t. x = argmin_x g(x, w)
#
#  here: f(x, w) is on valset
#        g(x, w) is on trainset
#
#  f_x = df/dx
#  f_w = df/dw
#  g_x = dg/dx
#  g_w = dg/dw
#
################################################################################


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_data', action='store_true',
            default=False, help='whether to create data')
    parser.add_argument('--pretrain', action='store_true',
            default=False, help='whether to create data')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--iterations', type=int, default=10, help='T')
    parser.add_argument('--data_path', default='./data', help='where to save data')
    parser.add_argument('--model_path', default='./save_l2reg', help='where to save model')
    parser.add_argument('--x_lr', type=float, default=100)
    parser.add_argument('--xhat_lr', type=float, default=100)
    parser.add_argument('--w_lr', type=float, default=1000)

    parser.add_argument('--w_momentum', type=float, default=0.9)
    parser.add_argument('--x_momentum', type=float, default=0.9)

    parser.add_argument('--K', type=int, default=10, help='k')

    parser.add_argument('--u1', type=float, default=1.0)
    parser.add_argument('--BVFSM_decay', type=str, default='log', choices=['log', 'power2'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--alg', type=str, default='BOME', choices=[
        'BOME', 'BSG_1', 'penalty', 'AID_CG', 'AID_FP', 'ITD', 'BVFSM', 'baseline', 'VRBO', 'reverse', 'stocBiO', 'MRBO']
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    return args


def get_data(args):

    def from_sparse(x):
        x = x.tocoo()
        values = x.data
        indices = np.vstack((x.row, x.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = x.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    val_size = 0.5
    train_x, train_y = fetch_20newsgroups_vectorized(subset='train',
                                                     return_X_y=True,
                                                     data_home=args.data_path,
                                                     download_if_missing=True)

    test_x, test_y = fetch_20newsgroups_vectorized(subset='test',
                                                   return_X_y=True,
                                                   data_home=args.data_path,
                                                   download_if_missing=True)

    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, stratify=train_y, test_size=val_size)
    test_x, teval_x, test_y, teval_y = train_test_split(test_x, test_y, stratify=test_y, test_size=0.5)

    train_x, val_x, test_x, teval_x = map(from_sparse, [train_x, val_x, test_x, teval_x])
    train_y, val_y, test_y, teval_y = map(torch.LongTensor, [train_y, val_y, test_y, teval_y])

    print(train_y.shape[0], val_y.shape[0], test_y.shape[0], teval_y.shape[0])
    return (train_x, train_y), (val_x, val_y), (test_x, test_y), (teval_x, teval_y)


### original f, g, and gradients

def f(x, w, dataset):
    data_x, data_y = dataset
    y = data_x.mm(x)
    loss = F.cross_entropy(y, data_y, reduction='mean')
    return loss

def g(x, w, dataset):
    data_x, data_y = dataset
    y = data_x.mm(x)
    loss = F.cross_entropy(y, data_y, reduction='mean')
    reg_loss = 0.5 * (x.pow(2) * w.view(-1, 1).exp()).mean() # l2 reg loss
    return loss + reg_loss

def g_x(x, w, dataset, retain_graph=False, create_graph=False):
    loss = g(x, w, dataset)
    grad = torch.autograd.grad(loss, x,
                               retain_graph=retain_graph,
                               create_graph=create_graph)[0]
    return grad

def g_w(x, w, dataset, retain_graph=False, create_graph=False):
    loss = g(x, w, dataset)
    grad = torch.autograd.grad(loss, w,
                               retain_graph=retain_graph,
                               create_graph=create_graph)[0]
    return grad

def g_x_xhat_w(x, xhat, w, dataset, retain_graph=False, create_graph=False):
    loss = g(x, w, dataset) - g(xhat.detach(), w, dataset)
    grad = torch.autograd.grad(loss, [x, w],
                               retain_graph=retain_graph,
                               create_graph=create_graph)
    return loss, grad[0], grad[1]

def g_x_xhat_w_bo(x, xhat, w, dataset, retain_graph=False, create_graph=False):
    loss = g(x, w, dataset) - g(xhat, w, dataset)
    grad = torch.autograd.grad(loss, [x, xhat, w],
                               retain_graph=retain_graph,
                               create_graph=create_graph)
    return grad[0], grad[1], grad[2]

def f_x(x, w, dataset, retain_graph=False, create_graph=False):
    loss = f(x, w, dataset)
    grad = torch.autograd.grad(loss, x,
                               retain_graph=retain_graph,
                               create_graph=create_graph)[0]
    return grad

### Define evaluation metric

def evaluate(x, w, testset):
    with torch.no_grad():
        test_x, test_y = testset  
        y = test_x.mm(x)
        loss = F.cross_entropy(y, test_y).detach().item()
        acc = (y.argmax(-1).eq(test_y).sum() / test_y.shape[0]).detach().cpu().item()
    return loss, acc


def baseline(args, x, w, trainset, valset, testset, tevalset): # no regularization
    opt = torch.optim.SGD([x], lr=args.x_lr, momentum=args.x_momentum)
    n = trainset[0].shape[0]

    best_teval_loss = np.inf
    best_config = None
    for epoch in range(args.epochs):
        opt.zero_grad()
        x.grad = f_x(x, None, trainset).data
        opt.step()
        test_loss, test_acc = evaluate(x, None, testset)
        teval_loss, teval_acc = evaluate(x, None, tevalset)
        if teval_loss < best_teval_loss:
            best_teval_loss = teval_loss
            best_config = (test_loss, test_acc, x.data.clone())
        #print(f"[baseline] epoch {epoch:5d} test loss {test_loss:10.4f} test acc {test_acc:10.4f}")
    print(f"[baseline] best test loss {best_config[0]} best test acc {best_config[1]}")
    return best_config


def BOME(args, x, w, trainset, valset, testset, tevalset):
    xhat = copy.deepcopy(x)

    total_time = 0.0
    n = trainset[0].shape[0]
    stats = []

    outer_opt = torch.optim.SGD(
        [
            {'params': [x], 'lr': args.x_lr},
            {'params': [w], 'lr': args.w_lr},
        ], momentum=args.w_momentum)
    inner_opt = torch.optim.SGD([xhat], lr=args.xhat_lr, momentum=args.x_momentum)

    n_params_w = w.numel()
    zz = torch.zeros(n_params_w).to(x.device)

    for epoch in range(args.epochs):

        xhat.data = x.data.clone()
        t0 = time.time()
        for it in range(args.iterations):
            inner_opt.zero_grad()
            xhat.grad = g_x(xhat, w, trainset)
            inner_opt.step()

        # prepare gradients 
        fx = f_x(x, w, valset)
        loss, gx, gw_minus_gw_k = g_x_xhat_w(x, xhat, w, trainset)

        df = torch.cat([fx.view(-1), zz])
        dg = torch.cat([gx.view(-1), gw_minus_gw_k.view(-1)])
        norm_dq = dg.norm().pow(2)
        dot = df.dot(dg)
        lmbd = F.relu((args.u1 * loss - dot)/(norm_dq + 1e-8))

        outer_opt.zero_grad()
        x.grad = fx + lmbd * gx
        w.grad = lmbd * gw_minus_gw_k
        outer_opt.step()
        t1 = time.time()
        total_time += t1 - t0
        #print(x.grad.norm().item(), w.grad.norm().item())

        test_loss, test_acc = evaluate(x, w, testset)
        teval_loss, teval_acc = evaluate(x, w, tevalset)
        stats.append((total_time, test_loss, test_acc, teval_loss, teval_acc))
        print(f"[info] epoch {epoch:5d} te loss {test_loss:10.4f} te acc {test_acc:10.4f} teval loss {teval_loss:10.4f} teval acc {teval_acc:10.4f} time {total_time:8.2f}")
    return stats


def BSG_1(args, x, w, trainset, valset, testset, tevalset):
    total_time = 0.0
    n = trainset[0].shape[0]
    stats = []
    outer_opt = torch.optim.SGD([w], lr=args.w_lr, momentum=args.w_momentum)
    inner_opt = torch.optim.SGD([x], lr=args.x_lr, momentum=args.x_momentum)

    for epoch in range(args.epochs):

        t0 = time.time()
        for it in range(args.iterations):
            inner_opt.zero_grad()
            x.grad = g_x(x, w, trainset).data
            inner_opt.step()

        # prepare gradients 
        fx = f_x(x, w, valset)
        gx = g_x(x, w, trainset)
        gw = g_w(x, w, trainset)

        outer_opt.zero_grad()
        w.grad = (-fx.view(-1).dot(gx.view(-1)) / (gx.view(-1).dot(gx.view(-1))+1e-4) * gw).data
        outer_opt.step()
        t1 = time.time()
        total_time += t1 - t0

        test_loss, test_acc = evaluate(x, w, testset)
        teval_loss, teval_acc = evaluate(x, w, tevalset)
        stats.append((total_time, test_loss, test_acc, teval_loss, teval_acc))
        print(f"[info] epoch {epoch:5d} te loss {test_loss:10.4f} te acc {test_acc:4.2f} teval loss {teval_loss:10.4f} teval acc {teval_acc:10.4f} time {total_time:8.2f}")
    return stats


def reverse(args, x, w, trainset, valset, testset, tevalset):
    return implicit(args, x, w, trainset, valset, testset, tevalset, opt='reverse')

def AID_CG(args, x, w, trainset, valset, testset, tevalset):
    return implicit(args, x, w, trainset, valset, testset, tevalset, opt='AID_CG')

def AID_FP(args, x, w, trainset, valset, testset, tevalset):
    return implicit(args, x, w, trainset, valset, testset, tevalset, opt='AID_FP')

def implicit(args, x, w, trainset, valset, testset, tevalset, opt):
    outer_loss = lambda x, w: f(x[0], w[0], valset)
    inner_loss = lambda x, w, d: g(x[0], w[0], d)

    #inner_opt = hg.GradientDescent(inner_loss, args.x_lr, data_or_iter=trainset)
    inner_opt = hg.Momentum(inner_loss, args.x_lr, args.x_momentum, data_or_iter=trainset)
    inner_opt_cg = hg.GradientDescent(inner_loss, 1., data_or_iter=trainset)
    outer_opt = torch.optim.SGD([w], lr=args.w_lr, momentum=args.w_momentum)

    total_time = 0.0
    stats = []

    for epoch in range(args.epochs):

        momentum = torch.zeros_like(x) 
        t0 = time.time()
        x_history = [[x, momentum]]
        for it in range(args.iterations):
            x_history.append(inner_opt(x_history[-1], [w], create_graph=False))

        outer_opt.zero_grad()
        if args.alg == 'reverse':
            hg.reverse(x_history[-args.K-1:], [w], [inner_opt]*args.K, outer_loss)
        elif opt == 'AID_CG':
            hg.CG([x_history[-1][0]], [w], args.K, inner_opt_cg, outer_loss, stochastic=False, set_grad=True)
        elif opt == 'AID_FP':
            hg.fixed_point(x_history[-1], [w], args.K, inner_opt, outer_loss, stochastic=False, set_grad=True)
        else:
            raise NotImplementedError
        outer_opt.step()
        t1 = time.time()
        total_time += t1 - t0

        x.data = x_history[-1][0].data.clone()

        test_loss, test_acc = evaluate(x, w, testset)
        teval_loss, teval_acc = evaluate(x, w, tevalset)
        stats.append((total_time, test_loss, test_acc, teval_loss, teval_acc))
        print(f"[info] epoch {epoch:5d} te loss {test_loss:10.4f} te acc {test_acc:4.2f} teval loss {teval_loss:10.4f} teval acc {teval_acc:10.4f} time {total_time:8.2f}")
    return stats


def ITD(args, x, w, trainset, valset, testset, tevalset):
    outer_loss = lambda x, w: f(x[0], w[0], valset)
    inner_loss = lambda x, w, d: g(x[0], w[0], d)

    #inner_opt = hg.GradientDescent(inner_loss, args.x_lr, data_or_iter=trainset)
    inner_opt = hg.Momentum(inner_loss, args.x_lr, args.x_momentum, data_or_iter=trainset)
    outer_opt = torch.optim.SGD([w], lr=args.w_lr, momentum=args.w_momentum)

    total_time = 0.0
    stats = []

    for epoch in range(args.epochs):

        momentum = torch.zeros_like(x) 
        t0 = time.time()
        x_history = [[x, momentum]]
        for it in range(args.iterations):
            x_history.append(inner_opt(x_history[-1], [w], create_graph=True))

        outer_opt.zero_grad()
        loss = outer_loss([x_history[-1][0]], [w])
        grad = torch.autograd.grad(loss, w)[0]
        w.grad = grad.data.clone()
        outer_opt.step()
        t1 = time.time()
        total_time += t1 - t0

        x.data = x_history[-1][0].data.clone()

        test_loss, test_acc = evaluate(x, w, testset)
        teval_loss, teval_acc = evaluate(x, w, tevalset)
        stats.append((total_time, test_loss, test_acc, teval_loss, teval_acc))
        print(f"[info] epoch {epoch:5d} te loss {test_loss:10.4f} te acc {test_acc:4.2f} teval loss {teval_loss:10.4f} teval acc {teval_acc:10.4f} time {total_time:8.2f}")
    return stats



def BVFSM(args, x, w, trainset, valset, testset, tevalset):
    z = copy.deepcopy(x)
    w_opt = torch.optim.SGD([w], lr=args.w_lr, momentum=args.w_momentum)
    z_opt = torch.optim.SGD([z], lr=args.xhat_lr, momentum=args.x_momentum)
    x_opt = torch.optim.SGD([x], lr=args.x_lr, momentum=args.x_momentum)

    n = trainset[0].shape[0]
    total_time = 0.0
    stats = []

    z_l2_reg_coef = 1e-6
    y_l2_reg_coef = 1e-6
    y_ln_reg_coef = 0.01

    decay_rate = 1.1

    for epoch in range(args.epochs):

        if args.BVFSM_decay == 'log':
            reg_decay_rate = 1 / (math.log(decay_rate * (epoch+1)))
        elif args.BVFSM_decay == 'power2':
            reg_decay_rate = 1 / ((epoch+1) ** decay_rate)

        t0 = time.time()
        for it in range(args.iterations):
            z_opt.zero_grad()
            loss_z = g(z, w, trainset) + z_l2_reg_coef * reg_decay_rate * z.norm(2).pow(2)
            loss_z.backward()
            z_opt.step()

        for it in range(args.iterations):
            x_opt.zero_grad()
            loss_x = g(x, w, trainset)
            loss_z = g(z, w, trainset) + z_l2_reg_coef * reg_decay_rate * z.norm(2).pow(2)
            log_barrier = -y_ln_reg_coef * reg_decay_rate * torch.log(loss_x.detach() + 1e-4 + loss_z.detach() - loss_x)
            loss_x = f(x, w, valset) + log_barrier + y_l2_reg_coef * reg_decay_rate * x.norm(2).pow(2)
            loss_x.backward()
            x_opt.step()

        # prepare gradients 
        w_opt.zero_grad()
        loss_x = g(x, w, trainset)
        loss_z = g(z, w, trainset) + z_l2_reg_coef * reg_decay_rate * z.norm(2).pow(2)
        log_barrier = -y_ln_reg_coef * reg_decay_rate * torch.log(loss_x.detach() + 1e-4 + loss_z - loss_x)
        loss_w = f(x, w, valset) + log_barrier + y_l2_reg_coef * reg_decay_rate * x.norm(2).pow(2)
        loss_w.backward()
        w_opt.step()

        t1 = time.time()
        total_time += t1 - t0

        test_loss, test_acc = evaluate(x, w, testset)
        teval_loss, teval_acc = evaluate(x, w, tevalset)
        stats.append((total_time, test_loss, test_acc, teval_loss, teval_acc))
        print(f"[info] epoch {epoch:5d} te loss {test_loss:10.4f} te acc {test_acc:4.2f} teval loss {teval_loss:10.4f} teval acc {teval_acc:10.4f} time {total_time:8.2f}")
    return stats


def penalty(args, x, w, trainset, valset, testset, tevalset):
    total_time = 0.0
    n = trainset[0].shape[0]
    stats = []

    def penalty_gx(x, w, trainset, valset, gamma_k, nu_k):
        gx = torch.autograd.grad(g(x, w, trainset), x, create_graph=True, allow_unused=True)[0]
        loss = f(x, w, valset) + (nu_k * gx).sum() + lmbd_g * g(x, w, trainset) + 0.5 * gamma_k * gx.norm(2).pow(2)
        grad_x = torch.autograd.grad(loss, x, allow_unused=True)[0]
        return grad_x, grad_x.norm().detach().cpu().item()

    def penalty_gw(x, w, trainset, valset, gamma_k, nu_k):
        gx = torch.autograd.grad(g(x, w, trainset), x, create_graph=True, allow_unused=True)[0]
        loss = f(x, w, valset) + (nu_k * gx).sum() + 0.5 * gamma_k * gx.norm(2).pow(2)
        grad_w = torch.autograd.grad(loss, w, allow_unused=True)[0]
        return grad_w, grad_w.norm().detach().cpu().item(), gx


    lmbd_g = 1e-2
    eps    = 1e-2
    gamma  = 1e-2
    nu     = torch.ones_like(x) * 1e-4

    c_gamma = 1.1
    c_eps = 0.9
    c_lmbd = 0.9

    w_lr = args.w_lr
    x_lr = args.x_lr

    outer_opt = torch.optim.SGD([w], lr=w_lr, momentum=args.w_momentum)
    inner_opt = torch.optim.SGD([x], lr=x_lr, momentum=args.x_momentum)

    for epoch in range(args.epochs):

        t0 = time.time()
        for it in range(args.iterations):
            inner_opt.zero_grad()
            grad, gx_norm = penalty_gx(x, w, trainset, valset, gamma, nu)
            x.grad = grad.data
            inner_opt.step()

        # prepare gradients 
        outer_opt.zero_grad()
        grad, gw_norm, gx = penalty_gw(x, w, trainset, valset, gamma, nu)
        w.grad = grad.data
        outer_opt.step()

        # update gamma and eps if necessary
        if gx_norm**2 + gw_norm**2 < eps**2:
            gamma *= c_gamma
            eps *= c_eps
            lmbd_g *= c_lmbd
            nu += gx.detach().data * gamma
            w_lr *= 0.9
            x_lr *= 0.9
            outer_opt = torch.optim.SGD([w], lr=w_lr, momentum=0.9)
            inner_opt = torch.optim.SGD([x], lr=x_lr)
            print("update gamma and eps", gamma, eps)

        t1 = time.time()
        total_time += t1 - t0

        test_loss, test_acc = evaluate(x, w, testset)
        teval_loss, teval_acc = evaluate(x, w, tevalset)
        stats.append((total_time, test_loss, test_acc, teval_loss, teval_acc))
        print(f"[info] epoch {epoch:5d} te loss {test_loss:10.4f} te acc {test_acc:4.2f} teval loss {teval_loss:10.4f} teval acc {teval_acc:10.4f} time {total_time:8.2f}")
    return stats


###############################################################################
#
# Below are stochastic methods
#
###############################################################################

def stocBiO(args, x, w, trainset, valset, testset, tevalset):
    train_batch_size = 1000
    val_batch_size = 1000

    train_dense = trainset[0].to_dense()
    val_dense = valset[0].to_dense()

    train_x_list = list(torch.split(train_dense, train_batch_size))
    train_y_list = list(torch.split(trainset[1], train_batch_size))

    val_x_list = list(torch.split(val_dense, val_batch_size))
    val_y_list = list(torch.split(valset[1], val_batch_size))

    train_x_list = [a.to_sparse() for a in train_x_list]
    val_x_list   = [a.to_sparse() for a in val_x_list]

    n_train = len(train_x_list)
    n_val = len(val_x_list)

    def stocbio(x, w, idx0, idx1, idx2):
        eta = 100.0

        fx = f_x(x, w, (val_x_list[idx0], val_y_list[idx0])) # Fy_gradient
        v_0 = fx.view(-1, 1).detach()

        # Hessian
        z_list = []
        gx = g_x(x, w, (train_x_list[idx1], train_y_list[idx1]), retain_graph=True, create_graph=True)
        gx_ = x.view(-1) - eta * gx.view(-1) # G_gradient
        
        for _ in range(10): # number of Hessian Q steps
            Jacobian = torch.matmul(gx_, v_0)
            v_new = torch.autograd.grad(Jacobian, x, retain_graph=True)[0]
            v_0 = v_new.view(-1, 1).detach()
            z_list.append(v_0)

        v_Q = eta * v_0 + torch.sum(torch.stack(z_list), dim=0)

        # Gyx_gradient
        gx = g_x(x, w, (train_x_list[idx2], train_y_list[idx2]), retain_graph=True, create_graph=True).view(-1)
        gxw = torch.autograd.grad(torch.matmul(gx, v_Q.detach()), w, retain_graph=True)[0]
        return -gxw

    total_time = 0.0
    n = trainset[0].shape[0]
    stats = []

    inner_opt = torch.optim.SGD([x], lr=args.x_lr, momentum=args.x_momentum)
    outer_opt = torch.optim.SGD([w], lr=args.w_lr, momentum=args.w_momentum)

    w_old = copy.deepcopy(w)

    for epoch in range(args.epochs):

        idx0 = np.random.randint(n_val)
        idx1 = np.random.randint(n_train)
        idx2 = np.random.randint(n_train)
        idx3 = np.random.randint(n_train, size=(args.iterations))

        t0 = time.time()
        for t in range(args.iterations):
            idx = idx3[t]
            gx = g_x(x, w, (train_x_list[idx], train_y_list[idx]))
            inner_opt.zero_grad()
            x.grad = gx
            inner_opt.step()

        grad_w = stocbio(x, w, idx0, idx1, idx2)

        outer_opt.zero_grad()
        w.grad = grad_w
        #import pdb; pdb.set_trace()
        outer_opt.step()
        t1 = time.time()
        total_time += t1 - t0

        test_loss, test_acc = evaluate(x, w, testset)
        teval_loss, teval_acc = evaluate(x, w, tevalset)
        stats.append((total_time, test_loss, test_acc, teval_loss, teval_acc))
        print(f"[info] epoch {epoch:5d} te loss {test_loss:10.4f} te acc {test_acc:4.2f} teval loss {teval_loss:10.4f} teval acc {teval_acc:10.4f} time {total_time:8.2f}")
    return stats


def VRBO(args, x, w, trainset, valset, testset, tevalset):

    train_batch_size = 1000
    val_batch_size = 1000

    train_dense = trainset[0].to_dense()
    val_dense = valset[0].to_dense()

    train_x_list = list(torch.split(train_dense, train_batch_size))
    train_y_list = list(torch.split(trainset[1], train_batch_size))

    val_x_list = list(torch.split(val_dense, val_batch_size))
    val_y_list = list(torch.split(valset[1], val_batch_size))

    train_x_list = [a.to_sparse() for a in train_x_list]
    val_x_list   = [a.to_sparse() for a in val_x_list]

    n_train = len(train_x_list)
    n_val = len(val_x_list)

    def stocbio(x, w, idx0, idx1, idx2):
        eta = 100.0

        fx = f_x(x, w, (val_x_list[idx0], val_y_list[idx0])) # Fy_gradient
        v_0 = fx.view(-1, 1).detach()

        # Hessian
        z_list = []
        gx = g_x(x, w, (train_x_list[idx1], train_y_list[idx1]), retain_graph=True, create_graph=True)
        gx_ = x.view(-1) - eta * gx.view(-1) # G_gradient
        
        for _ in range(10): # number of Hessian Q steps
            Jacobian = torch.matmul(gx_, v_0)
            v_new = torch.autograd.grad(Jacobian, x, retain_graph=True)[0]
            v_0 = v_new.view(-1, 1).detach()
            z_list.append(v_0)
        v_Q = eta * v_0 + torch.sum(torch.stack(z_list), dim=0)

        # Gyx_gradient
        gx = g_x(x, w, (train_x_list[idx2], train_y_list[idx2]), retain_graph=True, create_graph=True).view(-1)
        gxw = torch.autograd.grad(torch.matmul(gx, v_Q.detach()), w, retain_graph=True)[0]
        return -gxw

    def vrbo(x, w, w_old, grad_x, grad_w): 

        idx0 = np.random.randint(n_val)
        idx1 = np.random.randint(n_train)
        idx2 = np.random.randint(n_train)

        gx = g_x(x, w, (train_x_list[idx1], train_y_list[idx1]), retain_graph=True)
        gx_old = g_x(x, w_old, (train_x_list[idx1], train_y_list[idx1]), retain_graph=True)

        dw = stocbio(x, w, idx0, idx1, idx2)
        dw_old = stocbio(x, w_old, idx0, idx1, idx2)

        v_t = grad_w + dw - dw_old
        u_t = grad_x + gx - gx_old

        x_new = copy.deepcopy(x)
        x_new.data = (x - args.x_lr * u_t).data.clone()
        for t in range(args.iterations):

            idx0 = np.random.randint(n_val)
            idx1 = np.random.randint(n_train)
            idx2 = np.random.randint(n_train)

            gx = g_x(x_new, w, (train_x_list[idx1], train_y_list[idx1]), retain_graph=True)
            gx_old = g_x(x, w, (train_x_list[idx1], train_y_list[idx1]), retain_graph=True)

            dw = stocbio(x_new, w, idx0, idx1, idx2)
            dw_old = stocbio(x, w, idx0, idx1, idx2)

            v_t = v_t + dw - dw_old
            u_t = u_t + gx - gx_old

            x.data = x_new.data
            x_new.data = (x - args.x_lr * u_t).data
        return x_new, v_t, u_t

    total_time = 0.0
    n = trainset[0].shape[0]
    stats = []

    outer_opt = torch.optim.SGD([w], lr=args.w_lr, momentum=args.w_momentum)

    w_old = copy.deepcopy(w)

    for epoch in range(args.epochs):

        t0 = time.time()
        if epoch % 1 == 0:
            idx0 = np.random.randint(n_val)
            idx1 = np.random.randint(n_train)
            idx2 = np.random.randint(n_train)
            grad_w = stocbio(x, w, idx0, idx1, idx2)
            grad_x = g_x(x, w, (train_x_list[idx1], train_y_list[idx1]))

        x_, grad_w, grad_x = vrbo(x, w, w_old, grad_x, grad_w)
        x.data = x_.data.clone()
        w_old = copy.deepcopy(w)

        outer_opt.zero_grad()
        w.grad = grad_w
        outer_opt.step()
        t1 = time.time()
        total_time += t1 - t0

        test_loss, test_acc = evaluate(x, w, testset)
        teval_loss, teval_acc = evaluate(x, w, tevalset)
        stats.append((total_time, test_loss, test_acc, teval_loss, teval_acc))
        print(f"[info] epoch {epoch:5d} te loss {test_loss:10.4f} te acc {test_acc:4.2f} teval loss {teval_loss:10.4f} teval acc {teval_acc:10.4f} time {total_time:8.2f}")
    return stats


def MRBO(args, x, w, trainset, valset, testset, tevalset):
    train_batch_size = 1000
    val_batch_size = 1000

    train_dense = trainset[0].to_dense()
    val_dense = valset[0].to_dense()

    train_x_list = list(torch.split(train_dense, train_batch_size))
    train_y_list = list(torch.split(trainset[1], train_batch_size))

    val_x_list = list(torch.split(val_dense, val_batch_size))
    val_y_list = list(torch.split(valset[1], val_batch_size))

    train_x_list = [a.to_sparse() for a in train_x_list]
    val_x_list   = [a.to_sparse() for a in val_x_list]

    n_train = len(train_x_list)
    n_val = len(val_x_list)

    def stocbio(x, w, idx0, idx1, idx2):
        eta = 100.0

        fx = f_x(x, w, (val_x_list[idx0], val_y_list[idx0])) # Fy_gradient
        v_0 = fx.view(-1, 1).detach()

        # Hessian
        z_list = []
        gx = g_x(x, w, (train_x_list[idx1], train_y_list[idx1]), retain_graph=True, create_graph=True)
        gx_ = x.view(-1) - eta * gx.view(-1) # G_gradient
        
        for _ in range(10): # number of Hessian Q steps
            Jacobian = torch.matmul(gx_, v_0)
            v_new = torch.autograd.grad(Jacobian, x, retain_graph=True)[0]
            v_0 = v_new.view(-1, 1).detach()
            z_list.append(v_0)

        v_Q = eta * v_0 + torch.sum(torch.stack(z_list), dim=0)

        # Gyx_gradient
        gx = g_x(x, w, (train_x_list[idx2], train_y_list[idx2]), retain_graph=True, create_graph=True).view(-1)
        gxw = torch.autograd.grad(torch.matmul(gx, v_Q.detach()), w, retain_graph=True)[0]
        return -gxw

    total_time = 0.0
    n = trainset[0].shape[0]
    stats = []

    inner_opt = torch.optim.SGD([x], lr=args.x_lr, momentum=args.x_momentum)
    outer_opt = torch.optim.SGD([w], lr=args.w_lr, momentum=args.w_momentum)

    eta_k, alpha_k, beta_k, m = 1.0, 0.99, 1.0, 0.1

    for epoch in range(args.epochs):
        idx0 = np.random.randint(n_val)
        idx1 = np.random.randint(n_train)
        idx2 = np.random.randint(n_train)
        idx3 = np.random.randint(n_train, size=(args.iterations))

        t0  = time.time()

        if epoch == 0:
            grad_w = stocbio(x, w, idx0, idx1, idx2)
            grad_x = g_x(x, w, (train_x_list[idx1], train_y_list[idx1]))
            x_old, grad_x_old = x, grad_x

            inner_opt.zero_grad()
            x.grad = grad_x
            inner_opt.step()
        else:
            update_x = g_x(x, w, (train_x_list[idx1], train_y_list[idx1]))
            update_x_old = g_x(x_old, w_old, (train_x_list[idx1], train_y_list[idx1]))
            grad_x = update_x + (1 - beta_k) * (grad_x_old - update_x_old)
            x_old, grad_x_old = x, grad_x

            update_w = stocbio(x, w, idx0, idx1, idx2)
            update_w_old = stocbio(x_old, w_old, idx0, idx1, idx2)
            grad_w = update_w + (1-alpha_k) * (grad_w_old - update_w_old)

        x_old, w_old, grad_w_old, grad_x_old = x, w, grad_w, grad_x
        #x.data = x.data - args.x_lr * eta_k * grad_x
        #w.data = w.data - args.w_lr * eta_k * grad_w #- args.w_lr * grad_w
        inner_opt.zero_grad()
        x.grad = grad_x
        inner_opt.step()

        outer_opt.zero_grad()
        import pdb; pdb.set_trace()
        w.grad = grad_w
        outer_opt.step()

        t1 = time.time()

        #eta_k = eta_k * (((epoch+m)/(epoch+m+1))**(1/3))
        #alpha_k, beta_k = alpha_k * (eta_k**2), beta_k*(eta_k**2)

        total_time += t1 - t0
        test_loss, test_acc = evaluate(x, w, testset)
        teval_loss, teval_acc = evaluate(x, w, tevalset)
        stats.append((total_time, test_loss, test_acc, teval_loss, teval_acc))
        print(f"[info] epoch {epoch:5d} te loss {test_loss:10.4f} te acc {test_acc:4.2f} teval loss {teval_loss:10.4f} teval acc {teval_acc:10.4f} time {total_time:8.2f}")
    return stats


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    if args.generate_data:
        trainset, valset, testset, tevalset = get_data(args)
        torch.save((trainset, valset, testset, tevalset), os.path.join(args.data_path, "l2reg.pt"))
        print(f"[info] successfully generated data to {args.data_path}/l2reg.pt")

    elif args.pretrain:
        trainset, valset, testset, tevalset = torch.load(os.path.join(args.data_path, "l2reg.pt"))
        args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        device = args.device
        trainset = (trainset[0].float().to(device), trainset[1].to(device))
        valset   = (valset[0].float().to(device), valset[1].to(device))
        testset  = (testset[0].float().to(device), testset[1].to(device))
        tevalset = (tevalset[0].float().to(device), tevalset[1].to(device))

        # pretrain a model (training without regularization)
        n_feats  = trainset[0].shape[-1]
        num_classes = trainset[1].unique().shape[-1]

        x = torch.randn((n_feats, num_classes), requires_grad=True, device=device)
        x.data = nn.init.kaiming_normal_(x.data.t(), mode='fan_out').t()
        w = torch.zeros(n_feats, requires_grad=True, device=device)

        best_loss, best_acc, x_data = eval("baseline")(args=args,
                                                       x=x,
                                                       w=w,
                                                       trainset=trainset,
                                                       valset=valset,
                                                       testset=testset,
                                                       tevalset=tevalset)
        torch.save(x_data.cpu().data.clone(), f"./save_l2reg/pretrained.pt")

        loss, acc = evaluate(x_data, w, testset)
        torch.save({
            "pretrain_test_loss": loss,
            "pretrain_test_acc": acc,
            }, os.path.join(f"./save_l2reg/pretrained.stats"))
        print(f"[info] Training without regularization results in loss {loss:.2f} acc {acc:.2f}")

    else:

        trainset, valset, testset, tevalset = torch.load(os.path.join(args.data_path, "l2reg.pt"))
        args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        device = args.device
        trainset = (trainset[0].float().to(device), trainset[1].to(device))
        valset   = (valset[0].float().to(device), valset[1].to(device))
        testset  = (testset[0].float().to(device), testset[1].to(device))
        tevalset = (tevalset[0].float().to(device), tevalset[1].to(device))

        n_feats  = trainset[0].shape[-1]
        num_classes = trainset[1].unique().shape[-1]

        #x = torch.randn((n_feats, num_classes), requires_grad=True, device=device)
        x = torch.zeros((n_feats, num_classes), requires_grad=True, device=device)
        x.data = nn.init.kaiming_normal_(x.data.t(), mode='fan_out').t()
        x.data.copy_(torch.load("./save_l2reg/pretrained.pt").to(args.device))
        w = torch.zeros(n_feats, requires_grad=True, device=device)

        pretrained_stats = torch.load("./save_l2reg/pretrained.stats")
        loss = pretrained_stats["pretrain_test_loss"]
        acc  = pretrained_stats["pretrain_test_acc"]
        print(f"[info] pretrained without regularization achieved loss {loss:.2f} acc {acc:.2f}")

        stats = eval(args.alg)(args=args,
                               x=x,
                               w=w,
                               trainset=trainset,
                               valset=valset,
                               testset=testset,
                               tevalset=tevalset)

        if args.alg == "BOME":
            save_path = f"./{args.model_path}/{args.alg}u1{args.u1}_k{args.iterations}_xlr{args.x_lr}_wlr{args.w_lr}_xhatlr{args.xhat_lr}_sd{args.seed}"
        elif args.alg == 'BVFSM':
            save_path = f"./{args.model_path}/{args.alg}_k{args.iterations}_xlr{args.x_lr}_wlr{args.w_lr}_xhatlr{args.xhat_lr}_sd{args.seed}"
        else:
            save_path = f"./{args.model_path}/{args.alg}_k{args.iterations}_xlr{args.x_lr}_wlr{args.w_lr}_sd{args.seed}"
        torch.save(stats, save_path)
