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

from torchvision import datasets


################################################################################
#
#  Bilevel Optimization
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

METHODS = [
    'BSG_1',
    'penalty',
    'VRBO',
    'AID_CG',
    'AID_FP',
    'ITD',
    'BVFSM',
    'stocBiO',
    'reverse',
    'MRBO',
    'BOME',
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="mnist", choices=["mnist", "fashion"])
    parser.add_argument('--train_size', type=int, default=50000)
    parser.add_argument('--val_size', type=int, default=5000)
    parser.add_argument('--pretrain', action='store_true',
                                      default=False, help='whether to create data and pretrain on valset')
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--iterations', type=int, default=10, help='T')
    parser.add_argument('--K', type=int, default=10, help='k')
    parser.add_argument('--data_path', default='./data', help='where to save data')
    parser.add_argument('--model_path', default='./save_data_cleaning', help='where to save model')
    parser.add_argument('--noise_rate', type=float, default=0.5)
    parser.add_argument('--x_lr', type=float, default=0.01)
    parser.add_argument('--xhat_lr', type=float, default=0.01)
    parser.add_argument('--w_lr', type=float, default=100)
    parser.add_argument('--w_momentum', type=float, default=0.9)
    parser.add_argument('--x_momentum', type=float, default=0.9)

    parser.add_argument('--eta', type=float, default=0.01)
    parser.add_argument('--u1', type=float, default=0.1)
    parser.add_argument('--BVFSM_decay', type=str, default='log', choices=['log', 'power2'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--alg', type=str, default='BOME', choices=METHODS)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    return args


def get_data(args):

    data = {
        'mnist': datasets.MNIST,
        'fashion': datasets.FashionMNIST,
    }

    trainset = data[args.dataset](root=args.data_path,
                                  train=True,
                                  download=True)
    testset  = data[args.dataset](root=args.data_path,
                                  train=False,
                                  download=True)

    indices = torch.randperm(len(trainset))

    train_x = trainset.data[indices[:args.train_size]] / 255.
    val_x   = trainset.data[indices[args.train_size:args.train_size+args.val_size]] / 255.
    teval_x = trainset.data[indices[args.train_size+args.val_size:]] / 255.
    test_x  = testset.data / 255.

    targets = trainset.targets if args.dataset in ["mnist", "fashion"] else torch.LongTensor(trainset.targets) 
    train_y = targets[indices[:args.train_size]]
    val_y   = targets[indices[args.train_size:args.train_size+args.val_size]]
    teval_y = targets[indices[args.train_size+args.val_size:]]
    test_y  = torch.LongTensor(testset.targets)

    num_classes = test_y.unique().shape[0]
    assert val_y.unique().shape[0] == num_classes

    ### poison training data with noise rate = args.noise_rate
    num_noisy = int(args.train_size * args.noise_rate)
    rand_indices = torch.randperm(args.train_size)
    noisy_indices = rand_indices[:num_noisy]
    noisy_y = torch.randint(num_classes, size=(num_noisy,))
    old_train_y = train_y.data.clone()
    train_y.data[noisy_indices] = noisy_y.data

    # normalizing inputs to mean 0 and std 1.
    mean = train_x.unsqueeze(1).mean([0,2,3])
    std  = train_x.unsqueeze(1).std([0,2,3])

    trainset = ( torch.flatten((train_x  - mean)/(std+1e-4), start_dim=1), train_y )
    valset   = ( torch.flatten((val_x    - mean)/(std+1e-4), start_dim=1), val_y   )
    testset  = ( torch.flatten((test_x   - mean)/(std+1e-4), start_dim=1), test_y  )
    tevalset = ( torch.flatten((teval_x  - mean)/(std+1e-4), start_dim=1), teval_y )
    return trainset, valset, testset, tevalset, old_train_y

### initialize a linear model

def get_model(in_features, out_features, device):
    x = torch.zeros(out_features, in_features, requires_grad=True, device=device)

    weight = torch.empty((out_features, in_features))
    bias = torch.empty(out_features)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(bias, -bound, bound)

    x[:,:in_features].data.copy_(weight.clone().to(device))
    x[:, -1].data.copy_(bias.clone().to(device))
    x.data.copy_(weight.clone().to(device))
    return x

def model_forward(x, inputs):
    in_features = 28*28
    A = x[:,:in_features] # (out_features, in_features)
    b = x[:,-1] # (out_features,)
    y = inputs.mm(A.t()) + b.view(1,-1)
    return y

### original f, g, and gradients

def f(x, w, dataset):
    data_x, data_y = dataset
    y = model_forward(x, data_x)
    loss = F.cross_entropy(y, data_y)
    return loss

def g(x, w, dataset):
    data_x, data_y = dataset
    y = model_forward(x, data_x)
    loss = F.cross_entropy(y, data_y, reduction='none')
    loss = (loss * torch.clip(w, 0, 1)).mean() + 0.001 * x.norm(2).pow(2)
    return loss

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

def evaluate(x, testset, tevalset):
    with torch.no_grad():
        test_x, test_y = testset  
        y = model_forward(x, test_x)
        test_loss = F.cross_entropy(y, test_y).detach().item()
        test_acc = y.argmax(-1).eq(test_y).float().mean().detach().cpu().item()
        # have a separate test val set since valset is used for training
        teval_x, teval_y = tevalset
        y_ = model_forward(x, teval_x)
        teval_loss = F.cross_entropy(y_, teval_y).detach().item()
    return test_loss, test_acc, teval_loss


def evaluate_importance_f1(w, clean_indices):
    with torch.no_grad():
        w_ = w.gt(0.5).float()
        TP = (w_ * clean_indices.float()).sum()
        recall = TP / (clean_indices.float().sum()+1e-4)
        precision = TP / (w_.sum()+1e-4)
        f1 = 2.0 * recall * precision / (recall + precision + 1e-4)
    return precision.cpu().item(), recall.cpu().item(), f1.cpu().item()


###############################################################################
#
# Bilevel Optimization Training Methods
#
###############################################################################

def simple_train(args, x, data_x, data_y, testset, tevalset, tag='pretrain', regularize=False): # directly train on the dataset
    opt = torch.optim.SGD([x], lr=args.x_lr, momentum=args.x_momentum)
    n = data_x.shape[0]

    n_epochs = 5000
    best_teval_loss = np.inf
    final_test_loss = 0.
    final_test_acc = 0.
    best_x = None

    for epoch in range(n_epochs):
        opt.zero_grad()
        y = model_forward(x, data_x)
        loss = F.cross_entropy(y, data_y)
        if regularize:
            loss += 0.001 * x.norm(2).pow(2)
        loss.backward()
        opt.step()

        test_loss, test_acc, teval_loss = evaluate(x, testset, tevalset)
        if teval_loss <= best_teval_loss:
            best_teval_loss = teval_loss
            final_test_loss = test_loss
            final_test_acc  = test_acc
            best_x = x.data.clone()
        print(f"[{tag}] epoch {epoch:5d} test loss {test_loss:10.4f} test acc {test_acc:10.4f}")
    return final_test_loss, final_test_acc, best_x


def BOME(args, x, w, trainset, valset, testset, tevalset, clean_indices):
    xhat = copy.deepcopy(x)

    total_time = 0.0
    n = trainset[0].shape[0]
    stats = []

    outer_opt = torch.optim.SGD([
        {'params': [x], 'lr': args.x_lr},
        {'params': [w], 'lr': args.w_lr}], momentum=args.w_momentum)
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
        #lmbd = F.relu(args.u1 - dot/(norm_dq + 1e-8))
        lmbd = F.relu((args.u1*loss-dot)/(norm_dq+1e-8))

        outer_opt.zero_grad()
        x.grad = fx + lmbd * gx
        w.grad = lmbd * gw_minus_gw_k
        outer_opt.step()
        t1 = time.time()
        total_time += t1 - t0
        w.data.clamp_(0.0, 1.0)

        test_loss, test_acc, teval_loss = evaluate(x, testset, tevalset)
        f1 = evaluate_importance_f1(w, clean_indices)
        stats.append((total_time, test_loss, test_acc, teval_loss))
        print(f"[info] epoch {epoch:5d} | te loss {test_loss:6.4f} | te acc {test_acc:4.2f} | teval loss {teval_loss:6.4f} | time {total_time:6.2f} | w-min {w.min().item():4.2f} w-max {w.max().item():4.2f} | f1 {f1[2]:4.2f}")
    return stats


def alter(args, x, w, trainset, valset, testset, tevalset, clean_indices):
    xhat = copy.deepcopy(x)

    total_time = 0.0
    n = trainset[0].shape[0]
    stats = []

    x_opt = torch.optim.SGD([x], lr=args.x_lr, momentum=args.w_momentum)
    w_opt = torch.optim.SGD([w], lr=args.w_lr, momentum=args.w_momentum)
    inner_opt = torch.optim.SGD([xhat], lr=args.xhat_lr, momentum=args.x_momentum)

    n_params_x = x.numel()
    n_params_w = w.numel()
    df = torch.zeros(n_params_x+n_params_w).to(x.device)
    dg = torch.zeros(n_params_x+n_params_w).to(x.device)

    eta = args.eta

    for epoch in range(args.epochs):

        xhat.data = x.data.clone()
        t0 = time.time()

        for it in range(args.iterations):
            inner_opt.zero_grad()
            xhat.grad = g_x(xhat, w, trainset).data
            inner_opt.step()

        g_gap = g(x, w, trainset) - g(xhat, w, trainset)
        # update w
        #fw = f_w(x, w, valset)
        gw = torch.autograd.grad(g_gap, w, allow_unused=True)[0].view(-1)
        #grad_w = fw + F.relu((eta * g_gap.item() - fw.dot(gw)) / (gw.dot(gw)+1e-8)) * gw
        grad_w = F.relu(eta * g_gap.item() / (gw.dot(gw)+1e-8)) * gw
        w_opt.zero_grad()
        w.grad = grad_w.view(*w.shape)
        w_opt.step()

        g_gap = g(x, w, trainset) - g(xhat, w, trainset)
        # update x
        fx = f_x(x, w, valset).view(-1)
        gx = torch.autograd.grad(g_gap, x, allow_unused=True)[0].view(-1)
        grad_x = fx + F.relu((eta * g_gap.item() - fx.dot(gx)) / (gx.dot(gx)+1e-8)) * gx
        x_opt.zero_grad()
        x.grad = grad_x.view(*x.shape)
        x_opt.step()

        t1 = time.time()
        total_time += t1 - t0
        w.data.clamp_(0.0, 1.0)

        test_loss, test_acc, teval_loss = evaluate(x, testset, tevalset)
        f1 = evaluate_importance_f1(w, clean_indices)
        stats.append((total_time, test_loss, test_acc, teval_loss))
        print(f"[info] epoch {epoch:5d} | te loss {test_loss:6.4f} | te acc {test_acc:4.2f} | teval loss {teval_loss:6.4f} | time {total_time:6.2f} | w-min {w.min().item():4.2f} w-max {w.max().item():4.2f} | f1 {f1[2]:4.2f}")
    return stats


def simul(args, x, w, trainset, valset, testset, tevalset, clean_indices):
    xhat = copy.deepcopy(x)

    total_time = 0.0
    n = trainset[0].shape[0]
    stats = []

    x_opt = torch.optim.SGD([x], lr=args.x_lr, momentum=args.w_momentum)
    w_opt = torch.optim.SGD([w], lr=args.w_lr, momentum=args.w_momentum)
    inner_opt = torch.optim.SGD([xhat], lr=args.xhat_lr, momentum=args.x_momentum)

    n_params_x = x.numel()
    n_params_w = w.numel()
    df = torch.zeros(n_params_x+n_params_w).to(x.device)
    dg = torch.zeros(n_params_x+n_params_w).to(x.device)

    eta = args.eta

    for epoch in range(args.epochs):

        xhat.data = x.data.clone()
        t0 = time.time()

        for it in range(args.iterations):
            inner_opt.zero_grad()
            xhat.grad = g_x(xhat, w, trainset).data
            inner_opt.step()

        g_gap = g(x, w, trainset) - g(xhat, w, trainset)

        # update w
        fx = f_x(x, w, valset).view(-1)
        gx, gw = torch.autograd.grad(g_gap, [x, w])
        gx = gx.view(-1)
        gw = gw.view(-1)

        #grad_w = fw + F.relu((eta * g_gap.item() - fw.dot(gw)) / (gw.dot(gw)+1e-8)) * gw
        grad_w = F.relu(eta * g_gap.item() / (gw.dot(gw)+1e-8)) * gw
        grad_x = fx + F.relu((eta * g_gap.item() - fx.dot(gx)) / (gx.dot(gx)+1e-8)) * gx

        w_opt.zero_grad()
        x_opt.zero_grad()

        w.grad = grad_w.view(*w.shape)
        x.grad = grad_x.view(*x.shape)

        w_opt.step()
        x_opt.step()

        t1 = time.time()
        total_time += t1 - t0
        w.data.clamp_(0.0, 1.0)

        test_loss, test_acc, teval_loss = evaluate(x, testset, tevalset)
        f1 = evaluate_importance_f1(w, clean_indices)
        stats.append((total_time, test_loss, test_acc, teval_loss))
        print(f"[info] epoch {epoch:5d} | te loss {test_loss:6.4f} | te acc {test_acc:4.2f} | teval loss {teval_loss:6.4f} | time {total_time:6.2f} | w-min {w.min().item():4.2f} w-max {w.max().item():4.2f} | f1 {f1[2]:4.2f}")
    return stats


def BSG_1(args, x, w, trainset, valset, testset, tevalset, clean_indices):
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
        w.grad = (-fx.view(-1).dot(gx.view(-1)) / (gx.norm(2).pow(2)+1e-4) * gw).data.clone()
        outer_opt.step()
        t1 = time.time()
        total_time += t1 - t0
        w.data.clamp_(0.0, 1.0)

        test_loss, test_acc, teval_loss = evaluate(x, testset, tevalset)
        stats.append((total_time, test_loss, test_acc, teval_loss))
        print(f"[info] epoch {epoch:5d} | te loss {test_loss:6.4f} | te acc {test_acc:4.2f} | teval loss {teval_loss:6.4f} | time {total_time:6.2f} | w-min {w.min().item():4.2f} w-max {w.max().item():4.2f}")
    return stats

def reverse(args, x, w, trainset, valset, testset, tevalset, clean_indices):
    return implicit(args, x, w, trainset, valset, testset, tevalset, clean_indices, opt='reverse')

def AID_CG(args, x, w, trainset, valset, testset, tevalset, clean_indices):
    return implicit(args, x, w, trainset, valset, testset, tevalset, clean_indices, opt='AID_CG')

def AID_FP(args, x, w, trainset, valset, testset, tevalset, clean_indices):
    return implicit(args, x, w, trainset, valset, testset, tevalset, clean_indices, opt='AID_FP')

def implicit(args, x, w, trainset, valset, testset, tevalset, clean_indices, opt):
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
        w.data.clamp_(0.0, 1.0)

        x.data = x_history[-1][0].data.clone()
        test_loss, test_acc, teval_loss = evaluate(x, testset, tevalset)
        stats.append((total_time, test_loss, test_acc, teval_loss))
        print(f"[info] epoch {epoch:5d} | te loss {test_loss:6.4f} | te acc {test_acc:4.2f} | teval loss {teval_loss:6.4f} | time {total_time:6.2f} | w-min {w.min().item():4.2f} w-max {w.max().item():4.2f}")
    return stats


def ITD(args, x, w, trainset, valset, testset, tevalset, clean_indices):
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
            x_history.append(inner_opt(x_history[-1], [w], create_graph=True))

        outer_opt.zero_grad()
        loss = outer_loss([x_history[-1][0]], [w])
        grad = torch.autograd.grad(loss, w)[0]
        w.grad = grad.data.clone()
        outer_opt.step()
        t1 = time.time()
        total_time += t1 - t0
        w.data.clamp_(0.0, 1.0)

        x.data = x_history[-1][0].data.clone()

        test_loss, test_acc, teval_loss = evaluate(x, testset, tevalset)
        stats.append((total_time, test_loss, test_acc, teval_loss))
        print(f"[info] epoch {epoch:5d} | te loss {test_loss:6.4f} | te acc {test_acc:4.2f} | teval loss {teval_loss:6.4f} | time {total_time:6.2f} | w-min {w.min().item():4.2f} w-max {w.max().item():4.2f}")
    return stats


def BVFSM(args, x, w, trainset, valset, testset, tevalset, clean_indices):
    z = copy.deepcopy(x)
    w_opt = torch.optim.SGD([w], lr=args.w_lr, momentum=args.w_momentum)
    n = trainset[0].shape[0]
    total_time = 0.0
    stats = []

    z_l2_reg_coef = 0.01
    y_l2_reg_coef = 0.01
    y_ln_reg_coef = 1.

    decay_rate = 1.1

    for epoch in range(args.epochs):

        if args.BVFSM_decay == 'log':
            reg_decay_rate = 1 / (math.log(decay_rate * (epoch+1)))
        elif args.BVFSM_decay == 'power2':
            reg_decay_rate = 1 / ((epoch+1) ** decay_rate)

        z_opt = torch.optim.SGD([z], lr=args.xhat_lr, momentum=args.x_momentum)
        x_opt = torch.optim.SGD([x], lr=args.x_lr, momentum=args.x_momentum)
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
        w.data.clamp_(0.0, 1.0)

        test_loss, test_acc, teval_loss = evaluate(x, testset, tevalset)
        stats.append((total_time, test_loss, test_acc, teval_loss))
        print(f"[info] epoch {epoch:5d} | te loss {test_loss:6.4f} | te acc {test_acc:4.2f} | teval loss {teval_loss:6.4f} | time {total_time:6.2f} | w-min {w.min().item():4.2f} w-max {w.max().item():4.2f}")
    return stats


def penalty(args, x, w, trainset, valset, testset, tevalset, clean_indices):
    total_time = 0.0
    n = trainset[0].shape[0]
    stats = []

    def penalty_gx(x, w, trainset, valset, gamma_k, nu_k):
        gx = torch.autograd.grad(g(x, w, trainset), x, create_graph=True, allow_unused=True)[0]
        loss = f(x, w, valset) + (nu_k * gx).mean() + lmbd_g * g(x, w, trainset) + 0.5 * gamma_k * gx.norm(2).pow(2)
        grad_x = torch.autograd.grad(loss, x, allow_unused=True)[0]
        return grad_x, grad_x.norm().detach().cpu().item()

    def penalty_gw(x, w, trainset, valset, gamma_k, nu_k):
        gx = torch.autograd.grad(g(x, w, trainset), x, create_graph=True, allow_unused=True)[0]
        loss = f(x, w, valset) + (nu_k * gx).mean() + 0.5 * gamma_k * gx.norm(2).pow(2)
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
            outer_opt = torch.optim.SGD([w], lr=w_lr, momentum=args.w_momentum)
            inner_opt = torch.optim.SGD([x], lr=x_lr, momentum=args.x_momentum)
            print("update gamma and eps", gamma, eps)

        t1 = time.time()
        total_time += t1 - t0
        w.data.clamp_(0.0, 1.0)

        test_loss, test_acc, teval_loss = evaluate(x, testset, tevalset)
        stats.append((total_time, test_loss, test_acc, teval_loss))
        print(f"[info] epoch {epoch:5d} | te loss {test_loss:6.4f} | te acc {test_acc:4.2f} | teval loss {teval_loss:6.4f} | time {total_time:6.2f} | w-min {w.min().item():4.2f} w-max {w.max().item():4.2f}")
    return stats


###############################################################################
#
# Below are stochastic methods
#
###############################################################################


def stocBiO(args, x, w, trainset, valset, testset, tevalset, clean_indices):

    batch_size = 1000

    n_val = valset[0].shape[0]
    n_train = trainset[0].shape[0]

    list_idx  = torch.randperm(n_train).to(x.device)
    list_idx0 = torch.randperm(n_val).to(x.device)
    list_idx1 = torch.randperm(n_train).to(x.device)
    list_idx2 = torch.randperm(n_train).to(x.device)

    n_idx  = n_train // batch_size
    n_idx1 = n_train // batch_size
    n_idx2 = n_train // batch_size
    n_idx0 = n_val // batch_size

    def stocbio(x, w, idx0, idx1, idx2):
        eta = 0.0001 if args.dataset == "fashion" else 0.1

        fx = f_x(x, w[idx0], (valset[0][idx0], valset[1][idx0])) # Fy_gradient
        v_0 = fx.view(-1, 1).detach()

        # Hessian
        z_list = []
        gx = g_x(x, w[idx1], (trainset[0][idx1], trainset[1][idx1]), retain_graph=True, create_graph=True)
        gx_ = x.view(-1) - eta * gx.view(-1) # G_gradient
        
        for _ in range(3): # number of Hessian Q steps
            Jacobian = torch.matmul(gx_, v_0)
            v_new = torch.autograd.grad(Jacobian, x, retain_graph=True)[0]
            v_0 = v_new.view(-1, 1).detach()
            z_list.append(v_0)
        v_Q = eta * v_0 + torch.sum(torch.stack(z_list), dim=0)

        # Gyx_gradient
        idx = torch.randperm(n_val)[:batch_size].to(x.device)
        gx = g_x(x, w[idx2], (trainset[0][idx2], trainset[1][idx2]), retain_graph=True, create_graph=True).view(-1)
        gxw = torch.autograd.grad(torch.matmul(gx, v_Q.detach()), w, retain_graph=True)[0]
        return -gxw

    total_time = 0.0
    n = trainset[0].shape[0]
    stats = []

    inner_opt = torch.optim.SGD([x], lr=args.x_lr, momentum=args.x_momentum)
    outer_opt = torch.optim.SGD([w], lr=args.w_lr, momentum=args.w_momentum)

    w_old = copy.deepcopy(w)

    for epoch in range(args.epochs):

        idx3 = torch.randperm(n_train).to(x.device)
        idx0 = torch.randperm(n_val)[:batch_size].to(x.device)
        idx1 = torch.randperm(n_train)[:batch_size].to(x.device)
        idx2 = torch.randperm(n_train)[:batch_size].to(x.device)

        t0 = time.time()
        for t in range(args.iterations):
            #idx = torch.randperm(n_train)[:batch_size].to(x.device)
            idx = idx3[t*batch_size:(t+1)*batch_size]
            tx = time.time()
            gx = g_x(x, w[idx], (trainset[0][idx], trainset[1][idx]))
            inner_opt.zero_grad()
            x.grad = gx
            inner_opt.step()
            ty = time.time()
            total_time += ty-tx
            #import pdb; pdb.set_trace()

        #t02 = time.time()
        grad_w = stocbio(x, w, idx0, idx1, idx2)

        outer_opt.zero_grad()
        w.grad = grad_w
        outer_opt.step()
        t1 = time.time()
        #total_time += t1 - t0
        total_time += t1-ty
        w.data.clamp_(0.0, 1.0)

        test_loss, test_acc, teval_loss = evaluate(x, testset, tevalset)
        f1 = evaluate_importance_f1(w, clean_indices)
        stats.append((total_time, test_loss, test_acc, teval_loss))
        #if epoch % 100 == 0:
        #    import pdb; pdb.set_trace()
        print(f"[info] epoch {epoch:5d} | te loss {test_loss:6.4f} | te acc {test_acc:4.2f} | teval loss {teval_loss:6.4f} | time {total_time:6.2f} | w-min {w.min().item():4.2f} w-max {w.max().item():4.2f} | f1 {f1[2]:4.2f}")
    return stats


def VRBO(args, x, w, trainset, valset, testset, tevalset, clean_indices):

    train_batch_size = 2000
    val_batch_size = 200

    n_val = valset[0].shape[0]
    n_train = trainset[0].shape[0]

    def stocbio(x, w, idx0, idx1, idx2):
        #eta = 0.001 if args.dataset == "fashion" else 0.1
        eta = 0.1

        fx = f_x(x, w[idx0], (valset[0][idx0], valset[1][idx0])) # Fy_gradient
        v_0 = fx.view(-1, 1).detach()

        # Hessian
        z_list = []
        gx = g_x(x, w[idx1], (trainset[0][idx1], trainset[1][idx1]), retain_graph=True, create_graph=True)
        gx_ = x.view(-1) - eta * gx.view(-1) # G_gradient
        
        for _ in range(3): # number of Hessian Q steps
            Jacobian = torch.matmul(gx_, v_0)
            v_new = torch.autograd.grad(Jacobian, x, retain_graph=True)[0]
            v_0 = v_new.view(-1, 1).detach()
            z_list.append(v_0)
        v_Q = eta * v_0 + torch.sum(torch.stack(z_list), dim=0)

        # Gyx_gradient
        gx = g_x(x, w[idx2], (trainset[0][idx2], trainset[1][idx2]), retain_graph=True, create_graph=True).view(-1)
        gxw = torch.autograd.grad(torch.matmul(gx, v_Q.detach()), w, retain_graph=True)[0]
        return -gxw

    def vrbo(x, w, w_old, grad_x, grad_w): 
        idx0 = torch.randperm(n_val)[:val_batch_size*(args.iterations+1)].to(x.device)
        idx1 = torch.randperm(n_train)[:train_batch_size*(args.iterations+1)].to(x.device)
        idx2 = torch.randperm(n_train)[:train_batch_size*(args.iterations+1)].to(x.device)
        idx0 = torch.split(idx0, val_batch_size)
        idx1 = torch.split(idx1, train_batch_size)
        idx2 = torch.split(idx2, train_batch_size)

        t0 = time.time()
        gx = g_x(x, w[idx1[0]], (trainset[0][idx1[0]], trainset[1][idx1[0]]), retain_graph=True)
        gx_old = g_x(x, w_old[idx1[0]], (trainset[0][idx1[0]], trainset[1][idx1[0]]), retain_graph=True)

        dw = stocbio(x, w, idx0[0], idx1[0], idx2[0])
        dw_old = stocbio(x, w_old, idx0[0], idx1[0], idx2[0])

        v_t = grad_w + dw - dw_old
        u_t = grad_x + gx - gx_old

        x_new = copy.deepcopy(x)
        x_new.data = (x - args.x_lr * u_t).data.clone()

        for t in range(args.iterations):

            gx = g_x(x_new, w[idx1[t+1]], (trainset[0][idx1[t+1]], trainset[1][idx1[t+1]]), retain_graph=True)
            gx_old = g_x(x, w[idx1[t+1]], (trainset[0][idx1[t+1]], trainset[1][idx1[t+1]]), retain_graph=True)

            dw = stocbio(x_new, w, idx0[t+1], idx1[t+1], idx2[t+1])
            dw_old = stocbio(x, w, idx0[t+1], idx1[t+1], idx2[t+1])

            v_t = v_t + dw - dw_old
            u_t = u_t + gx - gx_old

            x.data = x_new.data.clone()
            x_new.data = (x - args.x_lr * u_t).data.clone()
        t1 = time.time()
        return x_new, v_t, u_t, t1-t0

    total_time = 0.0
    n = trainset[0].shape[0]
    stats = []

    outer_opt = torch.optim.SGD([w], lr=args.w_lr, momentum=args.w_momentum)

    w_old = copy.deepcopy(w)

    for epoch in range(args.epochs):

        if epoch % 3 == 0:
            idx0 = torch.randperm(n_val)[:val_batch_size].to(x.device)
            idx1 = torch.randperm(n_train)[:train_batch_size].to(x.device)
            idx2 = torch.randperm(n_train)[:train_batch_size].to(x.device)

            grad_w = stocbio(x, w, idx0, idx1, idx2)
            grad_x = g_x(x, w[idx1], (trainset[0][idx1], trainset[1][idx1]))

        x_, grad_w, grad_x, time_ = vrbo(x, w, w_old, grad_x, grad_w)
        x.data = x_.data.clone()
        w_old.data = w.data.clone()

        outer_opt.zero_grad()
        w.grad = grad_w
        outer_opt.step()
        total_time += time_
        w.data.clamp_(0.0, 1.0)

        test_loss, test_acc, teval_loss = evaluate(x, testset, tevalset)
        f1 = evaluate_importance_f1(w, clean_indices)
        stats.append((total_time, test_loss, test_acc, teval_loss))
        print(f"[info] epoch {epoch:5d} | te loss {test_loss:6.4f} | te acc {test_acc:4.2f} | teval loss {teval_loss:6.4f} | time {total_time:6.2f} | w-min {w.min().item():4.2f} w-max {w.max().item():4.2f} | f1 {f1[2]:4.2f}")
    return stats


def MRBO(args, x, w, trainset, valset, testset, tevalset, clean_indices):

    train_batch_size = 1
    val_batch_size = 1

    n_val = valset[0].shape[0]
    n_train = trainset[0].shape[0]

    def stocbio(x, w, idx0, idx1, idx2):
        #eta = 0.001 if args.dataset == "fashion" else 0.1
        #eta = 0.1
        eta = 0.0001 if args.dataset == "fashion" else 0.1

        fx = f_x(x, w[idx0], (valset[0][idx0], valset[1][idx0])) # Fy_gradient
        v_0 = fx.view(-1, 1).detach()

        # Hessian
        z_list = []
        gx = g_x(x, w[idx1], (trainset[0][idx1], trainset[1][idx1]), retain_graph=True, create_graph=True)
        gx_ = x.view(-1) - eta * gx.view(-1) # G_gradient
        
        for _ in range(3): # number of Hessian Q steps
            Jacobian = torch.matmul(gx_, v_0)
            v_new = torch.autograd.grad(Jacobian, x, retain_graph=True)[0]
            v_0 = v_new.view(-1, 1).detach()
            z_list.append(v_0)
        v_Q = eta * v_0 + torch.sum(torch.stack(z_list), dim=0)

        # Gyx_gradient
        #w_ = w[idx2].clone().requires_grad_()
        gx = g_x(x, w[idx2], (trainset[0][idx2], trainset[1][idx2]), retain_graph=True, create_graph=True).view(-1)
        #gx = g_x(x, w_, (trainset[0][idx2], trainset[1][idx2]), retain_graph=True, create_graph=True).view(-1)
        gxw = torch.autograd.grad(torch.matmul(gx, v_Q.detach()), w, retain_graph=True)[0]
        return -gxw

    total_time = 0.0
    n = trainset[0].shape[0]
    stats = []

    outer_opt = torch.optim.SGD([w], lr=args.w_lr, momentum=args.w_momentum)

    w_old = copy.deepcopy(w)

    eta_k, alpha_k, beta_k, m = 1.0, 0.9, 0.9, 0.1

    for epoch in range(args.epochs):

        idx0 = torch.randperm(n_val)[:val_batch_size].to(x.device)
        idx1 = torch.randperm(n_train)[:train_batch_size].to(x.device)
        idx2 = torch.randperm(n_train)[:train_batch_size].to(x.device)

        t0  = time.time()

        if epoch == 0:
            grad_w = stocbio(x, w, idx0, idx1, idx2)
            grad_x = g_x(x, w[idx1], (trainset[0][idx1], trainset[1][idx1]))
            x_old, grad_x_old = x, grad_x
            x = x - args.x_lr * eta_k * grad_x
        else:
            update_x = g_x(x, w[idx1], (trainset[0][idx1], trainset[1][idx1]))
            update_x_old = g_x(x_old, w_old[idx1], (trainset[0][idx1], trainset[1][idx1]))
            grad_x = update_x + (1 - beta_k) * (grad_x_old - update_x_old)
            x_old, grad_x_old = x, grad_x

            update_w = stocbio(x, w, idx0, idx1, idx2)
            update_w_old = stocbio(x_old, w_old, idx0, idx1, idx2)
            grad_w = update_w + (1-alpha_k) * (grad_w_old - update_w_old)

        x_old, w_old, grad_w_old, grad_x_old = x, w, grad_w, grad_x
        #import pdb; pdb.set_trace()
        x.data = x.data - args.x_lr * eta_k * grad_x
        w.data = w.data - args.w_lr * eta_k * grad_w #- args.w_lr * grad_w

        t1 = time.time()

        #eta_k = eta_k * (((epoch+m)/(epoch+m+1))**(1/3))
        #alpha_k, beta_k = alpha_k * (eta_k**2), beta_k*(eta_k**2)
        #outer_opt.zero_grad()
        #w.grad = grad_w
        #outer_opt.step()
        total_time += t1 - t0
        w.data.clamp_(0.0, 1.0)

        test_loss, test_acc, teval_loss = evaluate(x, testset, tevalset)
        f1 = evaluate_importance_f1(w, clean_indices)
        stats.append((total_time, test_loss, test_acc, teval_loss))
        print(f"[info] epoch {epoch:5d} | te loss {test_loss:6.4f} | te acc {test_acc:4.2f} | teval loss {teval_loss:6.4f} | time {total_time:6.2f} | w-min {w.min().item():4.2f} w-max {w.max().item():4.2f} | f1 {f1[2]:4.2f}")
    return stats


if __name__ == "__main__":
    args = parse_args()

    if args.pretrain: # preprocess data and pretrain a model on validation set

        if not os.path.exists(args.data_path):
            os.makedirs(args.data_path)

        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)

        ### generate data
        trainset, valset, testset, tevalset, old_train_y = get_data(args)
        torch.save((trainset, valset, testset, tevalset, old_train_y),
                   os.path.join(args.data_path, f"{args.dataset}_data_cleaning.pt"))
        print(f"[info] successfully generated data to {args.data_path}/{args.dataset}_data_cleaning.pt")

        ### pretrain a model and save it
        n_feats = np.prod(*trainset[0].shape[1:])
        num_classes = trainset[1].unique().shape[-1]
        args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        trainset = (trainset[0].to(args.device), trainset[1].to(args.device))
        valset   = (valset[0].to(args.device),   valset[1].to(args.device))
        testset  = (testset[0].to(args.device),  testset[1].to(args.device))
        tevalset = (tevalset[0].to(args.device), tevalset[1].to(args.device))
        old_train_y = old_train_y.to(args.device)

        x = get_model(n_feats, num_classes, args.device)
        sd = x.data.clone()

        # lower bound (train on noisy train + valset)
        tmp_x = torch.cat([trainset[0], valset[0]], 0)
        tmp_y = torch.cat([trainset[1], valset[1]], 0)
        test_loss1, test_acc1, best_x1 = simple_train(args, x, tmp_x, tmp_y, testset, tevalset, regularize=True)
        torch.save(best_x1.data.cpu().clone(),
                   os.path.join(args.model_path, f"{args.dataset}_pretrained.pt"))

        # a baseline: train on valset
        x.data.copy_(sd)
        test_loss2, test_acc2, best_x2 = simple_train(args, x, valset[0], valset[1], testset, tevalset)
        torch.save(best_x2.data.cpu().clone(),
                   os.path.join(args.model_path, f"{args.dataset}_pretrained_val.pt"))

        # upper bound (train on correct train + valset)
        x.data.copy_(sd)
        tmp_x = torch.cat([trainset[0], valset[0]], 0)
        tmp_y = torch.cat([old_train_y, valset[1]], 0)
        test_loss3, test_acc3, best_x3 = simple_train(args, x, tmp_x, tmp_y, testset, tevalset)
        torch.save(best_x3.data.cpu().clone(),
                   os.path.join(args.model_path, f"{args.dataset}_pretrained_trainval.pt"))

        print(f"[pretrained] noisy train + val   : test loss {test_loss1} test acc {test_acc1}")
        print(f"[pretrained] val                 : test loss {test_loss2} test acc {test_acc2}")
        print(f"[pretrained] correct train + val : test loss {test_loss3} test acc {test_acc3}")

        torch.save({
            "pretrain_test_loss": test_loss1,
            "pretrain_test_acc": test_acc1,
            "pretrain_val_test_loss": test_loss2,
            "pretrain_val_test_acc": test_acc2,
            "pretrain_trainval_test_loss": test_loss3,
            "pretrain_trainval_test_acc": test_acc3,
            }, os.path.join(args.model_path, f"{args.dataset}_pretrained.stats"))


    else: # load pretrained model on valset and then start model training
        trainset, valset, testset, tevalset, old_train_y = torch.load(
                os.path.join(args.data_path, f"{args.dataset}_data_cleaning.pt"))
        args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        n_feats = np.prod(*trainset[0].shape[1:])
        num_classes = trainset[1].unique().shape[-1]

        trainset = (trainset[0].to(args.device), trainset[1].to(args.device))
        valset   = (valset[0].to(args.device),   valset[1].to(args.device))
        testset  = (testset[0].to(args.device),  testset[1].to(args.device))
        tevalset = (tevalset[0].to(args.device), tevalset[1].to(args.device))
        old_train_y = old_train_y.to(args.device)

        x = get_model(n_feats, num_classes, args.device)
        x.data.copy_(torch.load(os.path.join(args.model_path, f"{args.dataset}_pretrained.pt")).to(args.device))

        # load the pretrained model on validation set
        pretrained_stats = torch.load(
            os.path.join(args.model_path, f"{args.dataset}_pretrained.stats"))

        test_loss1 = pretrained_stats['pretrain_test_loss']
        test_loss2 = pretrained_stats['pretrain_val_test_loss']
        test_loss3 = pretrained_stats['pretrain_trainval_test_loss']
        test_acc1  = pretrained_stats['pretrain_test_acc']
        test_acc2  = pretrained_stats['pretrain_val_test_acc']
        test_acc3  = pretrained_stats['pretrain_trainval_test_acc']
        print(f"[pretrained] noisy train + val   : test loss {test_loss1} test acc {test_acc1}")
        print(f"[pretrained] val                 : test loss {test_loss2} test acc {test_acc2}")
        print(f"[pretrained] correct train + val : test loss {test_loss3} test acc {test_acc3}")

        test_loss, test_acc, teval_loss = evaluate(x, testset, tevalset)
        print("original test loss ", test_loss, "original test acc ", test_acc)

        clean_indices = old_train_y.to(args.device).eq(trainset[1])
        w = torch.zeros(trainset[0].shape[0], requires_grad=True, device=args.device)
        w.data.add_(0.5)

        stats = eval(args.alg)(args=args,
                               x=x,
                               w=w,
                               trainset=trainset,
                               valset=valset,
                               testset=testset,
                               tevalset=tevalset,
                               clean_indices=clean_indices)

        if args.alg == "BOME":
            save_path = f"./{args.model_path}/{args.dataset}_{args.alg}u1{args.u1}_k{args.iterations}_xlr{args.x_lr}_wlr{args.w_lr}_xhatlr{args.xhat_lr}_sd{args.seed}"
        elif args.alg == 'BVFSM':
            save_path = f"./{args.model_path}/{args.dataset}_{args.alg}_k{args.iterations}_xlr{args.x_lr}_wlr{args.w_lr}_xhatlr{args.xhat_lr}_sd{args.seed}"
        elif args.alg == 'VRBO':
            save_path = f"./{args.model_path}/{args.dataset}_new{args.alg}_k{args.iterations}_xlr{args.x_lr}_wlr{args.w_lr}_xhatlr{args.xhat_lr}_sd{args.seed}"
        else:
            save_path = f"./{args.model_path}/{args.dataset}_{args.alg}_k{args.iterations}_xlr{args.x_lr}_wlr{args.w_lr}_sd{args.seed}"
        torch.save(stats, save_path)
