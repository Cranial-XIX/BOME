import os
import argparse

os.system("mkdir -p ./trainlogs/")


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="mnist", choices=["mnist", "fashion", "l2reg"])
parser.add_argument('--alg', type=str)
args = parser.parse_args()


iterations_list = [1, 5, 10, 20]
w_lr_list = [10.0, 50.0, 100.0, 500.0, 1000.0]


if args.dataset in ["mnist", "fashion"]:
    dataset = args.dataset
    epoch = 5000
    x_lr = 0.01
    xhat_lr = 0.01

    seeds = [1,2,3,4,5]

    for seed in seeds:
        alg = "BOME"
        for u1 in [0.1, 0.5, 0.9]:
            for iterations in iterations_list:
                for w_lr in w_lr_list:
                    os.system(f"python data_cleaning.py --dataset {dataset} --x_momentum 0.0 --alg {alg} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} --xhat_lr {xhat_lr} --u1 {u1} > trainlogs/{dataset}_{alg}u1{u1}_{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_{seed}.log")

        alg = "BSG_1"
        for iterations in iterations_list:
            for w_lr in w_lr_list:
                os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{x_lr}_{seed}.log")

        alg = "penalty"
        for iterations in iterations_list:
            for w_lr in w_lr_list:
                os.system(f"python data_cleaning.py --dataset {dataset} --x_momentum 0.0 --alg {alg} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{x_lr}_{seed}.log")

        alg = "ITD"
        for iterations in iterations_list:
            for w_lr in w_lr_list:
                os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{x_lr}_{seed}.log")

        alg = "AID_CG"
        for iterations in iterations_list:
            for w_lr in w_lr_list:
                os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{x_lr}_{seed}.log")

        alg = "AID_FP"
        for iterations in iterations_list:
            for w_lr in w_lr_list:
                os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{x_lr}_{seed}.log")
        alg = "reverse"
        for iterations in iterations_list:
            for w_lr in w_lr_list:
                os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{x_lr}_{seed}.log")

        alg = "BVFSM"
        for iterations in iterations_list:
            for w_lr in w_lr_list:
                os.system(f"python data_cleaning.py --dataset {dataset} --x_momentum 0.0 --alg {alg} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} --xhat_lr {xhat_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_{seed}.log")

        alg = "VRBO"
        for iterations in iterations_list:
            for w_lr in w_lr_list:
                os.system(f"python data_cleaning.py --dataset {dataset} --x_momentum 0.0 --alg {alg} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{w_lr}_{seed}.log")

else:
    x_lr = xhat_lr = 100.0
    seeds = [1,2,3,4,5]

    for seed in seeds:
        u1 = 0.5
        alg = "BOME"
        for iterations, w_lr in zip(iterations_list, [100.0, 100.0, 100.0, 100.0]):
            os.system(f"python l2reg.py --alg {alg} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} --xhat_lr {xhat_lr} --u1 {u1} > trainlogs/l2reg_{alg}u1{u1}_{iterations}_xlr{x_lr}_w{w_lr}_xhatlr{xhat_lr}_{seed}.log")

        alg = "BSG_1"
        for iterations, w_lr in zip(iterations_list, [1000.0, 10000.0, 10000.0, 10000.0]):
            os.system(f"python l2reg.py --alg {alg} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} > trainlogs/l2reg_{alg}_{iterations}_xlr{x_lr}_w{w_lr}_{seed}.log")

        alg = "penalty"
        for iterations, w_lr in zip(iterations_list, [100.0, 100.0, 100.0, 100.0]):
            os.system(f"python l2reg.py --alg {alg} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} > trainlogs/l2reg_{alg}_{iterations}_xlr{x_lr}_w{w_lr}_{seed}.log")

        alg = "ITD"
        for iterations, w_lr in zip(iterations_list, [500.0, 500.0, 100.0, 100.0]):
            os.system(f"python l2reg.py --alg {alg} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} > trainlogs/l2reg_{alg}_{iterations}_xlr{x_lr}_w{w_lr}_{seed}.log")

        alg = "AID_CG"
        for iterations, w_lr in zip(iterations_list, [0.5, 1.0, 1.0, 1.0]):
            os.system(f"python l2reg.py --alg {alg} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} > trainlogs/l2reg_{alg}_{iterations}_xlr{x_lr}_w{w_lr}_{seed}.log")

        alg = "AID_FP"
        for iterations, w_lr in zip(iterations_list, [1.0, 100.0, 100.0, 100.0]):
            os.system(f"python l2reg.py --alg {alg} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} > trainlogs/l2reg_{alg}_{iterations}_xlr{x_lr}_w{w_lr}_{seed}.log")

        alg = "reverse"
        for iterations, w_lr in zip(iterations_list, [1.0, 100.0, 100.0, 100.0]):
            os.system(f"python l2reg.py --alg {alg} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} > trainlogs/l2reg_{alg}_{iterations}_xlr{x_lr}_w{w_lr}_{seed}.log")

        alg = "BVFSM"
        for (iterations, w_lr) in zip(iterations_list, [1000.0, 500.0, 100.0, 50.0]):
            os.system(f"python l2reg.py --alg {alg} --x_momentum 0.0 --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} --xhat_lr {xhat_lr} > trainlogs/l2reg_{alg}_{iterations}_xlr{x_lr}_w{w_lr}_xhatlr{xhat_lr}_{seed}.log")

        alg = "VRBO"
        x_lr = xhat_lr = 1000.0
        for iterations, w_lr in zip(iterations_list, [1000.0, 1000.0, 1000.0, 1000.0]):
            os.system(f"python l2reg.py --alg {alg} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} > trainlogs/l2reg_{alg}_{iterations}_xlr{x_lr}_w{w_lr}_{seed}.log")
