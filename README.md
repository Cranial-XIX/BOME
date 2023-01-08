# Bilevel Optimization Made Easy (BOME)
This repo contains the source code for BOME, which has been accepted to **NeurIPS 2022**.

*This repo is still under construction, please feel free to reach out via bliu@cs.utexas.edu if
you have any question.*

## Data cleaning and L2 Regularization
Please go the ```hpo``` folder and follow the ```grid.py``` as an example to run the code
as well as other baseline methods. For your convenience, we have pre-processed the dataset
and saved them into ```save_data_cleaning``` and ```save_l2reg``` folders respectively.

Please download the [dataset](https://drive.google.com/file/d/14deh-F4YlEH1c_s0P5DSliU042QV39K3/view?usp=sharing)
and unzip it under the ```hpo``` folder.

## Toys (adversarial, low-level singleton, coreset selection)
Please go to the ```toy``` folder and read the corresponding python script.

## Citations
If you find our work interesting or the repo useful, please consider citing [this paper](https://arxiv.org/pdf/2209.08709.pdf):
```
@article{ye2022bome,
  title={Bome! bilevel optimization made easy: A simple first-order approach},
  author={Liu, Bo and Ye, Mao and Wright, Stephen and Stone, Peter and Liu, Qiang},
  journal={arXiv preprint arXiv:2209.08709},
  year={2022}
}
```
