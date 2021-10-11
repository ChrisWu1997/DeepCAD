# DeepCAD
This repository provides source code for our paper:

[DeepCAD: A Deep Generative Network for Computer-Aided Design Models](https://arxiv.org/abs/2105.09492)

[Rundi Wu](https://chriswu1997.github.io), [Chang Xiao](http://chang.engineer), [Changxi Zheng](http://www.cs.columbia.edu/~cxz/index.htm)

ICCV 2021 (camera ready version coming soon)

<p align="center">
  <img src='teaser.png' width=600>
</p>

We also release the Onshape CAD data parsing scripts here: [onshape-cad-parser](https://github.com/ChrisWu1997/onshape-cad-parser).

## Prerequisites

- Linux
- NVIDIA GPU + CUDA CuDNN
- Python 3.7, PyTorch 1.5+


## Dependencies

Install python package dependencies through pip:

```bash
$ pip install -r requirements.txt
```

Install [pythonocc](https://github.com/tpaviot/pythonocc-core) (OpenCASCADE) by conda:

```bash
$ conda install -c conda-forge pythonocc-core=7.5.1
```


## Data

Download data from [here](http://www.cs.columbia.edu/cg/deepcad/data.tar) ([backup](https://drive.google.com/drive/folders/1mSJBZjKC-Z5I7pLPTgb4b5ZP-Y6itvGG?usp=sharing)) and extract them under `data` folder. 
- `cad_json` contains the original json files that we parsed from Onshape and each file describes a CAD construction sequence. 
- `cad_vec` contains our vectorized representation for CAD sequences, which serves for fast data loading. They can also be obtained using `dataset/json2vec.py`.
TBA.
- Some evaluation metrics that we use requires ground truth point clouds. Run:
  ```bash
  $ cd dataset
  $ python json2pc.py --only_test
  ```
The data we used are parsed from Onshape public documents with links from [ABC dataset](https://archive.nyu.edu/handle/2451/61215). We also release our parsing scripts [here](https://github.com/ChrisWu1997/onshape-cad-parser) for anyone who are interested in parsing their own data.


## Training
See all hyper-parameters and configurations under `config` folder. To train the autoencoder:

```bash
$ python train.py --exp_name newDeepCAD -g 0
```

For random generation, further train a latent GAN:

```bash
# encode all data to latent space
$ python test.py --exp_name newDeepCAD --mode enc --ckpt 1000 -g 0

# train latent GAN (wgan-gp)
$ python lgan.py --exp_name newDeepCAD --ae_ckpt 1000 -g 0
```

The trained models and experment logs will be saved in `proj_log/newDeepCAD/` by default. 



## Testing and Evaluation

#### __Autoencoding__

  After training the autoencoder, run the model to reconstruct all test data:

  ```bash
  $ python test.py --exp_name newDeepCAD --mode rec --ckpt 1000 -g 0
  ```
  The results will be saved in`proj_log/newDeepCAD/results/test_1000` by default in the format of `h5` (CAD sequence saved in vectorized representation).

  To evaluate the results:

  ```bash
  $ cd evaluation
  # for command accuray and parameter accuracy
  $ python evaluate_ae_acc.py --src ../proj_log/newDeepCAD/results/test_1000
  # for chamfer distance and invalid ratio
  $ python evaluate_ae_cd.py --src ../proj_log/newDeepCAD/results/test_1000 --parallel
  ```

#### __Random Generation__

  After training the latent GAN, run latent GAN and the autoencoder to do random generation:

  ```bash
  # run latent GAN to generate fake latent vectors
  $ python lgan.py --exp_name newDeepCAD --ae_ckpt 1000 --ckpt 200000 --test --n_samples 9000 -g 0
  
  # run the autoencoder to decode into final CAD sequences
  $ python test.py --exp_name newDeepCAD --mode dec --ckpt 1000 --z_path proj_log/newDeepCAD/lgan_1000/results/fake_z_ckpt200000_num9000.h5 -g 0
  ```
  The results will be saved in`proj_log/newDeepCAD/lgan_1000/results` by default.

  To evaluate the results by COV, MMD and JSD:

  ```bash
  $ cd evaluation
  $ sh run_eval_gen.sh ../proj_log/newDeepCAD/lgan_1000/results/fake_z_ckpt200000_num9000_dec 1000 0
  ```
  The script `run_eval_gen.sh` combines `collect_gen_pc.py` and `evaluate_gen_torch.py`. 
  You can also run these two files individually with specified arguments.


## Pre-trained models

Download pretrained model from [here](http://www.cs.columbia.edu/cg/deepcad/pretrained.tar) ([backup](https://drive.google.com/file/d/16RzOChCdLM5L1VUSFpgHwqU7JoQOF2Nd/view?usp=sharing)) and extract it under `proj_log`. All testing commands shall be able to excecuted directly, by specifying `--exp_name=pretrained` when needed.


## Visualization and Export
We provide scripts to visualize CAD models and export the results to `.step` files, which can be loaded by almost all modern CAD softwares.
```bash
$ cd utils
$ python show.py --src {source folder} # visualize with opencascade
$ python export2step.py --src {source folder} # export to step format
```
Script to create CAD modeling sequence in Onshape according to generated outputs: TBA.

## Acknowledgement
We would like to thank and acknowledge referenced codes from [DeepSVG](https://github.com/alexandre01/deepsvg), [latent 3d points](https://github.com/optas/latent_3d_points) and [PointFlow](https://github.com/stevenygd/PointFlow).

## Cite

Please cite our work if you find it useful:
```
@InProceedings{Wu_2021_ICCV,
    author    = {Wu, Rundi and Xiao, Chang and Zheng, Changxi},
    title     = {DeepCAD: A Deep Generative Network for Computer-Aided Design Models},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {6772-6782}
}
```
