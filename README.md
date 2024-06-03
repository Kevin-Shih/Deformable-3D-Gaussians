# Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction

## [Project page](https://ingra14m.github.io/Deformable-Gaussians/) | [Paper](https://arxiv.org/abs/2309.13101)

![Teaser image](assets/teaser.png)

This repository contains the official implementation associated with the paper "Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction".

# This is a clone of the original repo!!
## changes made
- added differnt annealing strategies
- hyperparameter tuning
- add GES version in `train_ges.py`, which use GEF instead of Gaussians to represent 3D scene.

## Instructions for using different annealing strategies
For additional details please refer to the file arguments/\_\_init\_\_.py

Below are the parameters that can be changed in the command line to modify the annealing parameters.
```
--ast_init  0.1
--ast_final 1e-15
--ast_delay_mult  0.01
--ast_delay_steps  0
--ast_max_steps 20000
--ast_strategy  "linear"
--ast_decay_coef  0.5
--ast_interval_steps 5000
```

## Instructions of using Point-E to generate the initial point cloud
Please use [image2pointcloud.ipynb](https://github.com/openai/point-e/blob/main/point_e/examples/image2pointcloud.ipynb) to generate the 3D point cloud at t=0.
To ensure compatibility with the program, please format it according to `fetchPly()` and `storePly()` in [dataset_readers.py](https://github.com/kie4280/Deformable-3D-Gaussians/blob/main/scene/dataset_readers.py). Save the point clouds as `points3d.ply` in the dataset.

## Dataset

In our paper, we use:

- synthetic dataset from [D-NeRF](https://www.albertpumarola.com/research/D-NeRF/index.html).
- real-world dataset from [NeRF-DS](https://jokeryan.github.io/projects/nerf-ds/) and [Hyper-NeRF](https://hypernerf.github.io/).
- The dataset in the supplementary materials comes from [DeVRF](https://jia-wei-liu.github.io/DeVRF/).

We organize the datasets as follows:

```shell
├── data
│   | D-NeRF 
│     ├── hook
│     ├── standup 
│     ├── ...
│   | NeRF-DS
│     ├── as
│     ├── basin
│     ├── ...
│   | HyperNeRF
│     ├── interp
│     ├── misc
│     ├── vrig
```

> I have identified an **inconsistency in the D-NeRF's Lego dataset**. Specifically, the scenes corresponding to the training set differ from those in the test set. This discrepancy can be verified by observing the angle of the flipped Lego shovel. To meaningfully evaluate the performance of our method on this dataset, I recommend using the **validation set of the Lego dataset** as the test set. See more in [D-NeRF dataset used in Deformable-GS](https://github.com/ingra14m/Deformable-3D-Gaussians/releases/tag/v0.1-pre-released)



## Pipeline

![Teaser image](assets/pipeline.png)



## Run

### Environment

```shell
git clone https://github.com/ingra14m/Deformable-3D-Gaussians --recursive
cd Deformable-3D-Gaussians

conda create -n deformable_gaussian_env python=3.7
conda activate deformable_gaussian_env

# install pytorch
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# install dependencies
pip install -r requirements.txt
```



### Train

**D-NeRF:**

```shell
python train.py -s path/to/your/d-nerf/dataset -m output/exp-name --eval --is_blender
```

**NeRF-DS/HyperNeRF:**

```shell
python train.py -s path/to/your/real-world/dataset -m output/exp-name --eval --iterations 20000
```

**6DoF Transformation:**

We have also implemented the 6DoF transformation of 3D-GS, which may lead to an improvement in metrics but will reduce the speed of training and inference.

```shell
# D-NeRF
python train.py -s path/to/your/d-nerf/dataset -m output/exp-name --eval --is_blender --is_6dof

# NeRF-DS & HyperNeRF
python train.py -s path/to/your/real-world/dataset -m output/exp-name --eval --is_6dof --iterations 20000
```

You can also **train with the GUI:**

```shell
python train_gui.py -s path/to/your/dataset -m output/exp-name --eval --is_blender
```

- click `start` to start training, and click `stop` to stop training.
- The GUI viewer is still under development, many buttons do not have corresponding functions currently. We plan to :
  - [ ] reload checkpoints from the pre-trained model.
  - [ ] Complete the functions of the other vacant buttons in the GUI.



### Render & Evaluation

```shell
python render.py -m output/exp-name --mode render
python metrics.py -m output/exp-name
```

We provide several modes for rendering:

- `render`: render all the test images
- `time`: time interpolation tasks for D-NeRF dataset
- `all`: time and view synthesis tasks for D-NeRF dataset
- `view`: view synthesis tasks for D-NeRF dataset
- `original`: time and view synthesis tasks for real-world dataset



## Results

### D-NeRF Dataset

**Quantitative Results**

<img src="assets/results/D-NeRF/Quantitative.jpg" alt="Image1" style="zoom:50%;" />

**Qualitative Results**

 <img src="assets/results/D-NeRF/bouncing.gif" alt="Image1" style="zoom:25%;" />  <img src="assets/results/D-NeRF/hell.gif" alt="Image1" style="zoom:25%;" />  <img src="assets/results/D-NeRF/hook.gif" alt="Image3" style="zoom:25%;" />  <img src="assets/results/D-NeRF/jump.gif" alt="Image4" style="zoom:25%;" /> 

 <img src="assets/results/D-NeRF/lego.gif" alt="Image5" style="zoom:25%;" />  <img src="assets/results/D-NeRF/mutant.gif" alt="Image6" style="zoom:25%;" />  <img src="assets/results/D-NeRF/stand.gif" alt="Image7" style="zoom:25%;" />  <img src="assets/results/D-NeRF/trex.gif" alt="Image8" style="zoom:25%;" /> 



### NeRF-DS Dataset

<img src="assets/results/NeRF-DS/Quantitative.jpg" alt="Image1" style="zoom:50%;" />

See more visualization on our [project page](https://ingra14m.github.io/Deformable-Gaussians/).



### HyperNeRF Dataset

Since the **camera pose** in HyperNeRF is less precise compared to NeRF-DS, we use HyperNeRF as a reference for partial visualization and the display of Failure Cases, but do not include it in the calculation of quantitative metrics. The results of the HyperNeRF dataset can be viewed on the [project page](https://ingra14m.github.io/Deformable-Gaussians/).



### Real-Time Viewer

https://github.com/ingra14m/Deformable-3D-Gaussians/assets/63096187/ec26d0b9-c126-4e23-b773-dcedcf386f36



## Acknowledgments

We sincerely thank the authors of [3D-GS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), [D-NeRF](https://www.albertpumarola.com/research/D-NeRF/index.html), [HyperNeRF](https://hypernerf.github.io/), [NeRF-DS](https://jokeryan.github.io/projects/nerf-ds/), and [DeVRF](https://jia-wei-liu.github.io/DeVRF/), whose codes and datasets were used in our work. We thank [Zihao Wang](https://github.com/Alen-Wong) for the debugging in the early stage, preventing this work from sinking. We also thank the reviewers and AC for not being influenced by PR, and fairly evaluating our work. This work was mainly supported by ByteDance MMLab.




## BibTex

```
@article{yang2023deformable3dgs,
    title={Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction},
    author={Yang, Ziyi and Gao, Xinyu and Zhou, Wen and Jiao, Shaohui and Zhang, Yuqing and Jin, Xiaogang},
    journal={arXiv preprint arXiv:2309.13101},
    year={2023}
}
```

And thanks to the authors of [3D Gaussians](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) for their excellent code, please consider also cite this repository:

```
@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

Also, thanks to the authors of [GES](https://abdullahamdi.com/ges/) for their excellent code, please consider also cite this repository:
```
@InProceedings{hamdi_2024_CVPR,
    author    = {Hamdi, Abdullah and Melas-Kyriazi, Luke and Mai, Jinjie and Qian, Guocheng and Liu, Ruoshi and Vondrick, Carl and Ghanem, Bernard and Vedaldi, Andrea},
    title     = {GES: Generalized Exponential Splatting for Efficient Radiance Field Rendering},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2024},
    url       = {https://abdullahamdi.com/ges/}
}
```