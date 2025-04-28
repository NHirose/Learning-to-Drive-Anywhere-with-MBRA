# Learning to Drive Anywhere with Model-Based Reannotation
[![arXiv](https://img.shields.io/badge/arXiv-2407.08693-df2a2a.svg)](https://arxiv.org/pdf/2407.08693)
[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://learning-language-navigation.github.io)


[Noriaki Hirose](https://sites.google.com/view/noriaki-hirose/)<sup>1, 2</sup>, [Lydia Ignatova](https://www.linkedin.com/in/lydia-ignatova)<sup>1</sup>, [Kyle Stachowicz](https://kylesta.ch/)<sup>1</sup>, [Catherine Glossop](https://www.linkedin.com/in/catherineglossop/)<sup>1</sup>, [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/)<sup>1</sup>, [Dhruv Shah](https://robodhruv.github.io/)<sup>1, 3</sup>

<sup>1</sup> UC Berkeley (_Berkeley AI Research_),  <sup>2</sup> Toyota Motor North America,  <sup>3</sup> Princeton University

We present Model-Based ReAnnotation (MBRA), a framework that utilizes a learned short-horizon, model-based expert model to relabel or generate high-quality actions for passively collected data sources, including large volumes of crowd-sourced teleoperation data and unlabeled YouTube videos. This relabeled data is then distilled into LogoNav, a long-horizon navigation policy conditioned on visual goals or GPS waypoints. LogoNav, trained using MBRA-processed data, achieves state-of-the-art performance, enabling robust navigation over distances exceeding 300 meters in previously unseen indoor and outdoor environments.

![](media/teaser.png)


### Installation
Please down load our code and install some tools for making a conda environment to run our code. We recommend to run our code in the conda environment, although we do not mention the conda environments later.

1. Download the repository on your PC:
    ```
    git clone https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA.git
    ```
2. Set up the conda environment:
    ```
    cd Learning-to-Drive-Anywhere-with-MBRA
    conda env create -f environment_mbra.yml
    ```
3. Source the conda environment:
    ```
    conda activate mbra
    ```
4. Install the MBRA packages:
    ```
    pip install -e train/
    ```
5. Install the `lerobot` package from this [repo](https://github.com/huggingface/lerobot):
    ```
    git clone https://github.com/huggingface/lerobot.git
    cd lerobot
    git checkout 97b1feb0b3c5f28c148dde8a9baf0a175be29d05
    pip install -e .
    ``` 

6. (Optional) Install the diffusion_policy package from this [repo](https://github.com/real-stanford/diffusion_policy): 
    ```
    git clone git@github.com:real-stanford/diffusion_policy.git
    pip install -e diffusion_policy/
    ```

6. Download the model weights from this [link](https://drive.google.com/file/d/1PwQAqC1doeU5rCda4ytil6eRMFuAzUbo/view?usp=sharing)

7. Unzip the downloaded weights and place the folder in (your-directory)/Learning-to-Drive-Anywhere-with-MBRA

8. Download the sampler file from this [link](https://drive.google.com/file/d/1PwQAqC1doeU5rCda4ytil6eRMFuAzUbo/view?usp=sharing)

9. Unzip the sampler file and place the folder in (your-directory)/Learning-to-Drive-Anywhere-with-MBRA/train/vint_train/data

### Dataset
1. Prepare GNM dataset mixture. Please check [here](https://github.com/robodhruv/visualnav-transformer/tree/main)

2. Prepare Frodobots-2k dataset. The Frodobots-2k dataset will be available soon...

### Training
0. Change the directory
    ```
    cd ../train/
    ```
1. Edit the yaml files in (your-directory)/Learning-to-Drive-Anywhere-with-MBRA/train/config to make a path for all datasets and checkpoints

2. Train the re-labeler, imitation-MBRA
    ```
    python train.py -c ./config/MBRA_il.yaml
    ```
3. Train the re-labeler, MPC-MBRA
    ```
    python train.py -c ./config/MBRA_exaug.yaml
    ```
4. Train the navigation policy, LogoNav with imitation-MBRA
    ```
    python train.py -c ./config/LogoNav_il_label.yaml
    ```
5. Train the navigation policy, LogoNav with MPC-MBRA
    ```
    python train.py -c ./config/LogoNav_exaug_label.yaml
    ```
### Inference
1. Now we released the trained checkpoints. We will release the code soon. 

    
## Citing
Our main project
```
@inproceedings{hirose2025mbra,
  title     = {Learning to Drive Anywhere via Model-based Reannotation},
  author    = {Noriaki Hirose and Lydia Ignatova and Kyle Stachowicz, Catherine Glossop and Sergey Levine and Dhruv Shah},
  booktitle = {arXiv},
  year      = {2025},
  url       = {https://arxiv.org/abs/xxxxxxxx}
}
```
