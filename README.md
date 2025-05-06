# Learning to Drive Anywhere with Model-Based Reannotation
[![arXiv](https://img.shields.io/badge/arXiv-2407.08693-df2a2a.svg)](https://arxiv.org/pdf/2407.08693)
[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://model-base-reannotation.github.io/)


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
    conda env create -f train/environment_mbra.yml
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

7. Download the model weights from this [link](https://drive.google.com/file/d/1zZpGoJYPhQDN_riUsJBR4O9lF4uhM5_B/view?usp=sharing)

8. Unzip the downloaded weights and place the folder in (your-directory)/Learning-to-Drive-Anywhere-with-MBRA/deployment

9. Download the sampler file from this [link](https://drive.google.com/file/d/1PwQAqC1doeU5rCda4ytil6eRMFuAzUbo/view?usp=sharing)

10. Unzip the sampler file and place the folder in (your-directory)/Learning-to-Drive-Anywhere-with-MBRA/train/vint_train/data

### Dataset
1. Prepare GNM dataset mixture. Please check [here](https://github.com/robodhruv/visualnav-transformer/tree/main)

2. Prepare Frodobots-2k dataset. You can download the Frodobots-2k dataset from [here](https://huggingface.co/datasets/frodobots/FrodoBots-2K)

3. Download the repository to convert the dataset:
    ```
    cd ..
    git clone https://github.com/catglossop/frodo_dataset.git
    ```
4. Change the format of Frodobots-2k dataset. Note that you need to specify the dataset directory and the directory to export the proccessed dataset in "convert_to_hf.py".
    ```
    cd frodo_dataset
    python convert_to_hf.py
    ```

### Training
1. Change the directory
    ```
    cd ../Learning-to-Drive-Anywhere-with-MBRA/train/
    ```
2. Edit the yaml files in (your-directory)/Learning-to-Drive-Anywhere-with-MBRA/train/config to make a path for all datasets and checkpoints. 

3. Train the MBRA model to reannotate the dataset,
    ```
    python train.py -c ./config/MBRA.yaml
    ```
4. Train the LogoNav policy with MBRA model
    ```
    python train.py -c ./config/LogoNav.yaml
    ```
### Inference (ROS)
1. ROS system. Please setup ROS in your PC and run the following. Our node subscribes the image topic "/usb_cam/image_raw" and calculate our policy as a callback function. The goal pose at line 243 and 244 and the current robot pose at line 104, 105 and 106 in LogoNav_ros.py have to be decided by yourself. 
    ```
    cd ../deployment/
    python LogoNav_ros.py
    ```
### Inference (FrodoBots system)    
1. Dowload the FrodoBots SDK and setup following their website.
    ```
    cd ..
    git clone https://github.com/frodobots-org/earth-rovers-sdk.git
    ```
2. Move our code
    ```
    mv ./deployment/LogoNav_frodobot.py ./deployment/earth-rovers-sdk/utils/
    mv ./deployment/utils_logonav.py ./deployment/earth-rovers-sdk/utils/
    ```
3. Run our code. Before running our policy, you need to setup the Frodobots (ERZ) according to their [website](https://github.com/frodobots-org/earth-rovers-sdk). You can setup the goal pose at line 219 and 220. Note that we use our own GPS instead of the mounted GPS on Frodobot to conduct more reliable evaluation with accurate localization.
    ```
    cd ./deployment/utils/
    python LogoNav_frodobot.py"
    ```
   
## Citing
```
@inproceedings{hirose2025mbra,
  title     = {Learning to Drive Anywhere with Model-based Reannotation},
  author    = {Noriaki Hirose and Lydia Ignatova and Kyle Stachowicz, Catherine Glossop and Sergey Levine and Dhruv Shah},
  booktitle = {arXiv},
  year      = {2025},
  url       = {https://arxiv.org/abs/xxxxxxxx}
}
```
