# Learning to Drive Anywhere via Model-Based Reannotation
[![arXiv](https://img.shields.io/badge/arXiv-2407.08693-df2a2a.svg)](https://arxiv.org/pdf/2407.08693)
[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://learning-language-navigation.github.io)


[Noriaki Hirose](https://sites.google.com/view/noriaki-hirose/)<sup>1, 2</sup>, [Lydia Ignatova](https://www.linkedin.com/in/lydia-ignatova)<sup>1</sup>, [Kyle Stachowicz](https://kylesta.ch/)<sup>1</sup>, [Catherine Glossop](https://www.linkedin.com/in/catherineglossop/)<sup>1</sup>, [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/)<sup>1</sup>, [Dhruv Shah](https://robodhruv.github.io/)<sup>1, 3</sup>

<sup>1</sup> UC Berkeley (_Berkeley AI Research_),  <sup>2</sup> Toyota Motor North America,  <sup>3</sup> Princeton University

We present LeLaN, a novel method leverages foundation models to label in-the-wild video data with
language instructions for object navigation. We train an object navigation policy on this data, result-
ing in state-of-the-art performance on challenging zero-shot language-conditioned object navigation
task across a wide variety of indoor and outdoor environments.


![](media/teaser.png)


### Installation
Please down load our code and install some tools for making a conda environment to run our code. We recommend to run our code in the conda environment, although we do not mention the conda environments later.

1. Download the repository on your PC:
    ```
    git clone https://github.com/NHirose/Learning-to-Drive-Anywhere-via-MBRA.git
    ```
2. Set up the conda environment:
    ```
    cd Learning-to-Drive-Anywhere-via-MBRA
    conda env create -f environment_mbra.yml
    ```
3. Source the conda environment:
    ```
    conda activate mbra
    ```
4. Install the lelan packages:
    ```
    pip install -e train/
    ```
5. Install the `lerobot` package from this [repo](https://github.com/huggingface/lerobot):
    ```
    git clone https://github.com/huggingface/lerobot.git
    ``` 

6. Download the mode weights from this [link](https://drive.google.com/file/d/1qsBVYfes8wE6HFbfBv30srLeoTtjai4l/view?usp=sharing)

7. Download the map files from this [link](https://drive.google.com/file/d/1woJCPmk75qH7EIkMctyMMBAQDCphveqS/view?usp=sharing) (Note that we can not publish this map files due to the copyright.)

8. Edit the yaml files to make a path for all datasets

9. Train imitation-MBRA
    ```
    python train.py -c ./config/MBRA_il.yaml
    ```
10. Train MPC-MBRA
    ```
    python train.py -c ./config/MBRA_exaug.yaml
    ```
11. Train LogoNav with imitation-MBRA
    ```
    python train.py -c ./config/LogoNav_il_label.yaml
    ```
12. Train LogoNav with MPC-MBRA
    ```
    python train.py -c ./config/LogoNav_exaug_label.yaml
    ```
13. Train LogoNav using satellite map image with MPC-MBRA
    ```
    python train.py -c ./config/LogoNav_exaug_label_multi_task.yaml
    ```
    
## Citing
Our main project
```
@inproceedings{hirose2024lelan,
  title     = {LeLaN: Learning A Language-conditioned Navigation Policy from In-the-Wild Video},
  author    = {Noriaki Hirose and Catherine Glossop and Ajay Sridhar and Oier Mees and Sergey Levine},
  booktitle = {8th Annual Conference on Robot Learning},
  year      = {2024},
  url       = {https://arxiv.org/abs/xxxxxxxx}
}
```
Robotic navigation dataset: GO Stanford 2
```
@inproceedings{hirose2018gonet,
  title={Gonet: A semi-supervised deep learning approach for traversability estimation},
  author={Hirose, Noriaki and Sadeghian, Amir and V{\'a}zquez, Marynel and Goebel, Patrick and Savarese, Silvio},
  booktitle={2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={3044--3051},
  year={2018},
  organization={IEEE}
}
```
Robotic navigation dataset: GO Stanford 4
```
@article{hirose2019deep,
  title={Deep visual mpc-policy learning for navigation},
  author={Hirose, Noriaki and Xia, Fei and Mart{\'\i}n-Mart{\'\i}n, Roberto and Sadeghian, Amir and Savarese, Silvio},
  journal={IEEE Robotics and Automation Letters},
  volume={4},
  number={4},
  pages={3184--3191},
  year={2019},
  publisher={IEEE}
}
```
Robotic navigation dataset: SACSoN(HuRoN)
```
@article{hirose2023sacson,
  title={Sacson: Scalable autonomous control for social navigation},
  author={Hirose, Noriaki and Shah, Dhruv and Sridhar, Ajay and Levine, Sergey},
  journal={IEEE Robotics and Automation Letters},
  year={2023},
  publisher={IEEE}
}
```

