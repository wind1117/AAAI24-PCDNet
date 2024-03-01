# AAAI2024_PCDNet

## Exploiting Polarized Material Cues for Robust Car Detection

[Wen Dong](https://wind1117.github.io/), [Haiyang Mei](https://mhaiyang.github.io/), 
Ziqi Wei, Ao Jin, Sen Qiu, Qiang Zhang, [Xin Yang](https://xinyangdut.github.io/)

[[Paper](https://arxiv.org/abs/2401.02606)]
[[Project Page](https://wind1117.github.io/publication/2024-AAAI-PolarCar)]

### Abstract
Car detection is an important task that serves as a crucial prerequisite for many 
automated driving functions. The large variations in lighting/weather conditions 
and vehicle densities of the scenes pose significant challenges to existing car 
detection algorithms to meet the highly accurate perception demand for safety, 
due to the unstable/limited color information, which impedes the extraction of 
meaningful/discriminative features of cars. In this work, we present a novel 
learning-based car detection method that leverages trichromatic linear polarization 
as an additional cue to disambiguate such challenging cases. A key observation is 
that polarization, characteristic of the light wave, can robustly describe intrinsic 
physical properties of the scene objects in various imaging conditions and is 
strongly linked to the nature of materials for cars (e.g., metal and glass) and 
their surrounding environment (e.g., soil and trees), thereby providing 
***reliable*** and ***discriminative*** features for robust car detection in 
challenging scenes. To exploit polarization cues, we first construct a pixel-aligned 
RGB-Polarization car detection dataset, which we subsequently employ to train a 
novel multimodal fusion network. Our car detection network dynamically integrates 
RGB and polarization features in a request-and-complement manner and can explore the 
intrinsic material properties of cars across all learning samples. We extensively 
validate our method and demonstrate that it outperforms state-of-the-art detection 
methods. Experimental results show that polarization is a powerful cue for car 
detection.

### Citation
Please cite our paper if you find it is useful:
```
@InProceedings{Wen_2024_AAAI_PCDNet,  
    title = {Exploiting Polarized Material Cues for Robust Car Detection},  
    author = {Dong, Wen and Mei, Haiyang and Wei, Ziqi and Jin, Ao and Qiu, Sen and Zhang, Qiang and Yang, Xin},  
    booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},  
    year = {2024}  
}
```
### Requirements
- Python 3.8.18
- Pytorch 1.12.1
- Torchvision 0.13.1
- CUDA 11.6.0
- numpy
- Pillow
- PyYAML
- scipy
- tqdm

### Experiments
**Test**: Download pre-trained ``PCDNet.pt`` at 
[[Baidu Netdisk](https://pan.baidu.com/s/1Bjb6IeQuhbt1zvypJZ-cXQ?pwd=ujzb), 
fetch code: ujzb], then run ``infer.py``.

### License
Please see ``LICENSE``.