# CiVOS : Revisiting Click-Based Interactive Video Object Segmentation

Stephane Vujasinovic, Sebastian Bullinger, Stefan Becker, Norbert Scherer-Negenborn, Michael Arens, Rainer Stiefelhagen

### Paper

[[arXiv]](https://arxiv.org/abs/2203.01784) [[Paper PDF]](https://arxiv.org/pdf/2203.01784.pdf)

### Setting up the environment
1. The framework is built with Python 3.7 and relies on the following packages:
   - NumPy `1.21.4`
   - SciPy `1.7.2`
   - PyTorch `1.10.0`
   - torchvision `0.11.1`
   - OpenCV `4.5.4` (also opencv-python-headless)
   - Cython `0.29.24`
   - scikit-image  `0.18.3`
   - Pillow `8.4.0`
   - imgaug `0.4.0`
   - albumentations `1.10`
   - tqdm     `4.62.3`
   - PyYaml   `6.0`
   - easydict `1.9`
   - future   `0.18.2`
   - cffi     `1.15.0`
   - davis-interactive `1.0.4` (<https://github.com/albertomontesg/davis-interactive>)
   - networkx `2.6.3` for DAVIS
   - gdown `4.2.0` for downloading pretrained models 
2. Download the DAVIS dataset `download_dataset.py`
3. Download the pretrained models `download_models.py`

### Guide for Demo
1. Adapt the paths and variables in `Demo.yml`
2. Launch `CiVOS_Demo.py` (*Nota bene: only 1 object can be segmented in the Demo*)
3. Mouse and keyboard bindings:
    - Positive interaction: `left mouse click`
    - Negative interaction: `right mouse click`
    - Predict a mask of the object of interest for the video sequence: `space bar`
    - Visualize the results with the keys `x`(forward direction) and `y`(backward direction) 
    - Quit the demo with key `q`

### How to evaluate on DAVIS
7. CiVOS_for_DAVIS.py
8. CiVOS_pipeline.py





### Credits

**RiTM**: [GitHub](https://github.com/hkchengrex/MiVOS), [Paper](https://arxiv.org/pdf/2103.07941.pdf)

**MiVOS**: [GitHub](https://github.com/saic-vul/ritm_interactive_segmentation), [Paper](https://arxiv.org/pdf/2103.07941.pdf)

**DeepLabV3Plus**: [GitHub](https://github.com/VainF/DeepLabV3Plus-Pytorch), [Paper](https://arxiv.org/pdf/1802.02611.pdf)

**DAVIS-interactive**: [GitHub](https://github.com/albertomontesg/davis-interactive), [Project](https://interactive.davischallenge.org/)
