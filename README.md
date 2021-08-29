# LFR-CBT: Learning Fusion-modality Representations with CB-Transformer for Human Multimodal Emotion Recognition from Unaligned Multimodal Sequences

> Pytorch implementation for Learning Fusion-modality Representations with CB-Transformer for Human Multimodal Emotion Recognition from Unaligned Multimodal Sequences.

## Overview

### Overall Architecture

In this paper, we propose a network to learn fusion-modality representations with CB-Transformer (LFR-CBT) for human multimodal emotion recognition from unaligned multimodal sequences. Specifically, we first perform feature extraction for the three modalities respectively to obtain the local structure of the sequences. We design a novel transformer with cross-modal blocks (CB-Transformer) that enables complementary learning of different modalities, mainly divided into local temporal learning, cross-modal feature fusion and global self-attention representations. Then, we splice the modality-fused features with the original features to classify the emotions of the sequences. Finally, we conduct word-aligned and unaligned experiments on three mainstream multimodal emotion recognition datasets, IEMOCAP, CMU-MOSI, and CMU-MOSEI. The experimental results show the superiority and efficiency of our proposed method in both settings. Moreover, compared with the existing state-of-the-art methods, our model achieves the minimum number of parameters.

<img src="./assets/framework.png" alt="image-20210829114602811" style="zoom:80%;" />

### Datasets

Data files (containing processed MOSI, MOSEI and IEMOCAP datasets) can be downloaded from [here](https://www.dropbox.com/sh/hyzpgx1hp9nj37s/AAB7FhBqJOFDw2hEyvv2ZXHxa?dl=0).

To retrieve the meta information and the raw data, please refer to the [SDK for these datasets](https://github.com/A2Zadeh/CMU-MultimodalSDK).

### Run the Code

1. Create (empty) folders for data and pre-trained models:
~~~~
mkdir data pre_trained_models
~~~~

and put the downloaded data in 'data/'.

2. Command as follows
~~~~
python main.py [--FLAGS]
~~~~

Note that the defualt arguments are for unaligned version of MOSEI. For other datasets, please refer to Supplmentary.

### If Using CTC

Transformer requires no CTC module. However, as we describe in the paper, CTC module offers an alternative to applying other kinds of sequence models (e.g., recurrent architectures) to unaligned multimodal streams.

If you want to use the CTC module, plesase install warp-ctc from [here](https://github.com/baidu-research/warp-ctc).

The quick version:
~~~~
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc
mkdir build; cd build
cmake ..
make
cd ../pytorch_binding
python setup.py install
export WARP_CTC_PATH=/home/xxx/warp-ctc/build
~~~~

### Result

<img src="./assets/result.png" alt="image-20210829114952315" style="zoom: 50%;" />

## Reference

+ Note that some codes references [MulT](https://github.com/yaohungt/Multimodal-Transformer)


