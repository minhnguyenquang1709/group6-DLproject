# Group 6

# Members
22BI13264 - Nguyễn Thành Long
22BI13317 - Vũ Tuấn Minh
22BI13301 - Nguyễn Quang Minh
22BI13307 - Phan Nguyễn Tuấn Minh
22BI13259 - Hoàng Long
22BI13318 - Nguyễn Hà My

# Study of the Transformer based GAN for image super-resolution

# Abstract

Image super-resolution focuses on converting low-resolution images into high-resolution ones, useful in fields like object recognition and medical imaging. Traditional methods use GANs with convolutional neural networks (CNNs), but CNNs struggle with capturing global information. Transformers, which use self-attention, have proven effective in various tasks. We introduce SRTransGAN, a transformer-based GAN, for image super-resolution. Our model uses a novel transformer encoder-decoder as the generator to create 2x and 4x larger images. The discriminator is based on vision transformers, distinguishing real from synthesized images.

# Citation

This work is based on https://github.com/nbaghel777/SRTransGAN

@inproceedings{baghel2022srtransgan,
title={SRTransGAN: Image Super-Resolution using Transformer based Generative Adversarial Network},
author={Neeraj Baghel and Satish Singh and Shiv Ram Dubey},
year={2022}
}

# Related Works:

1. Restormer: Efficient Transformer for High-Resolution Image Restoration (CVPR 2022)
2. ViTGAN: Training GANs with Vision Transformers
