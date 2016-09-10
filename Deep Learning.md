# Tensorflow Multi-GPU VAE-GAN implementation

This is an [implementation](https://github.com/timsainb/Tensorflow-MultiGPU-VAE-GAN.git) of the VAE-GAN based on the implementation described in *<a href="http://arxiv.org/abs/1512.09300">Autoencoding beyond pixels using a learned similarity metric</a>*

The abstract reads: We present an autoencoder that leverages learned representations to better measure similarities in data space. By combining a variational autoencoder with a generative adversarial network we can use learned feature representations in the GAN discriminator as basis for the VAE reconstruction objective. Thereby, we replace element-wise errors with feature-wise errors to better capture the data distribution while offering invariance towards e.g. translation. We apply our method to images of faces and show that it outperforms VAEs with element-wise similarity measures in terms of visual fidelity. Moreover, we show that the method learns an embedding in which high-level abstract visual features (e.g. wearing glasses) can be modified using simple arithmetic.

We have three networks, an  <font color="#38761d"><strong>Encoder</strong></font>,
a <font color="#1155cc"><strong>Generator</strong></font>, and a <font color="#ff0000"><strong>Discriminator</strong></font>. 
    - The <font color="#38761d"><strong>Encoder</strong></font> learns to map input x onto z space (latent space)
    - The <font color="#1155cc"><strong>Generator</strong></font> learns to generate x from z space
    - The <font color="#ff0000"><strong>Discriminator</strong></font> learns to discriminate whether the image being put in is real, or generated

![vae gan outline](https://github.com/timsainb/Tensorflow-MultiGPU-VAE-GAN/blob/master/network_outline.png?raw=true)


# Unsupervised Learning for Physical Interaction through Video Prediction

A core challenge for an agent learning to interact with the world is to predict how its actions affect objects in its environment. Many existing methods for learning the dynamics of physical interactions require labeled object information. However, to scale real-world interaction learning to a variety of scenes and objects, acquiring labeled data becomes increasingly impractical. To learn about physical object motion without labels, we develop an action-conditioned video prediction model that explicitly models pixel motion, by predicting a distribution over pixel motion from previous frames. Because our model explicitly predicts motion, it is partially invariant to object appearance, enabling it to generalize to previously unseen objects. To explore video prediction for real-world interactive agents, we also introduce a dataset of 50,000 robot interactions involving pushing motions, including a test set with novel objects. In this dataset, accurate prediction of videos conditioned on the robot's future actions amounts to learning a "visual imagination" of different futures based on different courses of action. Our experiments show that our proposed method not only produces more accurate video predictions, but also more accurately predicts object motion, when compared to prior methods.

![Robot prediction](https://storage.googleapis.com/push_gens/gengifs0/10_55.gif)

The paper is availalbe at [http://arxiv.org/abs/1605.07157]

# SSD: Single Shot MultiBox Detector

Paper is available at [https://arxiv.org/abs/1512.02325] and code available at [https://github.com/weiliu89/caffe/tree/ssd]

![Example](https://camo.githubusercontent.com/ad9b147ed3a5f48ffb7c3540711c15aa04ce49c6/687474703a2f2f7777772e63732e756e632e6564752f7e776c69752f7061706572732f7373642e706e67)

We present a method for detecting objects in images using a single deep neural network. Our approach, named SSD, discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales per feature map location. At prediction time, the network generates scores for the presence of each object category in each default box and produces adjustments to the box to better match the object shape. Additionally, the network combines predictions from multiple feature maps with different resolutions to naturally handle objects of various sizes. Our SSD model is simple relative to methods that require object proposals because it completely eliminates proposal generation and subsequent pixel or feature resampling stage and encapsulates all computation in a single network. This makes SSD easy to train and straightforward to integrate into systems that require a detection component.

# YOLO You Only Look Once: Unified, Real-Time Object Detection

Paper is available at [http://arxiv.org/abs/1506.02640]

![Example](http://pjreddie.com/media/image/model_2.png)

YOLO uses a totally different approach. We apply a single neural network to the full image. This network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities.Finally, we can threshold the detections by some value to only see high scoring detections. One problem is that, it assumes there is only one object within each region.

# Text to Image
Tensorflow implementation of text to image synthesis using thought vectors is available at [https://github.com/paarthneekhara/text-to-image]

This is an experimental tensorflow implementation of synthesizing images from captions using [Skip Thought Vectors][1]. The images are synthesized using the GAN-CLS Algorithm from the paper [Generative Adversarial Text-to-Image Synthesis][2]. This implementation is built on top of the excellent [DCGAN in Tensorflow][3]. The following is the model architecture. The blue bars represent the Skip Thought Vectors for the captions.

![Model architecture](http://i.imgur.com/dNl2HkZ.jpg)

Image Source : [Generative Adversarial Text-to-Image Synthesis][2] Paper

# Deep Convolutional Generative Adversarial Networks
Tensorflow implementation of [Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434) which is a stabilize Generative Adversarial Networks. The referenced torch code can be found [here](https://github.com/soumith/dcgan.torch). The tensorflow implementation is availalbe at [https://github.com/carpedm20/DCGAN-tensorflow]

![alt tag](https://github.com/carpedm20/DCGAN-tensorflow/raw/master/DCGAN.png)

* [Brandon Amos](http://bamos.github.io/) wrote an excellent [blog post](http://bamos.github.io/2016/08/09/deep-completion/) and [image completion code](https://github.com/bamos/dcgan-completion.tensorflow) based on this repo.
* *To avoid the fast convergence of D (discriminator) network, G (generatior) network is updatesd twice for each D network update which is a different from original paper.*
