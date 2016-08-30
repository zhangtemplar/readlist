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
