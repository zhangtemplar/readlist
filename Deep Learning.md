# Attention and Augmented Recurrent Neural Networks

Recurrent neural networks are one of the staples of deep learning, allowing neural networks to work with sequences of data like text, audio and video. The basic RNN design struggles with longer sequences, but a special variant – “long short-term memory” networks – can even work with these. Such models have been found to be very powerful, achieving remarkable results in many tasks including translation, voice recognition, and image captioning. As a result, recurrent neural networks have become very widespread in the last few years. As this has happened, we’ve seen a growing number of attempts to augment RNNs with new properties. Four directions which stand out as particularly exciting are described [here](http://distill.pub/2016/augmented-rnns/)

![attention](http://distill.pub/2016/augmented-rnns/assets/rnn_preview_ai.svg)

# Pixel-CNN and Pixel-RNN

[Pixel-CNN](https://arxiv.org/abs/1606.05328) and [Pixel-RNN](https://arxiv.org/abs/1601.06759) showed that it was possible to generate complex natural images not only one pixel at a time, but one colour-channel at a time, requiring thousands of predictions per image.

![pixel-cnn](https://ai2-s2-public.s3.amazonaws.com/figures/2016-03-25/8e4ab54564fb492dcae9a1e862aedd3e52fb258b/3-Figure2-1.png)

# Wavenet

[Wavenet](https://deepmind.com/blog/wavenet-generative-model-raw-audio/) is a deep generative model of raw audio waveforms. WaveNets are able to generate speech which mimics any human voice and which sounds more natural than the best existing Text-to-Speech systems, reducing the gap with human performance by over 50%. The same network can be used to synthesize other audio signals such as music, and present some striking samples of automatically generated piano pieces. The network models the conditional probability to generate the next sample in the audio waveform, given all previous samples and possibly additional parameters. A tensor-flow based implementation can be found [here](https://github.com/ibab/tensorflow-wavenet)

![model](https://storage.googleapis.com/deepmind-live-cms.google.com.a.appspot.com/documents/BlogPost-Fig2-Anim-160908-r01.gif)

# Neural Network Zoo

With new neural network architectures popping up every now and then, it’s hard to keep track of them all. Knowing all the abbreviations being thrown around (DCIGN, BiLSTM, DCGAN, anyone?) can be a bit overwhelming at first. Asimov Institute composed a cheat sheet containing many of those architectures. Most of these are neural networks, some are completely different beasts. Though all of these architectures are presented as novel and unique, when we check the node structures… their underlying relations started to make more sense.

![model](http://www.asimovinstitute.org/wp-content/uploads/2016/09/neuralnetworks.png)

# UberNet

In this work we introduce a convolutional neural network (CNN) that jointly handles low-, mid-, and high-level vision tasks in a unified architecture that is trained end-to-end: (a) boundary detection (b) normal estimation © saliency estimation (d) semantic segmentation  (e) human part segmentation (f)  semantic boundary detection, (g) region proposal generation and object detection. We obtain competitive performance while jointly addressing all of these tasks in 0.7 seconds per frame on a single GPU.

The demo can be found [here](http://cvn.ecp.fr/ubernet/) and the paper [here](http://arxiv.org/abs/1609.02132)
![demo](http://cvn.ecp.fr/ubernet/im3.png)

# End-to-End Deep Learning for Self-Driving Cars

In a new automotive application, we have used convolutional neural networks (CNNs) to map the raw pixels from a front-facing camera to the steering commands for a self-driving car. This powerful end-to-end approach means that with minimum training data from humans, the system learns to steer, with or without lane markings, on both local roads and highways. The system can also operate in areas with unclear visual guidance such as parking lots or unpaved roads.

Read more [here](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) and the paper [here](https://arxiv.org/pdf/1604.07316v1.pdf)
![example](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/training.png)

# Deep Speech 2

[Speech Is 3x Faster than Typing for English and Mandarin Text Entry on Mobile Devices](http://hci.stanford.edu/research/speech/paper/speech_paper.pdf) by Sherry Ruan et al of Stanford & Baidu. Researchers evaluate Deep Speech 2, a deep learning-based speech recognition system, assessing that the system makes English text input 3.0X faster, and Mandarin Chinese input 2.8X faster than standard keyboard typing. The error rates were also dramatically reduced, and the results further highlight the potential & strength of speech interfaces.

# Text summarization with TensorFlow

Being able to develop Machine Learning models that can automatically deliver accurate summaries of longer text can be useful for digesting such large amounts of information in a compressed form, and is a long-term goal of the Google Brain team. Summarization can also serve as an interesting reading comprehension test for machines. To summarize well, machine learning models need to be able to comprehend documents and distill the important information, tasks which are highly challenging for computers, especially as the length of a document increases.

In an effort to push this research forward, we’re open-sourcing [TensorFlow model code](https://github.com/tensorflow/models/tree/master/textsum) for the task of generating news headlines on [Annotated English Gigaword](https://catalog.ldc.upenn.edu/LDC2012T21), a dataset often used in summarization research. We also specify the hyper-parameters in the documentation that achieve better than published state-of-the-art on the most commonly used metric as of the time of writing. Below we also provide samples generated by the model.

| *Input: Article 1st sentence* | *Model-written headline* |
| --- | --- |
| metro-goldwyn-mayer reported a third-quarter net loss of dlrs 16 million due mainly to the effect of accounting rules adopted this year | mgm reports 16 million net loss on higher revenue |
| starting from july 1, the island province of hainan in southern china will implement strict market access control on all incoming livestock and animal products to prevent the possible spread of epidemic diseases | hainan to curb spread of diseases |
| australian wine exports hit a record 52.1 million liters worth 260 million dollars (143 million us) in september, the government statistics office reported on monday | australian wine exports hit record high in september|


# Learning to Segment

The main new algorithms driving our advances are the [DeepMask](https://github.com/facebookresearch/deepmask) segmentation framework coupled with our new [SharpMask](https://github.com/facebookresearch/deepmask) segment refinement module. Together, they have enabled FAIR's machine vision systems to detect and precisely delineate every object in an image. The final stage of our recognition pipeline uses a specialized convolutional net, which we call [MultiPathNet](https://github.com/facebookresearch/multipathnet) (Corresponding paper: [A MultiPath Network for Object Detection](http://arxiv.org/abs/1604.02135)), to label each object mask with the object type it contains (e.g. person, dog, sheep). We will return to the details shortly.

The work can be found [here](https://research.facebook.com/blog/learning-to-segment/).
![example](https://scontent.fsnc1-1.fna.fbcdn.net/t39.2365-6/14146891_1635044716824695_382934461_n.jpg)

# Tensorflow Multi-GPU VAE-GAN implementation

This is an [implementation](https://github.com/timsainb/Tensorflow-MultiGPU-VAE-GAN.git) of the VAE-GAN based on the implementation described in *<a href="http://arxiv.org/abs/1512.09300">Autoencoding beyond pixels using a learned similarity metric</a>*

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
