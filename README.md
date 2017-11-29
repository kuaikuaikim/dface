<div align=center>
<img src="http://affluent.oss-cn-hangzhou.aliyuncs.com/html/images/dface_logo.png" width="350">
</div>

-----------------
# DFace • [![License](http://pic.dface.io/apache2.svg)](https://opensource.org/licenses/Apache-2.0) [![gitter](http://pic.dface.io/gitee.svg)](https://gitter.im/cmusatyalab/DFace)


| **`Linux CPU`** | **`Linux GPU`** | **`Mac OS CPU`** | **`Windows CPU`** |
|-----------------|---------------------|------------------|-------------------|
| [![Build Status](http://pic.dface.io/pass.svg)](http://pic.dface.io/pass.svg) | [![Build Status](http://pic.dface.io/pass.svg)](http://pic.dface.io/pass.svg) | [![Build Status](http://pic.dface.io/pass.svg)](http://pic.dface.io/pass.svg) | [![Build Status](http://pic.dface.io/pass.svg)](http://pic.dface.io/pass.svg) |


**Free and open source face detection and recognition with
deep learning. Based on the MTCNN and ResNet Center-Loss**

[中文版　README](https://github.com/kuaikuaikim/DFace/blob/master/README_zh.md)

**DFace** is an open source software for face detection and recognition. All features implemented by the **[pytorch](https://github.com/pytorch/pytorch)** (the facebook deeplearning framework). With PyTorch, we use a technique called reverse-mode auto-differentiation, which allows developer to change the way your network behaves arbitrarily with zero lag or overhead.
DFace inherit these advanced characteristic, that make it dynamic and ease code review.

DFace support GPU acceleration with NVIDIA cuda. We highly recommend you use the linux GPU version.It's very fast and extremely realtime.

Our inspiration comes from several research papers on this topic, as well as current and past work such as [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878) and face recognition topic [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)

**MTCNN Structure**　　

![mtcnn](http://affluent.oss-cn-hangzhou.aliyuncs.com/html/images/mtcnn_st.png)

**If you want to contribute to DFace, please review the CONTRIBUTING.md in the project.We use [GitHub issues](https://github.com/DFace/DFace/issues) for
tracking requests and bugs.**

## Installation

DFace has two major module, detection and recognition.In these two, We provide all tutorials about how to train a model and running.
First setting a pytorch and cv2. We suggest Anaconda to make a virtual and independent python envirment.

### Requirements
* cuda 8.0
* anaconda
* pytorch
* torchvision
* cv2
* matplotlib

Also we provide a anaconda environment dependency list called environment.yml in the root path. 
You can create your DFace environment very easily.
```shell
conda env create -f path/to/environment.yml
```

### Face Detetion

If you are interested in how to train a mtcnn model, you can follow next step.

#### Train mtcnn Model
MTCNN have three networks called **PNet**, **RNet** and **ONet**.So we should train it on three stage, and each stage depend on previous network which will generate train data to feed current train net, also propel the minimum loss between two networks.
Please download the train face **datasets** before your training. We use **[WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)** and **[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)**


* Generate PNet Train data and annotation file

```shell
python src/prepare_data/gen_Pnet_train_data.py --dataset_path {your dataset path} --anno_file {your dataset original annotation path}
```
* Assemble annotation file and shuffle it

```shell
python src/prepare_data/assemble_pnet_imglist.py
```

* Train PNet model


```shell
python src/train_net/train_p_net.py
```
* Generate RNet Train data and annotation file

```shell
python src/prepare_data/gen_Rnet_train_data.py --dataset_path {your dataset path} --anno_file {your dataset original annotation path} --pmodel_file {yout PNet model file trained before}
```
* Assemble annotation file and shuffle it

```shell
python src/prepare_data/assemble_rnet_imglist.py
```

* Train RNet model

```shell
python src/train_net/train_r_net.py
```

* Generate ONet Train data and annotation file

```shell
python src/prepare_data/gen_Onet_train_data.py --dataset_path {your dataset path} --anno_file {your dataset original annotation path} --pmodel_file {yout PNet model file trained before} --rmodel_file {yout RNet model file trained before}
```

* Generate ONet Train landmarks data

```shell
python src/prepare_data/gen_landmark_48.py
```

* Assemble annotation file and shuffle it

```shell
python src/prepare_data/assemble_onet_imglist.py
```

* Train ONet model

```shell
python src/train_net/train_o_net.py
```

#### Test face detection
```shell
python test_image.py
```    

### Face Recognition  

TODO  


## Demo  

![mtcnn](http://affluent.oss-cn-hangzhou.aliyuncs.com/html/images/dface_demo.png)  


## License

[Apache License 2.0](LICENSE)


## Reference

* [Seanlinx/mtcnn](https://github.com/Seanlinx/mtcnn)
