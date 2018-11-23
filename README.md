<div align=center>
<a href="http://dface.tech" target="_blank"><img src="http://dftech.oss-cn-hangzhou.aliyuncs.com/web/DFACE-logo_dark.png" width="350"></a>
</div>

-----------------
# DFace (Deeplearning Face) • [![License](http://dftech.oss-cn-hangzhou.aliyuncs.com/opendface/img/apache_2.svg)](https://opensource.org/licenses/Apache-2.0)


| **`Linux CPU`** | **`Linux GPU`** | **`Mac OS CPU`** | **`Windows CPU`** |
|-----------------|---------------------|------------------|-------------------|
| [![Build Status](http://dftech.oss-cn-hangzhou.aliyuncs.com/opendface/img/build_pass.svg)](http://dftech.oss-cn-hangzhou.aliyuncs.com/opendface/img/build_pass.svg) | [![Build Status](http://dftech.oss-cn-hangzhou.aliyuncs.com/opendface/img/build_pass.svg)](http://dftech.oss-cn-hangzhou.aliyuncs.com/opendface/img/build_pass.svg) | [![Build Status](http://dftech.oss-cn-hangzhou.aliyuncs.com/opendface/img/build_pass.svg)](http://dftech.oss-cn-hangzhou.aliyuncs.com/opendface/img/build_pass.svg) | [![Build Status](http://dftech.oss-cn-hangzhou.aliyuncs.com/opendface/img/build_pass.svg)](http://dftech.oss-cn-hangzhou.aliyuncs.com/opendface/img/build_pass.svg) |


**Free and open source face detection and recognition with
deep learning. Based on the MTCNN and ResNet Center-Loss**

[中文版　README](https://github.com/kuaikuaikim/DFace/blob/master/README_zh.md)  

[码云项目地址](https://gitee.com/kuaikuaikim/dface)  

**[Slack address](https://dfaceio.slack.com/)**


**DFace** is an open source software for face detection and recognition. All features implemented by the **[pytorch](https://github.com/pytorch/pytorch)** (the facebook deeplearning framework). With PyTorch, we use a technique called reverse-mode auto-differentiation, which allows developer to change the way your network behaves arbitrarily with zero lag or overhead.
DFace inherit these advanced characteristic, that make it dynamic and ease code review.

DFace support GPU acceleration with NVIDIA cuda. We highly recommend you use the linux GPU version.It's very fast and extremely realtime.

Our inspiration comes from several research papers on this topic, as well as current and past work such as [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878) and face recognition topic [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)

**MTCNN Structure**　　

![Pnet](http://dftech.oss-cn-hangzhou.aliyuncs.com/opendface/img/pnet.jpg)
![Rnet](http://dftech.oss-cn-hangzhou.aliyuncs.com/opendface/img/rnet.jpg)
![Onet](http://dftech.oss-cn-hangzhou.aliyuncs.com/opendface/img/onet.jpg)

**If you want to contribute to DFace, please review the CONTRIBUTING.md in the project.We use [Slack](https://dfaceio.slack.com/) for tracking requests and bugs. Also you can following the QQ group 681403076 or my wechat jinkuaikuai005**


## TODO(contribute to DFace)
- Based on cener loss or triplet loss implement the face conpare. Recommended Model is ResNet inception v2. Refer this [Paper](https://arxiv.org/abs/1503.03832) and [FaceNet](https://github.com/davidsandberg/facenet)
- Face Anti-Spoofing, distinguish from face light and texture。Recomend with the LBP algorithm and SVM.
- 3D mask  Anti-Spoofing.
- Mobile first with caffe2 and c++.
- Tensor rt migration.
- Docker support, gpu version

## Installation

DFace has two major module, detection and recognition.In these two, We provide all tutorials about how to train a model and running.
First setting a pytorch and cv2. We suggest Anaconda to make a virtual and independent python envirment.**If you want to train on GPU,please install Nvidia cuda and cudnn.**

### Requirements
* cuda 8.0
* anaconda
* pytorch
* torchvision
* cv2
* matplotlib  


```shell
git clone https://github.com/kuaikuaikim/DFace.git
```


Also we provide a anaconda environment dependency list called environment.yml (windows please use environment-win64.yml,Mac environment_osx.yaml) in the root path. 
You can create your DFace environment very easily.
```shell
cd DFace

conda env create -f path/to/environment.yml
```

Add DFace to your local python path  

```shell
export PYTHONPATH=$PYTHONPATH:{your local DFace root path}
```


### Face Detetion and Recognition

If you are interested in how to train a mtcnn model, you can follow next step.

#### Train mtcnn Model
MTCNN have three networks called **PNet**, **RNet** and **ONet**.So we should train it on three stage, and each stage depend on previous network which will generate train data to feed current train net, also propel the minimum loss between two networks.
Please download the train face **datasets** before your training. We use **[WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)** and **[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)**  .WIDER FACE is used for training face classification and face bounding box, also CelebA is used for face landmarks. The original wider face annotation file is matlab format, you must transform it to text. I have put the transformed annotation text file into [anno_store/wider_origin_anno.txt](https://github.com/kuaikuaikim/DFace/blob/master/anno_store/wider_origin_anno.txt). This file is related to the following parameter called  --anno_file.


* Create the DFace train data temporary folder, this folder is involved in the following parameter --dface_traindata_store 

```shell
mkdir {your dface traindata folder}
```   


* Generate PNet Train data and annotation file

```shell
python dface/prepare_data/gen_Pnet_train_data.py --prefix_path {annotation file image prefix path, just your local wider face images folder} --dface_traindata_store  {dface train data temporary folder you made before }  --anno_file ｛wider face original combined  annotation file, default anno_store/wider_origin_anno.txt}
```
* Assemble annotation file and shuffle it

```shell
python dface/prepare_data/assemble_pnet_imglist.py
```
* Train PNet model

```shell
python dface/train_net/train_p_net.py
```
* Generate RNet Train data and annotation file

```shell
python dface/prepare_data/gen_Rnet_train_data.py --prefix_path {annotation file image prefix path, just your local wider face images folder} --dface_traindata_store {dface train data temporary folder you made before } --anno_file ｛wider face original combined  annotation file, default anno_store/wider_origin_anno.txt} --pmodel_file {your PNet model file trained before}
```
* Assemble annotation file and shuffle it

```shell
python dface/prepare_data/assemble_rnet_imglist.py
```
* Train RNet model

```shell
python dface/train_net/train_r_net.py
```
* Generate ONet Train data and annotation file

```shell
python dface/prepare_data/gen_Onet_train_data.py --prefix_path {annotation file image prefix path, just your local wider face images folder} --dface_traindata_store {dface train data temporary folder you made before } --anno_file ｛wider face original combined  annotation file, default anno_store/wider_origin_anno.txt} --pmodel_file {your PNet model file trained before} --rmodel_file {your RNet model file trained before}
```
* Generate ONet Train landmarks data

```shell
python dface/prepare_data/gen_landmark_48.py
```
* Assemble annotation file and shuffle it

```shell
python dface/prepare_data/assemble_onet_imglist.py
```
* Train ONet model

```shell
python dface/train_net/train_o_net.py
```

#### Test face detection  
**If you don't want to train,i have put onet_epoch.pt,pnet_epoch.pt,rnet_epoch.pt in model_store folder.You just try test_image.py**

```shell
python test_image.py
```    

### Face Comparing  

TODO  


## Demo  

![mtcnn](http://dftech.oss-cn-hangzhou.aliyuncs.com/opendface/img/dface_demoall.PNG)  


### QQ交流群  
![](http://dftech.oss-cn-hangzhou.aliyuncs.com/opendface/img/dfaceqqsm.png)


#### 681403076  

#### 本人微信(wechat)

![](http://affluent.oss-cn-hangzhou.aliyuncs.com/html/images/perqr.jpg) 


## License

[Apache License 2.0](LICENSE)


## Reference

* [OpenFace](https://github.com/cmusatyalab/openface)
