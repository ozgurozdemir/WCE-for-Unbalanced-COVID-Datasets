# Weighted Cross-Entropy for Unbalanced Data with Application on Covid X-ray Images

This repository provides the codes used for _Weighted Cross-Entropy for Unbalanced Data with Application on Covid X-ray Images_
article published in [ASYU 2020](http://asyu.inista.org/?language=EN) conference. _The paper will be shared once the conference proceedings is published._

## Disclaimer:
:bangbang: **Although the networks achieve decent performances, a professional interpretation is required for final decision.
In real life, there may have some patients who is adverserial to the samples in test set.**

# Abstract
_Since December 2019 the world is infected by COVID-19 or Coronavirus disease, which spreads very quickly, out of control. The high number of precautions for laboratory access, which need to be taken to contain the virus, together with the difficulties in running the gold standard test for COVID-19, result in a practical incapability to make early diagnosis. Recent advances in deep learning algorithms allow efficient implementation of computer-aided diagnosis. This paper investigates on the performance of a very well known residual network, ResNet50, and a lightweight Atrous CNN (ACNN) network using a Weighted Cross-entropy (WCE) loss function, to alleviate imbalance on COVID datasets. As a result, ResNet50 model initialized with pretrained weights fine-tuned by ImageNet dataset and exploiting WCE achieved the state-of-the-art performance on COVIDXRay-5K test set, with a top balanced accuracy of 99.87%._

# Datasets
To train the networks, 3 different dataset is used. The datasets and required folders & files for implementation is given
below:

* [Covid Xray 5K](https://github.com/shervinmin/DeepCovid/tree/master/data): train/test folders are required.
* [Covid ChestXray](https://github.com/ieee8023/covid-chestxray-dataset): images folder and metadata.csv are required.
* [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/): No Finding/Other Disease folders are required.

All images are resized to 256x256 and the pixel values are normalized.

# Experimental Results
The experiments are conducted on Covid5K test set. The best results are as given below

|          | Sensitivity | Specificity |  bACC  |
|----------|:-----------:|:-----------:|:------:|
| ResNet50<sup>1</sup> |    1.000    |    0.9973   | 0.9987 |
|   ACNN<sup>2</sup>   |    0.9750   |    0.9783   | 0.9767 |

<sup>1</sup> _Trained with Covid Xray 5K+Covid ChestXray, pretrained with ImageNet, exploits weighted cross-entropy with ß=0.75_

<sup>2</sup> _Trained with Covid Xray 5K+Covid ChestXray, pretrained with CheXpert, exploits weighted cross-entropy with ß=0.75_

For the rest of the experiments, please refer the paper.

# Implementation
* [```model_factory.py```](models/model_factory.py): Provides interface for the networks ACNN and Preassembled networks like ResNet50 etc.
* [```dataset_factory.py```](datasets/dataset_factory.py): Provides interface for Covid Xray 5K, Covid ChestXray and CheXpert datasets. The implementation requires NumPy compressed files; to prepare the related files, [```prepare_dataset.py```](datasets/prepare_dataset.py) should be executed.
* [```prepare_dataset.py```](datasets/prepare_dataset.py): Provides codes for preparing NumPy compressed files for datasets.
* [```evaluation_metrics.py```](evaluation_metrics.py): Provides evaluation metrics, i.e. sensitivity, specificity, bACC and PPCR.
* [```train.py```](train.py): Provides codes for loading datasets, initialization and training of networks.
* [```test.py```](test.py): Provides codes for testing trained and saved networks.

# Usage
In order to read the dataset fast and easily, NumPy compressed files (.npz) are used in implementation. For preparing the
required files following commands can be executed:

> for Covid Xray 5K test set:

>> ``` $ python prepare_dataset.py -path=./data -test=True```

> for training set of datasets:

>> ``` $ python prepare_dataset.py -path=./data -dataset=Covid5K```

The arguments can change depends on the dataset. Required files to be downloaded for preparation are given above.

In order to train and test ResNet50 network following command can be executed:
>> ``` $ python train.py -path=./data -test_path=./data/Covid5K/test/ -model=resnet50 -pretrain=imagenet -test=True -epochs=100 -wce_b=0.75``` 

In order to test a pretrained ResNet50 network following command can be executed:
>> ``` $ python test.py -test_path=./data/Covid5K/test/ -model=resnet50 -pretrain_path=./pretraining/resnet50_chexpert``` 

# Citation
_To be shared once the proceedings of the conference is published_

