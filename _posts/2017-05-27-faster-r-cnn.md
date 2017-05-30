---
layout: post
title: "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
use_math: true
date: 2017-05-28 15:12:10 +0900
tags: [pr12, paper, machine-learning, cnn] 
published: true
---

이번 논문은 Microsoft Research에서 2015년 NIPS에 발표한 ["Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"](https://arxiv.org/abs/1506.01497)입니다.

이 논문은 computer vision 분야의 중요한 문제 중의 하나인 [object detection](https://en.wikipedia.org/wiki/Object_detection)을 다룹니다. 

이 논문은
Ross Girshick의 흔히 R-CNN이라고 하는 2013년 논문 ["Rich feature hierarchies for accurate object detection and semantic segmentation"](https://arxiv.org/abs/1311.2524),
그리고 2015년 논문 ["Fast R-CNN"](https://arxiv.org/abs/1504.08083)에 이어지는 연구입니다.


## Glossary ##

Object detection 계열의 논문에서 자주 나오는 용어에 대해 먼저 간단히 설명하겠습니다.

### Selective Search ###

![Selective Search]({{ site.baseurl }}/media/2017-05-27-faster-r-cnn-selective-search.png)

Uijlings의 [논문](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf)에서 제안된 이 방법은,
image에서 object의 candidate를 찾기 위한 알고리즘입니다.
Color space 정보와 다양한 similarity measure를 활용해서 복잡한 segmentation 결과를 (위의 그림의 가장 왼쪽에서 오른쪽으로) grouping합니다.
뒤에서 설명할 R-CNN의 처음 단계에서 실행되는 알고리즘이기도 합니다.

### Hard Negative Mining ###

[Hard Negative Mining](https://www.reddit.com/r/computervision/comments/2ggc5l/what_is_hard_negative_mining_and_how_is_it/)은 positive example과 negative example을 균형적으로 학습하기 위한 방법입니다.
단순히 random하게 뽑은 것이 아니라 confidence score가 가장 높은 순으로 뽑은 negative example을 (random하게 뽑은 positive example과 함께) training set에 넣어 training합니다.

![Hard Negative Mining]({{ site.baseurl }}/media/2017-05-27-faster-r-cnn-hnm.png)
(그림 출처: 한보형 님의 슬라이드 ["Lecture 6: CNNs for Detection, Tracking, and Segmentation"](http://cvlab.postech.ac.kr/~bhhan/class/cse703r_2016s/csed703r_lecture6.pdf))

### Non Maximum Suppression ###

Non Maximum Suppression은 edge thinning 기법으로, 여러 box가 겹치게 되면 가장 확실한 것만 고르는 방법입니다. 
아래 그림을 보면 바로 이해할 수 있습니다.

![Non Maximum Suppression]({{ site.baseurl }}/media/2017-05-27-faster-r-cnn-nms.jpg)
(그림 출처: PyImageSearch의 article ["Histogram of Oriented Gradients and Object Detection"](http://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/))


### Bounding Box Regression ###

Bound box의 parameter를 찾는 regression을 의미합니다.
초기의 region proposal이 CNN이 예측한 결과와 맞지 않을 수 있기 때문입니다.
[Bounding box regressor](https://www.quora.com/Convolutional-Neural-Networks-What-are-bounding-box-regressors-doing-in-Fast-RCNN)는 CNN의 마지막 pooling layer에서 얻은 feature 정보를 사용해 region proposal의 regression을 계산합니다.
뒤에서 소개할 R-CNN에서 bounding box regressor가 등장합니다.

## 2-stage Detection ##

지금부터 설명드릴 R-CNN 계열의 연구는 
모두 2-stage detection에 속합니다.

### R-CNN ###

![r-cnn concept]({{ site.baseurl }}/media/2017-05-27-faster-r-cnn-r-cnn-concept.jpg)

**R-CNN**은 CNN을 object detection에 적용한 첫 번째 연구입니다. 
위의 그림이 R-CNN 개념을 설명하는 가장 유명한 그림입니다.
그림을 차례로 살펴 보면,
(1) input image를 받아서 
(2) selective search로 2000개의 region proposal을 추출한 다음, 
(3) CNN으로 각 proposal의 feature를 계산하고
(4) 각 region의 classification 결과와 bounding box regression을 계산합니다.
Classifier로는 [SVM](https://en.wikipedia.org/wiki/Support_vector_machine)을 사용합니다.

R-CNN의 특징은 다음과 같습니다.
- Regional Proposal + CNN
- Regional proposal을 얻기 위해 selective search 사용
- CNN을 사용한 첫 번째 object detection method
- 각 proposal을 독립적으로 계산 (= 많은 계산 필요)
- Bounding box regression으로 detection 정확도 향상

R-CNN의 전체 구조를 다른 그림으로 살펴 보면 아래와 같습니다.

![r-cnn architecture]({{ site.baseurl }}/media/2017-05-27-faster-r-cnn-r-cnn-architecture.jpg)

R-CNN은 몇 가지 문제를 가지고 있습니다.
- Test 속도가 느림
	- 모든 region proposal에 대해 전체 CNN path를 다시 계산
	- GPU(K40)에서 장당 13초
	- CPU에서 장당 53초
- SVM과 bounding box regressor의 학습이 분리
	- Feature vector를 disk에 caching
	- CNN 학습 과정 후, SVN과 bounding box regressor의 학습이 나중에 진행됨(post-hoc)
- 학습 과정이 복잡함: 다단계 training pipeline
	- GPU(K40)에서 84시간

### Fast R-CNN ###

**Fast R-CNN**은 다음과 같은 특징을 가집니다.
- 같은 image의 proposal들이 convolution layer를 공유
- ROI Pooling 도입
- 전체 network이 End-to-end로 한 번에 학습
- R-CNN보다 빠르고 더 정확한 결과

![fast r-cnn concept]({{ site.baseurl }}/media/2017-05-27-faster-r-cnn-fast-r-cnn-concept.jpg)

Fast R-CNN도
처음에 initial RoI (= region proposal)를 찾는 것은 selective search를 사용합니다.
하지만 각 RoI를 매번 convolution 하는 것이 아니라,
전체 image를 한 번만 convolution 합니다.
그 결과로 나온 convolution feature map에서
RoI에 해당하는 영역을 추출해 pooling (= subsampling) 과정을 거쳐 fully connected layer에 넣는 것이 Fast R-CNN의 핵심입니다. 아래 그림은 RoI pooling 과정을 설명하고 있습니다.

![fast r-cnn roi pooling]({{ site.baseurl }}/media/2017-05-27-faster-r-cnn-fast-cnn-roi-pooling.jpg)
(그림 출처: 이진원 님의 발표 [동영상](https://youtu.be/kcPAGIgBGRs))

Fast R-CNN의 전체 구조를 다른 그림으로 살펴 보면 아래와 같습니다.
![fast-r-cnn architecture]({{ site.baseurl }}/media/2017-05-27-faster-r-cnn-fast-r-cnn-architecture.jpg)

아래는 Fast R-CNN을 R-CNN과 다시 한번 비교한 그림입니다. 
![fast-r-cnn comparison]({{ site.baseurl }}/media/2017-05-27-faster-r-cnn-fast-r-cnn-comparison.jpg)
(그림 출처: Kaiming He의 [ICCV 2015 Tutorial](http://kaiminghe.com/iccv15tutorial/iccv2015_tutorial_convolutional_feature_maps_kaiminghe.pdf))

Fast R-CNN은 R-CNN에서 training이 복잡했던 문제를 해결했지만,
여전히 아래와 같은 문제를 가지고 있습니다.
- Region proposal 계산이 neural network 밖에서 일어난다.
- Region proposal 계산(selective search)이 전체 성능의 bottleneck이 된다.

Selective search가 느린 이유 중의 하나는 GPU가 아니라 CPU로 계산하기 때문입니다.

## Faster R-CNN ##

지금부터가 오늘 소개 드리는 논문 ["Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"](https://arxiv.org/abs/1506.01497)의 내용입니다.

먼저, Faster R-CNN의 전체 구조는 아래 그림과 같습니다.

![faster-r-cnn]({{ site.baseurl }}/media/2017-05-27-faster-r-cnn-architecture.jpg)

**Faster R-CNN**은 앞에서 설명한 Fast R-CNN을 개선하기 위해
**Region Proposal Network (RPN)**을 도입합니다.
RPN은 region proposal을 만들기 위한 network입니다.
즉, Faster R-CNN에서는 외부의 느린 selective search (CPU로 계산) 대신, 내부의 빠른 RPN (GPU로 계산)을 사용합니다.
RPN은 마지막 convolutional layer 다음에 위치하고,
그 뒤에 Fast R-CNN과 마찬가지로 RoI pooling과 classifier, bounding box regressor가 위치합니다. 아래 그림은 RPN의 구조를 보입니다.

![faster-r-cnn rpn]({{ site.baseurl }}/media/2017-05-27-faster-r-cnn-rpn.jpg)
(그림 출처: Kaiming He의 [ICCV 2015 Tutorial](http://kaiminghe.com/iccv15tutorial/iccv2015_tutorial_convolutional_feature_maps_kaiminghe.pdf))

RPN은 sliding window에 $3 \times 3$ convolution을 적용해 
input feature map을 256 (ZF) 또는 512 (VGG) 크기의 feature로 mapping합니다. 
그 출력은 box classification layer (*cls*)와 box regression layer (*reg*)으로 들어갑니다. 
Box classification layer와 box regression layer는 각각
$1 \times 1$ convolution으로 구현됩니다.

Box regression을 위한 초기 값으로 *anchor*라는 pre-defined reference box를 사용합니다. 
이 논문에서는 3개의 크기와 3개의 aspect ratio를 가진 총 9개의 anchor를 각 sliding position마다 적용하고 있습니다.
아래 그림은 anchor의 개념을 보입니다.

![faster-r-cnn anchor]({{ site.baseurl }}/media/2017-05-27-faster-r-cnn-anchor.jpg)

Faster R-CNN의 실험 결과입니다.
PASCAL VOC 2007 test set을 사용한 실험에서
Faster R-CNN은 R-CNN의 250배, Fast R-CNN의 10배 속도를 내는 것을 볼 수 있습니다.
Faster R-CNN은 약 5 fps의 처리가 가능하기 때문에 
저자들은 near real-time이라고 주장합니다.

![faster-r-cnn result-cs231n]({{ site.baseurl }}/media/2017-05-27-faster-r-cnn-result-cs231n.jpg)
(그림 출처: Stanford cs231n의 Lecture 8, [ "Spatial Localization and Detection"](http://cs231n.stanford.edu/slides/2016/winter1516_lecture8.pdf))

Faster R-CNN의 특징을 정리하면 다음과 같습니다.
- RPN + Fast CNN
- Region proposal을 network 내부에서 계산
- RPN은 fully convolutional하다.
- RPN은 end-to-end로 train 된다.
- RPN은 detection network와 convolutional feature map을 공유한다.

## 그 밖의 최근 연구 ##

지금까지 설명 드린 R-CNN 계열은 2-stage detection에 해당합니다. 
2-stage detection 계열의 연구에서는, 
이 밖에도 
[SPP-net](https://arxiv.org/abs/1406.4729), 
[R-FCN](https://arxiv.org/abs/1605.06409),
[Mask R-CNN](https://arxiv.org/abs/1703.06870) 
등의 연구도 종종 인용됩니다.

한편,
2-stage detection 구조는 공통적으로 아래와 같은 단점을 가진다고 종종 지적 받습니다.

- 복잡한 pipeline
- 느리다. (real time 실행 불가능)
- 각 component를 optimize하기 어렵다.

이런 점을 극복하기 위해, 최근에는 unified detection에 해당하는 연구들이 등장하는 추세입니다.
대표적인 것으로는 [Yolo](https://arxiv.org/abs/1506.02640), [SSD (Single Shot multibox Detector)](https://arxiv.org/abs/1512.02325)가 있습니다.

Yolo
- Detection 문제를 regression 문제로 접근
- 하나의 convolution network 사용
- 전체 image를 한 번에 처리하므로 매우 빠르다.

SSD
- Yolo보다 빠르고 Faster R-CNN 만큼 정확하다.
- Category와 box offset을 prediction한다.
- Feature map에 small convolutional filter 사용
- 여러 크기의 feature map을 사용해 prediction한다.

이 논문들에 대해서는 조만간 별도의 post로 설명 드리겠습니다. 
그 밖에, [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144), [ION](https://arxiv.org/abs/1512.04143) 등의 연구도 주목받고 있습니다.

-- *[Jamie](http://twitter.com/JiyangKang);*
<br>
<iframe width="560" height="315" src="https://www.youtube.com/embed/kcPAGIgBGRs?list=PLlMkM4tgfjnJhhd4wn5aj8fVTYJwIpWkS" frameborder="0" allowfullscreen></iframe>
<br>

**References**

- Shaoqing Ren의 논문 ["Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"](https://arxiv.org/abs/1506.01497)
- J.R.R. Uijlings의 논문 ["Selective Search for Object Recognition"](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf)
- Ross Girshick의 논문 ["Rich feature hierarchies for accurate object detection and semantic segmentation"](https://arxiv.org/abs/1311.2524)
- Wikipedia의 [Support Vector Machine](https://en.wikipedia.org/wiki/Support_vector_machine)
- Ross Girshick의 논문 ["Fast R-CNN"](https://arxiv.org/abs/1504.08083)
- Kaiming He의 [ICCV 2015 Tutorial](http://kaiminghe.com/iccv15tutorial/iccv2015_tutorial_convolutional_feature_maps_kaiminghe.pdf)
- Shaoqing Ren의 GitHub ["ShaoqingRen/faster_rcnn"](https://github.com/ShaoqingRen/faster_rcnn)
- Ross Girshick의 GitHub ["rbgirshick/py-faster-rcnn"](https://github.com/rbgirshick/py-faster-rcnn)
- Ross Girshick의 GitHub ["rbgirshick/fast-rcnn"](https://github.com/rbgirshick/fast-rcnn)
- Andy Tsai의 슬라이드 ["Faster R-CNN: Towards Real-Time Object Detection"](https://www.slideshare.net/ssuser416c44/faster-rcnn)
- Xavier Giro의 슬라이드 ["Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"](https://www.slideshare.net/xavigiro/faster-rcnn-towards-realtime-object-detection-with-region-proposal-networks)
- Mathworks의 Documentation ["Object Detection Using Faster R-CNN Deep Learning"](https://kr.mathworks.com/help/vision/examples/object-detection-using-faster-r-cnn-deep-learning.html)
- 라온피플의 블로그 ["머신 러닝(Machine Learning) - Class 47 : Best CNN(Convolutional Neural Network) Architecture - ResNet/Faster R-CNN(part5)"](http://laonple.blog.me/220782324594)
- Liao Yuan-Hong의 블로그 ["Video Object Detection using Faster R-CNN"](https://andrewliao11.github.io/object_detection/faster_rcnn/)
- Wikipedia의 [Object detection](https://en.wikipedia.org/wiki/Object_detection)
- Joseph Redmon의 논문 ["You Only Look Once: Unified, Real-Time Object Detection"](https://arxiv.org/abs/1506.02640)
- Wei Liu의 논문 ["SSD: Single Shot MultiBox Detector"](https://arxiv.org/abs/1512.02325)
- Jifeng Dai의 논문 ["R-FCN: Object Detection via Region-based Fully Convolutional Networks"](https://arxiv.org/abs/1605.06409)
- Tsung-Yi Lin의 논문 ["Feature Pyramid Networks for Object Detection"](https://arxiv.org/abs/1612.03144)
- Sean Bell의 논문 ["Inside-Outside Net: Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks"](https://arxiv.org/abs/1512.04143)
- 한보형 님의 슬라이드 ["Lecture 6: CNNs for Detection, Tracking, and Segmentation"](http://cvlab.postech.ac.kr/~bhhan/class/cse703r_2016s/csed703r_lecture6.pdf)
- PyImageSearch의 article ["Histogram of Oriented Gradients and Object Detection"](http://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/)
- Stanford cs231n의 Lecture 8, [ "Spatial Localization and Detection"](http://cs231n.stanford.edu/slides/2016/winter1516_lecture8.pdf)
- Kaiming He의 논문 ["Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition"](https://arxiv.org/abs/1406.4729)
- Kaiming He의 논문 ["Mask R-CNN"](https://arxiv.org/abs/1703.06870)

