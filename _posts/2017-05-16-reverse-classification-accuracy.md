---
layout: post
title: Reverse Classification Accuracy
use_math: true
date: 2017-05-16 09:29:10 +0900
tags: [pr12, paper, machine-learning, cnn] 
published: false
---

이번 논문은 Microsoft Research Asia에서 2017년 3월에 공개한 ["Deformable Convolutional Networks"](https://arxiv.org/abs/1703.06211)입니다.

오늘 소개할 논문은 "Reverse Classification Accuracy: Predicting Segmentation Performance in the Absence of Ground Truth" (https://arxiv.org/abs/1702.034074, IEEE TRANSACTIONS ON MEDICAL IMAGING, 2017)입니다.제목 그대로 ground truth 데이터가 없을 때 어떻게 performance를 측정할까에 대한 하나의 방법론 제시입니다.

ground truth 데이터가 없을 때의 성능에 관한 문제는 단순히 성능 평가에 대한 문제만은 아닙니다. 어떤 데이터베이스나 새로운 도메인 데이터를 판단할 때, 현재 방식이 그 데이터에 대해 실패하지 않고 평가하고 있냐라는 문제의 답이기도 하기 때문입니다. 이미 작년 Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning (https://arxiv.org/abs/1506.021422) 논문을 통해서 deep learning의 uncertainty에 관한 문제가 한번 이슈가 되었기에 다들 아실거 같습니다.

이 논문의 기본 아이디어는 reverse testing 방법에 기반하고 있습니다. 만약 저희가 학습 데이터에 대해서는 확인할 Ground truth가 있고, test data에 대해서는 없는 상황입니다. 그렇다면 reverse testing 방법은 test에서 생성된 prediction 값을 이용해 새로운 classifier을 생성해 이 것을 training 셋에 적용하여 확인하는 방법입니다. 그러니깐 test 데이터와 test 데이터로 생성된 prediction 결과가 하나의 reverse classifier의 training data가 되는 것이고, 그리고 원래 training data가 그 reverse classifier의 validation data정도로 활용되는 것입니다. 논문에서는 사용된 가정은 그렇게 reverse testing에서의 reverse classification accuracy가 좋으면 원래 classification 성능도 좋다라는 가정입니다. 실험에서 그 correlation 값을 구해서 보여주고 있고요.

논문의 실험 결과를 살펴보시면, 제한적이긴 하지만 특정 알고리즘을 reverse classifier의 학습 알고리즘으로 선택시 reverse classification accuracy와 실제 accuracy와 상관관계가 유의할만한 수준으로 나오고 있었습니다.

학습할 데이터가 없는 것도 문제지만, 알고리즘을 제대로 평가할 데이터가 없는 것도 문제입니다. 또한 새로운 도메인에 이미 있는 알고리즘을 적용할 때 이 알고리즘이 실패할지를 판단할 근거가 애매한 경우가 있고, 특히 의료영상에서는 이런 일들이 중요한 경우가 많습니다. 그럴 때 이런 방법도 한번 고려하여 생각해 보시면 좋을듯 합니다. 그리고 이 방법을 하나의 데이터를 분류하는 방법으로 바꿔 생각하면 데이터를 평가하는 좋은 메져로써도 바꾸어 연구할 수 있을거 같습니다.

이 논문의 저자들은, [**CNN** (Convolutional Neural Network)](https://en.wikipedia.org/wiki/Convolutional_neural_network)이 (지금까지 image 처리 분야에서 많은 성과를 거뒀지만) 근본적으로 한계가 있다고 주장합니다.
CNN에서 사용하는 여러 연산(convolution, pooling, RoI pooling 등)이 기하학적으로 일정한 패턴을 가정하고 있기 때문에 복잡한 transformation에 유연하게 대처하기 어렵다는 것입니다.
저자들은 그 예로 CNN layer에서 사용하는 [receptive field](https://www.quora.com/What-is-a-receptive-field-in-a-convolutional-neural-network)의 크기가 항상 같고, object detection에 사용하는 feature를 얻기 위해 사람의 작업이 필요한 점 등을 들고 있습니다.

이를 해결하기 위해 이 논문에서는 **Deformable Convolution**과 **Deformable ROI Pooling**이라는 두 가지 방법을 제안합니다.

## Deformable Convolution ##

Deformable Convolution은 아래 그림처럼 convolution에서 사용하는 sampling grid에 2D offset을 더한다는 아이디어에서 출발합니다.

![sampling grids]({{ site.baseurl }}/media/2017-04-16-deformable-convolutional-networks-fig1.png)

그림 (a)의 초록색 점이 일반적인 convolution의 sampling grid입니다. 
여기에 offset을 더해(초록색 화살표) (b)(c)(d)의 푸른색 점들처럼 다양한 패턴으로 변형시켜 사용할 수 있습니다.

아래 그림은 $3 \times 3$ deformable convolution의 예를 보이고 있습니다.

![3x3 deformable convolution]({{ site.baseurl }}/media/2017-04-16-deformable-convolutional-networks-fig2.png)

그림에서 보는 것처럼 deformable convolution에는 일반적인 convolution layer 말고 하나의 convolution layer가 더 있습니다. 그림에서 conv라는 이름이 붙은 이 초록색 layer가 각 입력의 2D offset을 학습하기 위한 것입니다. 
여기서 offset은 integer 값이 아니라 fractional number이기 때문에 0.5 같은 소수 값이 가능하며, 실제 계산은 linear interpolation (2D이므로 bilinear interpolation)으로 이뤄집니다. 

Training 과정에서, output feature를 만드는 convolution kernel과 offset을 정하는 convolution kernel을 동시에 학습할 수 있습니다.

![receptive field illustration]({{ site.baseurl }}/media/2017-04-16-deformable-convolutional-networks-fig5.png)

위의 그림은 convolution filter의 sampling 위치를 보여주는 예제입니다. 붉은 점은 deformable convolution filter에서 학습한 offset을 반영한 sampling location이며, 초록색 사각형은 filter의 output 위치입니다. 일정하게 샘플링 패턴이 고정되어 있지 않고, 큰 object에 대해서는 receptive field가 더 커진 것을 확인할 수 있습니다.

## Deformable ROI Pooling ##

[RoI (Region of Interest) pooling](https://deepsense.io/region-of-interest-pooling-explained/)은 크기가 변하는 사각형 입력 region을 고정된 크기의 feature로 변환하는 과정입니다.

![3x3 deformable RoI pooling]({{ site.baseurl }}/media/2017-04-16-deformable-convolutional-networks-fig3.png)

Deformable RoI pooling도 일반적인 RoI pooling layer와 offset을 학습하기 위한 layer로 구성됩니다. 
한 가지 deformable convolution과 다른 점은, offset을 학습하는 부분에 convolution이 아니라 *fc (fully-connected) layer*를 사용한 것인데 아쉽게도 그 이유가 논문에 밝혀져 있지 않습니다. 
Neural network에서 convolutional layer와 fully-connected layer의 차이에 대해서는 [Reddit의 관련 post](https://www.reddit.com/r/MachineLearning/comments/3yy7ko/what_is_the_difference_between_a_fullyconnected/)를 참고하시기 바랍니다.

마찬가지로 training 과정에서 offset을 결정하는 fc layer도 backpropagation을 통해 학습됩니다.

아래 그림은 노란색 입력 RoI에 대해 붉은색 deformable RoI pooling 결과를 보여줍니다. 
이 실험 결과에서, RoI에 해당하는 붉은 사각형의 모양이 object 형태에 따라 다양한 형태로 변형되는 것을 볼 수 있습니다. 

![roi pooling illustration]({{ site.baseurl }}/media/2017-04-16-deformable-convolutional-networks-fig6.png)

지금까지 deep learning 분야의 많은 연구들이 predictor의 weight 값 $w$를 구하는 방법에 초점을 맞췄던 반면, 이 논문은 어떤 데이터 $x$를 뽑을 것인가에 초점을 맞췄다는 점이 [참신하다는 평가](https://www.reddit.com/r/MachineLearning/comments/60kr4t/r_deformable_convolutional_networks_from_msra/)를 받고 있습니다. 이제 갓 발표된 논문인 만큼, 향후 다른 연구에 어떤 영향을 미칠지 앞으로 주목할 필요가 있을 것 같군요.


<iframe width="560" height="315" src="https://www.youtube.com/embed/jbnjzyJDldA" frameborder="0" allowfullscreen></iframe>

<br>
-- *[Jamie](http://twitter.com/JiyangKang);*

**References**

- Vanya V. Valindria의 논문 ["Reverse Classification Accuracy: Predicting Segmentation Performance in the Absence of Ground Truth"](https://arxiv.org/abs/1702.03407)
- 정동준 님의 동영상 ["PR-008: Reverse Classification Accuracy"](https://youtu.be/jbnjzyJDldA)
