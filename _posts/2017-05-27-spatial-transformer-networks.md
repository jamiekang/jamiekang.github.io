---
layout: post
title: Spatial Transformer Networks
use_math: true
date: 2017-05-27 22:29:10 +0900
tags: [pr12, paper, machine-learning, cnn] 
published: false
---

이번 논문은 Microsoft Research Asia에서 2017년 3월에 공개한 ["Deformable Convolutional Networks"](https://arxiv.org/abs/1703.06211)입니다.

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

DeepMind의 실험 결과
<iframe width="560" height="315" src="https://www.youtube.com/embed/Ywv0Xi2-14Y" frameborder="0" allowfullscreen></iframe>

Max Jaderberg의 발표 동영상
<iframe width="560" height="315" src="https://www.youtube.com/embed/T5k0GnBmZVI" frameborder="0" allowfullscreen></iframe>

PR12의 발표 동영상

<br>
-- *[Jamie](http://twitter.com/JiyangKang);*
<br>
<iframe width="560" height="315" src="https://www.youtube.com/embed/RRwaz0fBQ0Y?list=PLlMkM4tgfjnJhhd4wn5aj8fVTYJwIpWkS" frameborder="0" allowfullscreen></iframe>

**References**

- Max Jaderberg의 논문 ["Spatial Transformer Networks"](https://arxiv.org/abs/1506.02025)
- Max Jaderberg의 발표 동영상 ["Symposium: Deep Learning - Max Jaderberg"](https://youtu.be/T5k0GnBmZVI)
- GitHub의 [Lasagne의 example](https://github.com/Lasagne/Recipes/blob/master/examples/spatial_transformer_network.ipynb)
- Tensorflow GitHub의 [Spatial Transformer Networks](https://github.com/tensorflow/models/tree/master/transformer)
- Xavier Giro의 슬라이드 ["Spatial Transformer Networks"](https://www.slideshare.net/xavigiro/spatial-transformer-networks)
- Okay Arik의 슬라이드 ["Spatial Transformer Networks"](http://web.cs.hacettepe.edu.tr/~aykut/classes/spring2016/bil722/slides/w08-transformer_network.pdf)
- 강지양 님의 슬라이드 ["Spatial Transformer Networks (2015)"](https://www.slideshare.net/???)
- 강지양 님의 동영상 ["PR-011: Spatial Transformer Networks (2015)"]()
- Kevin Nguyen의 Medium article ["Spatial Transformer Networks with Tensorflow"](https://medium.com/wonks-this-way/spatial-transformer-networks-with-tensorflow-2eaed0374a16)
- Alban Desmaison의 torch article ["The power of Spatial Transformer Networks"](http://torch.ch/blog/2015/09/07/spatial_transformers.html)
- Kevin Zakka의 blog post ["Deep Learning Paper Implementations: Spatial Transformer Networks - Part I"](https://kevinzakka.github.io/2017/01/10/stn-part1/)
- Wikipedia의 [Thin plate spline](https://en.wikipedia.org/wiki/Thin_plate_spline)
