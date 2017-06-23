---
layout: post
title: Neural Architecture Search with Reinforcement Learning
use_math: true
date: 2017-06-19 10:29:10 +0900
tags: [pr12, paper, machine-learning, reinforcement-learning, rnn] 
published: true
---

오늘 소개하려는 논문은 Google Brain에서 ICLR 2017에 발표한 
["Neural Architecture Search with Reinforcement Learning"](https://arxiv.org/abs/1611.01578)입니다. 

이 논문은 Google의 AutoML에 대한 정보를 주는 논문이라고 할 수 있습니다.
즉, 딥러닝을 만드는 딥러닝에 대한 재미있는 내용입니다.
새로운 neural network을 설계하고 튜닝하는 것이 어려운데,
학습을 통해서 자동화할 수 있을까에 대한 첫 시도라는 점에서 의의가 있습니다.

## Introduction ##

![Figure 1]({{ site.baseurl }}/media/2017-06-19-neural-architecture-search-with-reinforcement-learning-fig1.jpg)

이 논문에서는 
새로운 구조를 gradient 기반으로 찾는 
*Neural Architecture Search*라는 방법을 제안합니다.

이 연구는
neural network의 structure와 connectivity를 
가변 길이의 configuration string으로 지정한다는 관찰에서 시작됩니다.
예를 들어, 
``Caffe``에서는 아래와 같은 형태의 string을 사용해서 
한 layer의 구조를 정해줍니다.
```
[“Filter Width: 5”, “Filter Height: 3”, “Num Filters: 24”]
```

String의 처리에는 RNN을 적용하는 것이 일반적이므로,
여기서도
RNN(**"Controller"**)을 사용해
그와 같은 configuration string을 generation하도록 합니다.
그렇게 
만들어진 네트워크(**"Child Network"**)의 성능을 validation set에서 측정하고,
결과로 얻은 accuracy를 *reinforcement learning*의 *reward*로 사용해
Child Network의 parameter를 update합니다.


## Methods ##

아래는 이 연구에서 사용한 방법을 한 장으로 요약한 그림입니다.

![Slide 7]({{ site.baseurl }}/media/2017-06-19-neural-architecture-search-with-reinforcement-learning-slide7.jpg)
(그림 출처: 서기호 님의 슬라이드 ["Neural Architecture Search with Reinforcement Learning"](https://www.slideshare.net/KihoSuh/neural-architecture-search-with-reinforcement-learning-76883153))

RNN으로 convolutional architecture를 만드는 단순한 방법부터 살펴보겠습니다.
이 RNN은 만들어진 architecture의 expected accuracy를 maximize하기 위해
policy gradient method로 training됩니다.
또한, 
skip connection (*ResNet*에서 사용한)으로 모델의 구조를 복잡하게 하고
parameter server를 도입해 분산 training의 속도를 높이는 시도에 대해서도
뒤에서 다시 설명 드리겠습니다.


### Generate Model Descriptions with a Controller RNN ###

RNN Controller는 neural network의 hyperparameter들을
token sequence를 만들어내는 방식으로 생성합니다.
아래 그림에서,
filter height, filter width, stride height, stride width, number of filters 등의
한 layer의 parameter를 생성하는 네트워크가 
전체 layer 수 만큼 반복되는 구조인 것을 볼 수 있습니다.

![Figure 2]({{ site.baseurl }}/media/2017-06-19-neural-architecture-search-with-reinforcement-learning-fig2.jpg)



### Training with REINFORCE ###

이 논문에서 적용한 reinforcement learning 알고리즘은
Ronald J. Williams의 [REINFORCE](http://incompleteideas.net/sutton/williams-92.pdf) 입니다.

REINFORCE를 사용한 이유는 
가장 간단하고 다른 방법들에 비해 튜닝하기 쉽기 때문이라고 합니다.
이 연구에서 하려는 실험의 스케일이 대단히 크다는 점을 감안한 것 같습니다.

사용한 loss function은 
non-differentiable한 reward signal $R$을 사용해
아래의 첫 식과 같이 표현됩니다.
Gradient인 아래의 두 번째 식은
계산의 편의와 너무 높은 variance를 줄이기 위해 전개를 거쳐
실제로는 마지막의 식으로 계산됩니다.

![Eq. 1]({{ site.baseurl }}/media/2017-06-19-neural-architecture-search-with-reinforcement-learning-eq1.jpg)
![Eq. 2]({{ site.baseurl }}/media/2017-06-19-neural-architecture-search-with-reinforcement-learning-eq2.jpg)

이 논문의 실험의 수백 개의 CPU 또는 GPU를 사용하기 때문에 분산 학습 구조를 사용합니다.
아래 그림과 같이,
$S$개의 shard로 나눠진 parameter server에서
받아온 parameter에 따라
$K$개의 controller는 
각각 $m$개 씩의 child architecture를 병렬로 실행합니다.
각 child의 accuracy는 기록되고 parameter server로 다시 전송됩니다.


![Figure 3]({{ site.baseurl }}/media/2017-06-19-neural-architecture-search-with-reinforcement-learning-fig3.jpg)

### Increase Architecture Complexity with Skip Connections and Other Layer Types ###

GoogleNet, Residual Net에서 사용한
skip connection 구조를 적용할 수 있도록,
아래 그림에 보는 것처럼 
각 layer에 기준점이 되는 *Anchor Point*를 도입했습니다.

![Figure 4]({{ site.baseurl }}/media/2017-06-19-neural-architecture-search-with-reinforcement-learning-fig4.jpg)

### Generate Recurrent Cell Architectures ###

이번에는, 
LSTM과 유사한 RNN의 기본 연산 단위인 cell을 생성해봤습니다.
RNN의 기본 cell은 input, output 뿐만 아니라 memory state로 구성되는데,
이 모델은 tree 형태로 일반화할 수 있습니다.
아래 그림은 왼쪽의 가장 간단한 형태의 tree 모델을 
Neural Architecture Search를 거쳐
오른쪽의 기본적 unit들의 조합으로 만들어낸 예제입니다.
즉, LSTM이나 GRU와 같은 RNN의 cell을 
이렇게 자동으로 만들어 내는 것이 가능합니다.

![Figure 5]({{ site.baseurl }}/media/2017-06-19-neural-architecture-search-with-reinforcement-learning-fig5.jpg)


## Experiments and Results ##

이 논문에서는 두 가지 실험 결과를 보여주고 있습니다.
하나는 [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)을 사용한 image classification이고,
다른 하나는 [Penn Treebank](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)를 사용한 language modeling 입니다.

### Learning Convolutional Architectures for CIFAR-10

이 실험은 filter height, filter width, number of filters를 controller RNN이 찾는 문제입니다. 기본적인 search space는 convolutional architecture, ReLU, batch normalization, skip connection으로 구성되어 있습니다.

이 실험에서 controller RNN은 35개의 hidden unit으로 구성된 2계층의 LSTM 구조이고 ADAM optimizer를 사용했습니다. 
분산 학습을 위해 무려 800개의 GPU ($S$:20, $K$:100, $m$:8)를 사용했다고 합니다.

아래 표의 실험 결과에서, 맨 아래 부분 Neural Architecture Search로 만든 구조들이 DenseNet을 비롯한 인간이 설계한 state-of-the-art에 근접하는 성능을 보이는 것을 볼 수 있습니다.

![Table 1]({{ site.baseurl }}/media/2017-06-19-neural-architecture-search-with-reinforcement-learning-table1.jpg)

위 표의 맨 아래 부분에서 "Neural Architecture Search v1 no stride or pooling"로 표시된 결과가 아래 그림과 같습니다.
이 구조는 15개의 layer만으로 구성되어 있지만 비교적 뛰어난 성능을 보이고 있습니다.

![Figure 7]({{ site.baseurl }}/media/2017-06-19-neural-architecture-search-with-reinforcement-learning-fig7.jpg)

### Learning Recurrent Cells for Penn Treebank

다음 실험은
language modeling task에 적용할 RNN cell을 생성하는 것입니다.
앞에서 보인 Figure 5에서 보인 것과 같이,
[*add*, *elem_mult*] 등의 combination method와
[*identity*, *tanh*, *sigmoid*, *relu*] 등의 activation method를 
조합해서 tree의 각 node를 표현하도록 합니다.

실험에는 400개의 CPU를 사용했고, 총 15,000개의 child network을 만들어 평가했다고 합니다. 실험 결과로 만들어진 RNN cell은 LSTM 대비 0.5 BLEU의 향상을 보였습니다.

아래 그림은 기본 LSTM과 이 실험의 결과로 만든 2가지 버전의 RNN cell을 보입니다.

![Figure 8]({{ site.baseurl }}/media/2017-06-19-neural-architecture-search-with-reinforcement-learning-fig8.jpg)

지금까지 
RNN을 사용해 neural network 구조를 만드는 Neural Architecture Search에 대해 설명 드렸습니다.
자동으로 neural network의 구조를 탐색하는 이런 연구가 향후 어떻게 발전해 나갈지 기대가 큽니다. 

이 논문에서 사용한 코드는 GitHub [TensorFlow Models](https://github.com/tensorflow/models) 페이지에 공개될 예정이며,
앞에서 만든 RNN cell은 [NASCell](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/NASCell)이라는 이름으로 TensorFlow API에 추가되었습니다.

-- *[Jamie](http://twitter.com/JiyangKang);*
<br>
<iframe width="560" height="315" src="https://www.youtube.com/embed/XP3vyVrrt3Q?list=PLlMkM4tgfjnJhhd4wn5aj8fVTYJwIpWkS" frameborder="0" allowfullscreen></iframe>
<br>

**References**
- Barret Zoph의 논문 ["Neural Architecture Search with Reinforcement Learning"](https://arxiv.org/abs/1611.01578)
- Quoc Le와 Barret Zoph의 슬라이드 ["Neural Architecture Search with Reinforcement Learning"](http://rll.berkeley.edu/deeprlcourse/docs/quoc_barret.pdf)
- 서기호 님의 슬라이드 ["Neural Architecture Search with Reinforcement Learning"](https://www.slideshare.net/KihoSuh/neural-architecture-search-with-reinforcement-learning-76883153)
- Ronald J. Williams의 논문 ["Simple statistical gradient-following algorithms for connectionist reinforcement learning"](http://incompleteideas.net/sutton/williams-92.pdf)
- the morning paper 블로그의 [논문 요약](https://blog.acolyer.org/2017/05/10/neural-architecture-search-with-reinforcement-learning/)
- Carlos E. Perez의 article ["Taxonomy of Methods for Deep Meta Learning"](https://medium.com/intuitionmachine/machines-that-search-for-deep-learning-architectures-c88ae0afb6c8)
- CIFAR-10 dataset의 [공식 웹 페이지](https://www.cs.toronto.edu/~kriz/cifar.html)
- Penn Treebank 프로젝트의 [공식 웹 페이지](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)
- GitHub [TensorFlow Models](https://github.com/tensorflow/models)
- TensorFlow [tf.contrib.rnn.NASCell](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/NASCell)




