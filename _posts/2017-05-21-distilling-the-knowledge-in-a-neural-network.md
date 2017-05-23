---
layout: post
title: Distilling the Knowledge in a Neural Network
use_math: true
date: 2017-05-21 21:42:34 +0900
tags: [pr12, paper, machine-learning] 
---

이 논문은 [Geoffrey Hinton](https://www.cs.toronto.edu/~hinton/) 교수님이 2015년에 발표한 [논문](https://arxiv.org/abs/1503.02531)입니다.

논문의 내용에 들어가기 전에, 먼저 아래와 같은 간단한 개념을 이해하는 것이 도움이 됩니다. 
Google과 같은 큰 회사에서 machine learning으로 어떤 서비스를 만든다고 가정한다면, 개발 단계에 따라 training에 사용되는 모델과 실제 서비스로 deploy되는 모델에 차이가 있을 수 밖에 없을 것입니다. 
즉, training에 사용되는 모델은 대규모 데이터를 가지고 batch 처리를 할 수 있고, 리소스를 비교적 자유롭게 사용할 수 있으며 최적화를 위해 비슷한 여러 변종이 존재할 수 있습니다.
반면, 실제 deployment 단계의 모델은 데이터의 실시간 처리가 필요하고 리소스에 제약을 받으며 빠른 처리가 중요합니다.

이 두가지 단계의 모델을 구분하는 것이 이 논문에서 중요한데, 그때 그때 다른 이름으로 부르기 때문에 논문 읽기가 쉽지 않습니다. 편의를 위해 아래에서 저는 그냥 1번 모델, 2번 모델로 부르겠습니다.

|#|model| stage 	| structure|characteristic|meaning|
|:-:|:----------:|:----------:|:-------------:|:------:|:------:|
|1|"cumbersome model"|training|large ensemble|slow, complex 	|teacher|
|2|"small model"|deployment|single small|fast, compact|student|

머신 러닝 알고리즘의 성능을 올리는 아주 쉬운 방법은, 많은 모델을 만들어 같은 데이터로 training한 다음, 그 prediction 결과 값을 average하는 것입니다. 이 많은 모델의 집합을 *ensemble*이라고 부르며, 위의 표에서 1번 모델에 해당합니다.

하지만 실제 서비스에서 사용할 모델은 2번이므로 어떻게 1번의 training 결과를 2번에게 잘 가르치느냐 하는 문제가 생깁니다.

이 논문의 내용을 한 문장으로 말하자면, 1번 모델이 축적한 지식(*"dark knowledge"*)을 2번 모델에 효율적으로 전달하는 (기존 연구보다 더 general한) 방법에 대한 설명이라고 할 수 있습니다.

이 논문은 크게 두 부분으로 나뉘어집니다.
- Model Compression: 1번 ensemble 모델의 지식을 2번 모델로 전달하는 방법
- Specialist Networks: 작은 문제에 특화된 모델들을 training시켜 ensemble의 training 시간을 단축하는 방법

## Model Compression ##

핵심 아이디어는 1번 모델이 training되고 난 다음, 그 축적된 지식을 *distillation*이라는 새로운 training을 사용해서 2번 모델에 전달하겠다는 것입니다.
비슷한 아이디어가 2006년에 Caruana 등에 의해 [발표](https://www.cs.cornell.edu/~caruana/compression.kdd06.pdf)된 적이 있는데, Hinton의 이 논문에서는 그 연구가 자신들이 주장하는 방법의 special case임을 보입니다.

1번 모델로 training을 하게 되면 주어진 데이터 셋에 대한 결과 prediction의 분포를 얻을 수 있습니다. 
Neural network에서 마지막 단에서는 대개 *softmax* function을 사용하기 때문에, 가장 확률이 높은 경우가 1로, 나머지는 0으로 마치 *one-hot encoding*처럼 만들어진 결과를 얻게 됩니다.
저자들은 그런 형태의 결과를 *hard target*이라고 부르면서, 1과 0으로 변환되기 전의 실수 값(*soft target*)을 사용하는 것이 2번 모델로 지식을 전달하는데 효과적이라고 주장합니다.

즉, 1번 모델의 결과로 나온 soft target 분포를 목표로 2번 모델을 train시키면 1번 모델이 학습한 지식을 충분히 generalize해서 전달할 수 있다는 것입니다. 상식적으로 0/1 보다 실수 값에 정보가 많이 들어있으니 당연한 것처럼 생각되기도 합니다.

이를 위해 Hinton 교수는 temperature $T$가 parameter로 들어가는 아래와 같은 `softmax` 함수를 사용합니다. $T$를 1로 놓으면 보통 사용하는 `softmax` 함수가 됩니다.

$$
\begin{align} \tag{1}
p_i = \frac {exp(\frac{z_i}{T})} {\sum_{j} exp(\frac{z_j}{T})}
\end{align}
$$

이 식에서 분모는 확률 값을 0에서 1 사이 값으로 scaling해주기 위한 것이므로 신경 쓸 필요가 없고, 분자는 단순히 exponential 함수입니다. 
Temperature $T$ 값이 커지면(*high temperature*) exponential의 입력으로 들어가는 값이 작아지므로 결과값이 천천히 증가합니다. 
즉 `softmax` 함수를 거칠 때 큰 값이 (다른 작은 값들보다) 더 커지는 현상이 줄어들어 결과의 분포는 훨씬 부드러운(softer) distribution이 됩니다.
아래 그림은 Hinton 교수의 "Dark Knowledge"라는 발표 슬라이드에서 인용했습니다.

> ![An example of hard and soft targets]({{ site.baseurl }}/media/2017-05-21-distilling-the-knowledge-in-a-neural-network-fig1.jpg)

논문에 상세히 나와있지는 않지만, 이 방법을 MNIST에 적용했을 때 놀랄 만큼 좋은 성능을 보였다고 합니다.
특히 일부러 숫자 3을 제외하고 training했는데도 테스트에서 98.6%의 정확도를 보였다고 하는군요.

## Specialist Networks ##

1번 모델을 training할 때, 전체 data가 많으면 아무리 parallel training을 한다고 해도 많은 시간이 걸릴 수 밖에 없습니다.
이 논문에서는 혼동하기 쉬운 특별한 부분 집합에 대해서만 training하는 *specialist* 모델을 만들어 training을 효율적으로 할 수 있다고 주장합니다.
여기서 '혼동하기 쉬운 특별한 부분 집합'의 예로는, *ImageNet*에서 버섯이나 스포츠카만 모아놓은 데이터 셋을 들 수 있겠습니다.

이러한 특별한 부분 집합은 overfitting되기 쉽기 때문에, 각 specialist 모델은 일반적인 데이터 셋과 반반 섞은 데이터로 training합니다.
Training후에는 specialist 데이터가 전체에서 차지하는 비율에 맞춰 결과 값을 scaling합니다.

결론 짓자면, 이 논문의 저자들은 사용하는 모델이 아무리 크고 복잡하더라도 실제 서비스로 deploy 못할까 봐 걱정할 필요가 없다고 합니다.
1번 모델에서 knowledge를 추출해서 훨씬 작은 2번 모델로 옮길 수 있기 때문이지요. 
이 논문에서 specialist 모델에서 추출한 지식을 커다란 하나의 모델로 옮기는 방법은 나와있지 않습니다.

-- *[Jamie](http://twitter.com/JiyangKang);*
<br>
<iframe width="560" height="315" src="https://www.youtube.com/embed/tOItokBZSfU?list=PLlMkM4tgfjnJhhd4wn5aj8fVTYJwIpWkS" frameborder="0" allowfullscreen></iframe>
<br>

**References**
- Geoffrey Hinton의 [paper @arXiv.org](https://arxiv.org/abs/1503.02531) 
- Geoffrey Hinton의 슬라이드 ["Dark Knowledge"](http://www.ttic.edu/dl/dark14.pdf) 
- Bucila와 Caruana 등의 논문 ["Model Compression"](https://www.cs.cornell.edu/~caruana/compression.kdd06.pdf)
- Fang Hao의 블로그 [post](http://luofanghao.github.io/2016/07/20/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0%20%E3%80%8ADistilling%20the%20Knowledge%20in%20a%20Neural%20Network%E3%80%8B/) 
- James Chan의 GitHub [repository](https://github.com/chengshengchan/model_compression)
- Alex Korbonits의 슬라이드 ["Distilling dark knowledge from neural networks"](https://www.slideshare.net/AlexanderKorbonits/distilling-dark-knowledge-from-neural-networks)
- LU Yangyang의 슬라이드 ["Feature Transfer and Knowledge Distillation in Deep Neural Networks"](http://sei.pku.edu.cn/~luyy11/slides/slides_141231_ft_distill-nips14.pdf)
- KDnuggets의 post ["Dark Knowledge Distilled from Neural Network"](http://www.kdnuggets.com/2015/05/dark-knowledge-neural-network.html) 
