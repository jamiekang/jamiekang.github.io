---
layout: post
title: "Domain-Adversarial Training of Neural Networks"
use_math: true
date: 2017-06-05 12:12:10 +0900
tags: [pr12, paper, machine-learning, dann] 
published: true
---

이번 논문은 2016년 JMLR에서 발표된 ["Domain-Adversarial Training of Neural Networks"](https://arxiv.org/abs/1505.07818)입니다.

이 논문은
training time과 test time의 data distribution이 다른 경우,
domain adaptation을 효과적으로 할 수 있는 새로운 접근 방법을 제시합니다.

## Domain Adaptation ##

[*Domain Adaptation (DA)*](https://en.wikipedia.org/wiki/Domain_adaptation)은
training distribution과 test distribution 간에 차이가 있을 때
classifier 또는 predictor의 학습 문제를 다루는 연구 분야입니다.
즉, 
*source* (training time)과 *target* (test time) 사이의 mapping을 통해
source domain에서 학습한 classifier가 
target domain에서도 효과적으로 동작하는 것을 목표로 합니다.
DA는 
아래 그림에서 보는 것처럼
*Transfer Learning*에 속하며,
source domain에서만 labeled data가 존재하는 경우를 다룹니다.

![Taxonomy]({{ site.baseurl }}/media/2017-06-05-domain-adversarial-training-of-neural-networks-taxonomy.jpg)

DA의 이론적 배경은 2006년 S. Ben-David의 논문 ["Analysis of Representations for Domain Adaptation"](http://webee.technion.ac.il/people/koby/publications/nips06.pdf)에 기반하고 있습니다.

이 논문에서 풀려고 하는 문제는,
input space $X$에서 가능한 label의 집합인 $$Y = \{ 0,\cdots, L-1 \}$$로의 classification task입니다.
이 때 *source domain*과 
*target domain*을 
각각 
$\mathcal{D}_S$와 
$\mathcal{D}_T$로 정의합니다.

이 논문에서 제안하는 알고리즘의 목표는
target domain $$\mathcal{D}_T$$의 label에 대한 정보가 없더라도
target risk $R_{\mathcal{D}_T}(\eta)$가 낮도록
classifier $$\eta: X\rightarrow Y$$를 만드는 것입니다.

$$
\begin{align}
R_{\mathcal{D}_T}(\eta)=\Pr_{(x,y)\sim\mathbb{D}_T}\left( \eta(x) \neq y \right)
\end{align}
$$

먼저, 두 도메인 간의 거리는 
아래 식과 같이 
$\mathcal{H}$-divergence로
계산할 수 있습니다.

$$
\begin{align}
d_{\mathcal{H}}(\mathcal{D}_S^X,\mathcal{D}_T^X) = 2\sup_{\eta\in\mathcal{H}}\left| \Pr_{\mathbf{x}\sim \mathcal{D}_S^X}\left[ \eta(\mathbf{x})=1\right] - \Pr_{\mathbf{x}\sim \mathcal{D}_T^X}\left[ \eta(\mathbf{x})=1\right]\right|
\end{align}
$$

여기서
$\mathcal{H}$가 symmetric하다고 가정하면
*empirical $\mathcal{H}$-divergence*는 아래 식과 같이 계산됩니다.

$$
\begin{align}
\hat{d}_{\mathcal{H}}(S,T) = 2\left(1-\min_{\eta\in\mathcal{H}} \left[\frac{1}{n}\sum_{i=1}^n I \left[ \eta(x_i)=1\right] + \frac{1}{n'}\sum_{i=n+1}^N I \left[ \eta(x_i)=0\right]\right]\right)
\end{align}
$$

그런데, 
일반적으로 이 값을 정확하게 계산하는 것이 어렵기 때문에
아래의 식으로 근사하고 
*Proxy A Distance (PAD)*라고 부릅니다.
이후 이 논문의 실험들에서는 이 PAD 값을 사용합니다.

$$
\begin{align}
\hat{d}_{\mathcal{A}} = 2\left(1-2\epsilon \right)
\end{align}
$$

여기서 $\epsilon$은 classification error입니다.
즉, sample의 출처가 source domain인지 target domain인지
classifier가 정확히 구분할 수 있으면 $\epsilon = 0$ 입니다.

S. Ben-David의 논문에서
target risk $R_{\mathcal{D}_T}(\eta)$의
upper bound를 아래 식과 같이 계산했습니다.

$$
\begin{align} 
R_{\mathcal{D}_T}(\eta) \leq R_{S}(\eta) + \sqrt{\frac{4}{n}(d\log\frac{2e \, n}{d}+\log\frac{4}{\delta})} + \hat{d}_{\mathcal{H}}(S,T) + 4 \sqrt{\frac{1}{n}( d\log\frac{2 n}{d}+\log{4}{\delta})}+ \beta
\end{align}
$$

복잡해 보이지만,
요약하자면 결국 target risk $$R_{\mathcal{D}_T}(\eta)$$을 줄이려면
source risk $$R_{\mathcal{D}_S}(\eta)$$와
domain 간의 distance $\hat{d}_{\mathcal{H}}(S,T)$를 
모두 줄여야 하는 것을 알 수 있습니다.


## Domain-Adversarial Neural Networks (DANN) ##

앞의 수식들의 의미를 정리하면 이렇습니다.
도메인이 달라지더라도 충분히 일반화할 수 있도록 모델을 학습하려면,
source domain에서의 classifier 성능을 높이면서
한편 domain을 구분하는 성능은 낮아지게 훈련해야한다는 것입니다.

즉, 다른 말로 하면
label classifier의 loss를 minimize하면서
동시에 
domain classifier의 loss를 maximize하도록
optimize하는 문제를 푸는 것이 되기 때문에 
이 논문에서 *adversarial*이라고 표현하고 있습니다.

이 논문에서 제안하는 DANN의 구조는 다음과 같습니다.

![Figure 1]({{ site.baseurl }}/media/2017-06-05-domain-adversarial-training-of-neural-networks-fig1.jpg)

그림은 크게 <span style="color:green">green</span> 색의 <span style="color:green">*feature extractor*</span>와 
<span style="color:blue">blue</span> 색의 <span style="color:blue">*label predictor*</span>,
<span style="color:red">red</span> 색의 <span style="color:red">*domain classifier*</span>로 구성되어 있습니다.
앞에서 설명한 것처럼 
domain을 구분하는 성능을 낮추기 위해
추가된 부분이 <span style="color:red">domain classifier</span>인데,
앞 단의 <span style="color:green">feature extractor</span>와 
<span style="color:black">*gradient reversal layer*</span> (<span style="color:black">black</span>)를 통해 연결되는 것을 볼 수 있습니다.

일반적인 neural network에서는 
backpropagation을 통해 
prediction loss를 줄이는 방향으로 gradient를 계산하는데,
DANN에서는 <span style="color:red">domain classifier</span>가 prediction을 더 못하게 하려는 것이 목적이므로
<span style="color:black">gradient reversal layer</span>에서 negative constant를 곱해
부호를 바꿔 전달하는 것입니다.

아래는 더 보기 편하게 정리된 유재준 님의 그림입니다.
![Architecture 2]({{ site.baseurl }}/media/2017-06-05-domain-adversarial-training-of-neural-networks-arch2.jpg)
(그림 출처: 유재준 님의 슬라이드 ["Domain-Adversarial Training of Neural Networks"](https://www.slideshare.net/thinkingfactory/pr12-dann-jaejun-yoo))

이 구조를 간단한 SGD로 구현한 알고리즘은 다음과 같습니다.

![Algorithm 1]({{ site.baseurl }}/media/2017-06-05-domain-adversarial-training-of-neural-networks-algorithm1.jpg)

GRAAL-Research의 GitHub ["GRAAL-Research/domain_adversarial_neural_network"](https://github.com/GRAAL-Research/domain_adversarial_neural_network)와
유재준 님의 GitHub ["jaejun-yoo/shallow-DANN-two-moon-dataset"](https://github.com/jaejun-yoo/shallow-DANN-two-moon-dataset)에
각각 ``python``과 ``MATLAB``으로 구현된 코드가 있으니 참고하시기 바랍니다.


## Experiments ##

이 논문에서는
앞에서 보인 알고리즘을 
*inter-twinning moons* 2D problem라고 하는 
초승달 모양의 distribution을 가지는 dataset에 적용하고 그 결과를 보입니다.

아래 그림에서
<span style="color:red">red</span> 색의 <span style="color:red">upper moon</span>이 source distribution의 label 1이고,
<span style="color:green">green</span> 색의 <span style="color:green">lower moon</span>이 source distribution의 label 0입니다.
<span style="color:black">black</span> 색의 target distribution은 source distribution을 35도 회전시키고 label을 제거해서 만들었습니다.

![Figure 2]({{ site.baseurl }}/media/2017-06-05-domain-adversarial-training-of-neural-networks-fig2.jpg)

위 그림의 첫 번째 "Label Classification" 컬럼을 보면,
(a) 일반 NN의 경우 target sample (특히 D 부분)을 완전히 분리하고 있지 못하지만
(b) DANN은 훨씬 잘 분리하고 있는 것을 볼 수 있습니다.

또한,
위 그림의 세 번째 "Domain Classification" 컬럼을 보면,
(a) 일반 NN의 경우도 source와 target을 잘 분리하지 못하지만
(b) DANN은 훨씬 더 구분하지 못하는(이 논문에서 원하는 대로) 것을 확인할 수 있습니다.

다음은 MNIST와 SVHN 데이터셋을 사용한 실험 결과를 보여주는 그림입니다.

![Figure 5]({{ site.baseurl }}/media/2017-06-05-domain-adversarial-training-of-neural-networks-fig5.jpg)

그림에서 
<span style="color:blue">blue</span> 색은 <span style="color:blue">source domain</span>의 example이고,
<span style="color:red">red</span> 색의 <span style="color:red">target domain</span>의 example을 보여줍니다.
(a) DA를 거치기 전에는 두 색깔이 분리되어 있는 반면,
(b) 거친 후에는 분리되지 않고 잘 섞여 있는 것을 확인할 수 있습니다.

-- *[Jamie](http://twitter.com/JiyangKang);*
<br>
<iframe width="560" height="315" src="https://www.youtube.com/embed/n2J7giHrS-Y?list=PLlMkM4tgfjnJhhd4wn5aj8fVTYJwIpWkS" frameborder="0" allowfullscreen></iframe>
<br>

**References**

- Yaroslav Ganin의 논문 ["Domain-Adversarial Training of Neural Networks"](https://arxiv.org/abs/1505.07818)
- Pascal Germain의 슬라이드 ["Domain-Adversarial Neural Networks"](http://www.di.ens.fr/~germain/talks/nips2014_dann_slides.pdf)
- GRAAL-Research의 GitHub ["GRAAL-Research/domain_adversarial_neural_network"](https://github.com/GRAAL-Research/domain_adversarial_neural_network)
- 유재준 님의 슬라이드 ["Domain-Adversarial Training of Neural Networks"](https://www.slideshare.net/thinkingfactory/pr12-dann-jaejun-yoo)
- 유재준 님의 YouTube [동영상](https://youtu.be/n2J7giHrS-Y?list=PLlMkM4tgfjnJhhd4wn5aj8fVTYJwIpWkS)
- 유재준 님의 블로그 ["초짜 대학원생의 입장에서 이해하는 Domain-Adversarial Training of Neural Networks (DANN) (1)"](http://jaejunyoo.blogspot.com/2017/01/domain-adversarial-training-of-neural.html)
- 유재준 님의 GitHub ["jaejun-yoo/shallow-DANN-two-moon-dataset"](https://github.com/jaejun-yoo/shallow-DANN-two-moon-dataset)
- 유재준 님의 GitHub ["jaejun-yoo/tf-dann-py35"](https://github.com/jaejun-yoo/tf-dann-py35)
- 엄태웅 님의 YouTube [동영상](https://youtu.be/h8tXDbywcdQ)
- Wikipedia의 [Domain adaptation](https://en.wikipedia.org/wiki/Domain_adaptation)
- Shai Ben-David의 논문 ["Analysis of Representations for Domain Adaptation"](http://webee.technion.ac.il/people/koby/publications/nips06.pdf)
- Shai Ben-David의 논문 ["A Theory of Learning from Different Domains"](http://www.alexkulesza.com/pubs/adapt_mlj10.pdf)
