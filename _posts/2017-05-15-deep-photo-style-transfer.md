---
layout: post
title: Deep Photo Style Transfer
use_math: true
date: 2017-05-15 09:29:10 +0900
tags: [pr12, paper, machine-learning, cnn] 
published: true
---

이번 논문은 Cornell 대학과 Adobe Research의 Fujun Luan 등이 2017년 3월에 공개한 ["Deep Photo Style Transfer"](https://arxiv.org/abs/1703.07511)입니다. 
이 논문은 2015년 큰 충격을 주었던 Leon A. Gatys의 ["A Neural Algorithm of Artistic Style"](https://arxiv.org/abs/1508.06576) 라는 논문을 painting이 아닌 photography의 관점에서 더욱 발전시킨 것입니다. 

먼저 Gatys의 논문에 대해 간략히 살펴보도록 하겠습니다.

## A Neural Algorithm of Artistic Style ##

이 논문에서는 16개의 convolutional layer와 5개의 pooling layer가 있는 [VGG-19 Network](https://arxiv.org/abs/1409.1556)을 수정해 사용합니다. 

### Content Representation ###

2015년 Gatys의 논문에 따르면, content의 loss function은 아래의 식으로 표현됩니다. (아래 식들의 Notation은 원 논문의 것이 아니라 Luan의 논문을 따랐습니다.)

$$
\begin{align} \tag{1}
\mathcal{L}_{c}^l = \frac{1}{2 N_l D_l} 
\sum_{ij} (F_{l}[O] - F_{l}[I])_{ij}^2
\end{align}
$$

이 식에서 $I$는 input image, $O$는 output image, $F_{l}[I]$, $F_{l}[O]$는 각각 layer $l$의 filter 입력과 출력입니다.

### Style Representation ###

한편, style 정보를 표현하기 위해 저자는 자신의 2015년 NIPS 논문 ["Texture Synthesis Using Convolutional Neural Networks"](https://arxiv.org/abs/1505.07376)을 인용해 [Gram matrix](https://en.wikipedia.org/wiki/Gramian_matrix)라는 개념을 도입합니다.
style (=texture) 정보를 얻는 Gram matrix $G_{l}$은 layer $l$에서 $i$번째 feature와 $j$번째 feature 간의 correlation을 계산한 것으로, 각 element는 단순한 vector inner product로 계산됩니다.

$$
\begin{align} \tag{2}
G_{ij}^l = 
\sum_{k} F_{ik}^l F_{jk}^l
\end{align}
$$

이에 따르면 style의 loss function은 아래의 식으로 표현됩니다. 아래 식에서 $S$는 style의 reference image입니다.

$$
\begin{align} \tag{3}
\mathcal{L}_{s}^l = \frac{1}{2 N_l^2}
\sum_{ij} (G_{l}[O] - G_{l}[S])_{ij}^2
\end{align}
$$

### Algorithm: Content + Style Representation ###

위의 (1), (3)식을 이용해, Gatys의 ["A Neural Algorithm of Artistic Style"](https://arxiv.org/abs/1508.06576)에서 정의한 loss function은 아래와 같이 content와 style의 loss function을 둘다 minimize하는 식이 됩니다.

$$
\begin{align} \tag{4}
\mathcal{L}_{total} = 
\sum_{l=1}^L \alpha_l \mathcal{L}_c^l 
+ \Gamma \sum_{l=1}^L \beta_l \mathcal{L}_{s}^l
\end{align}
$$

이에 따라 style transfer하는 알고리즘을 한 눈에 보인 것이 아래 그림입니다.
![Gatys Fig.2]({{ site.baseurl }}/media/2017-05-15-deep-photo-style-transfer-gatys-fig2.jpg)

이 알고리즘을 사용해, 저자들이 속한 독일 튀빙겐 대학교의 강변 사진에 5개 명화의 스타일을 적용하면 아래와 같은 결과물이 나옵니다. 

![Gatys Fig.3]({{ site.baseurl }}/media/2017-05-15-deep-photo-style-transfer-gatys-fig3.jpg)

선행 연구에 대한 소개는 이것으로 마치고, 이제 오늘 소개하려는 Luan의 ["Deep Photo Style Transfer"](https://arxiv.org/abs/1703.07511) 논문에 대해 설명 드리겠습니다.

## Deep Photo Style Transfer ##

앞에서 설명한 알고리즘을 그림이 아니라 사진에 적용하면 다소 이상한 부분이 관찰됩니다. 
아래 그림에서, 원래 사진에서 사각형이었을 건물의 창문들이 삐뚤빼뚤한 모양인 점과 하늘의 구름 부분인 좌측과 중앙 상단에 노랗게 창문 불빛이 투영된 점이 눈에 띕니다.

![Fig.1(b)]({{ site.baseurl }}/media/2017-05-15-deep-photo-style-transfer-fig1b.jpg)

이 두 가지 문제를 해결하기 위해 이 논문에서는 두 가지 기법을 제안했습니다.
그 첫 번째는 *Photorealism Regularization*이라고 하고, 두 번째는 *Augmented Style Loss with Semantic Segmentation*이라고 합니다.

### Photorealism Regularization ###

창문과 같은 부분의 찌그러짐을 줄이기 위해, *Photorealism Regularization* 기법에서는 image distortion에 penalty를 주는 loss function $\mathcal{L}_m$을 정의합니다.

이 부분에서는 Anat Levin의 ["A Closed Form Solution to Natural Image Matting"](http://www.wisdom.weizmann.ac.il/~levina/papers/Matting-Levin-Lischinski-Weiss-CVPR06.pdf)이라는 논문의 image matting이라는 개념을 사용합니다. 
*Image matting*은 쉽게 말해, 아래 그림처럼 사진에서 foreground object를 뽑아내는 것을 의미합니다.
즉, foreground object를 정확히 뽑는다는 것은 입력 원본과 출력에서 해당 object의 경계 모양이 일치한다는 것과 같다는 뜻이므로, image matting에서 사용하는 cost function을 그대로 가져와서 image distortion을 줄이는데 이용하겠다는 것입니다.

![Levin Fig.2]({{ site.baseurl }}/media/2017-05-15-deep-photo-style-transfer-levin-fig2.jpg)

이를 위해 Levin은 Matting Laplacian이라는 matrix $\mathcal{M}_I$를 정의했습니다.
다소 복잡하지만 Levin의 논문에서 소개한 식에 따라 input image $I$에서 $\mathcal{M}_I$를 계산하고, 이를 이용해 $\mathcal{L}_m$을 다음과 같이 정의합니다.

$$
\begin{align} \tag{5}
\mathcal{L}_{m} = 
\sum_{c=1}^3 V_{c}[O]^T \mathcal{M}_I V_{c}[O] 
\end{align}
$$

이 loss function $\mathcal{L}_m$을 위의 (4)식에 추가하는 것이 바로 *Photorealism Regularization* 기법입니다.

### Augmented Style Loss with Semantic Segmentation ###

사진의 하늘 부분까지 창문 부분의 스타일이 적용되는 것을 'spillover' 현상이라고 합니다. 
이 문제를 없애려면 창문에는 창문, 하늘에는 하늘의 스타일을 적용하면 된다는 것이 이 논문의 접근법입니다. 

이를 위해, 기존 ["Champandard의 연구"](https://arxiv.org/abs/1603.01768), ["Chen의 연구"](https://arxiv.org/abs/1606.00915)에서처럼 semantic segmentation method를 적용해서 image를 'sky', 'building', 'water' 등의 label 붙은 segment로 먼저 나눕니다. 
그 다음, feature map에 각 segment에 해당하는 semantic segmentation mask를 씌워서 Gram matrix를 구합니다.

이에 따라 바뀐 style의 loss function 식 $\mathcal{L}_{s+}$는 아래와 같습니다.

$$
\begin{align} \tag{6}
\mathcal{L}_{s+}^l = \sum_{c=1}^C \frac{1}{2 N_{l,c}^2}
\sum_{ij} (G_{l,c}[O] - G_{l,c}[S])_{ij}^2
\end{align}
$$

이 식에서 $C$는 segment (channel)의 개수를 의미합니다.

### Algorithm: Photorealism Regularization + Augmented Style Loss ###

이제 종합해 보겠습니다.
앞의 Gatys 논문의 loss function 식을 다시 쓰면 아래와 같습니다.

$$
\begin{align} \tag{4}
\mathcal{L}_{total} = 
\sum_{l=1}^L \alpha_l \mathcal{L}_c^l 
+ \Gamma \sum_{l=1}^L \beta_l \mathcal{L}_{s}^l
\end{align}
$$

여기에 (5)식과 (6)식을 적용하면, 전체 loss function $\mathcal{L}_{total}$은 이제 아래 식과 같이 계산됩니다.

$$
\begin{align} \tag{7}
\mathcal{L}_{total} = 
\sum_{l=1}^L \alpha_l \mathcal{L}_c^l 
+ \Gamma \sum_{l=1}^L \beta_l \mathcal{L}_{s+}^l
+ \lambda \mathcal{L}_m
\end{align}
$$

즉, 전체 loss function은 content loss + modified style loss + photorealism regularization이 됩니다.

이를 적용한 결과물은 아래 그림과 같습니다. 창문을 비롯해서 원본의 직선이 그대로 유지되고 있고, 하늘에 있던 노란 번짐도 없어졌습니다.

![Fig.1(d)]({{ site.baseurl }}/media/2017-05-15-deep-photo-style-transfer-fig1d.jpg)

<iframe width="560" height="315" src="https://www.youtube.com/embed/YF6nLVDlznE" frameborder="0" allowfullscreen></iframe>

<br>
-- *[Jamie](http://twitter.com/JiyangKang);*

**References**

- Fujun Luan의 논문 ["Deep Photo Style Transfer"](https://arxiv.org/abs/1703.07511)
- Fujun Luan과 저자들의 GitHub [repository](https://github.com/luanfujun/deep-photo-styletransfer)
- 김승일 님의 슬라이드 ["Deep Photo Style Transfer"](http://www.modulabs.co.kr/DeepLAB_library/13532)
- 김승일 님의 동영상 ["PR-007: Deep Photo Style Transfer"](https://youtu.be/YF6nLVDlznE)
- [kurzweilai.net](http://www.kurzweilai.net/)의 기사 ["A deep-learning tool that lets you clone an artistic style onto a photo"](http://www.kurzweilai.net/a-deep-learning-tool-that-lets-you-clone-an-artistic-style-onto-a-photo)
- Leon A. Gatys의 2015년 논문 ["A Neural Algorithm of Artistic Style"](https://arxiv.org/abs/1508.06576)
- Leon A. Gatys의 2015년 NIPS 논문 ["Texture Synthesis Using Convolutional Neural Networks"](https://arxiv.org/abs/1505.07376)
- Leon A. Gatys의 2016년 CVPR 논문 ["Image Style Transfer Using Convolutional Neural Networks"](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
- 전상혁 님의 블로그 ["고흐의 그림을 따라그리는 Neural Network, a Neural Algorithm of Artistic Style (2015)"](http://sanghyukchun.github.io/92/)
- Karen Simonyan의 VGG-19 논문 ["Very Deep Convolutional Networks for Large-Scale Image Recognition"](https://arxiv.org/abs/1409.1556) 
- Wikipedia의 [Gram matrix](https://en.wikipedia.org/wiki/Gramian_matrix)
- Anat Levin의 ["A Closed Form Solution to Natural Image Matting"](http://www.wisdom.weizmann.ac.il/~levina/papers/Matting-Levin-Lischinski-Weiss-CVPR06.pdf)
- Alex J. Champandard의 ["Semantic Style Transfer and Turning Two-Bit Doodles into Fine Artworks"](https://arxiv.org/abs/1603.01768)
- Liang-Chieh Chen의 ["DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs"](https://arxiv.org/abs/1606.00915)
- Mark Chang의 슬라이드 ["Neural Art"](https://www.slideshare.net/ckmarkohchang/neural-art-english-version)
