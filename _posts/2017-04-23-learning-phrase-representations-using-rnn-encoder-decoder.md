---
layout: post
title: "Learning Phrase Representations Using RNN Encoder-Decoder for Statistical Machine Translation"
use_math: true
date: 2017-04-23 15:12:10 +0900
tags: [pr12, paper, machine-learning, rnn, smt] 
published: true
---

이번 논문은 2013년 NYU [조경현](http://www.kyunghyuncho.me/) 교수님이 발표하신 ["Learning Phrase Representations Using RNN Encoder-Decoder for Statistical Machine Translation"](https://arxiv.org/abs/1406.1078)입니다.

이 논문은 두 가지 내용으로 유명한데, 
하나는 LSTM의 대안으로 떠오른 [**Gated Recurrent Unit (GRU)**](https://en.wikipedia.org/wiki/Gated_recurrent_unit)의 도입이고, 
다른 하나는 기계 번역 [Neural Machine Translation (NMT)](https://en.wikipedia.org/wiki/Neural_machine_translation) 분야에서 널리 쓰이고 있는 [**sequence-to-sequence (seq2seq)**](https://www.tensorflow.org/tutorials/seq2seq) 모델의 제안입니다.

## Introduction ##

이 논문을 이해하기 위해서는 **RNN**과 **LSTM**에 대한 사전 지식이 필요합니다.

[LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory)에 대해서는 
추후 다른 [논문](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory)에서 설명드릴 계획이므로 
여기서는 자세히 언급하지 않겠습니다.
간단히 리뷰가 필요하신 분께는 유명한 Christopher Olah의 블로그 post ["Understanding LSTM Networks"](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)를 추천 드립니다.


## RNN Encoder-Decoder ##

이 논문에서는 기계 번역에서 뛰어난 성능을 보이는 새로운 neural network 모델을 제안합니다.
저자들은 *RNN Encoder-Decoder*라고 부르고 있지만,
일반적으로는 seq2seq라는 이름으로 더 잘 알려져 있습니다.

Seq2seq 모델은 아래 그림과 같이 두 개의 RNN 모델로 구성되어 있습니다.
![Figure 1]({{ site.baseurl }}/media/2017-04-23-learning-phrase-representations-using-rnn-encoder-decoder-fig1.png)

첫 번째 RNN(*encoder*)은 가변 길이의 문장을 단어 단위로 쪼개어 고정 길이의 벡터로 mapping합니다.
두 번째 RNN(*decoder*)은 인코딩된 벡터를 하나의 단어씩 디코딩하여 다시 하나의 문장으로 만듭니다.

Encoder가 input sequence $\mathbf{x}$의 각 symbol을 순차적으로 읽는 것에 따라
내부의 hidden state가 update 됩니다.
Encoder가 input sequence의 마지막 심볼 (*eos*, *end-of-sequence*)을 읽고 나면, 
hidden state는 전체 input sequence의 summary인 벡터 $\mathbf{c}$가 됩니다.

Decoder는 주어진 hidden state $$\mathbf{h}_{\langle t \rangle}$$에서
다음 symbol $y_t$를 생성하도록 train된 또 다른 RNN입니다.
그런데, decoder의 $y_{t}$와 $$\mathbf{h}_{\langle t \rangle}$$는
이전 symbol $y_{t-1}$ 뿐만 아니라 input sequence의 summary인 $\mathbf{c}$에도 의존성이 있습니다.
즉, decoder의 hidden state는 아래 식과 같이 계산됩니다. 

$$
\begin{align}
\mathbf{h}_{\langle t \rangle} = f \left( \mathbf{h}_{\langle t-1 \rangle}, y_{t-1}, \mathbf{c} \right)
\end{align}
$$

여기서 $f \left( \right)$는 non-linear activation function입니다.

이 두 개의 네트워크는 주어진 source sequence에 대해 target sequence가 나올 조건부 확률을 maximize하도록 
함께 training 됩니다.
수식으로 쓰면 아래와 같이 log-likelihood로 표현됩니다.

$$
\begin{align}
\max_{\theta}\frac{1}{N}\sum_{n=1}^{N}logP_{\theta}(\mathbf{y}_{n}|\mathbf{x}_{n})
\end{align}
$$

여기서 $\theta$는 모델의 parameter이고, 이를 추정하기 위해 gradient 기반의 알고리즘을 사용할 수 있습니다.

이 모델은 두 개의 다른 언어(예: 입력-영어, 출력-프랑스어) 간 번역에 사용할 수 있고,
챗봇과 같은 질문-답변 대화 모델로도 사용할 수 있습니다.

최근에 더 흔히 볼 수 있는 seq2seq 모델의 그림은 아래와 같은 형태입니다.

![seq2seq]({{ site.baseurl }}/media/2017-04-23-learning-phrase-representations-using-rnn-encoder-decoder-tf-seq2seq.png)
(그림 출처: TensorFlow 공식 사이트의 [Sequence-to-Sequence Models](https://www.tensorflow.org/tutorials/seq2seq))

## Gated Recurrent Unit (GRU) ##

GRU는 LSTM과 비슷한 성질을 갖지만,
구조가 더 간단해서 계산량이 적고 구현이 쉽습니다.

이 논문에서 제안하는 새로운 종류의 hidden unit (GRU)의 구조는 아래 그림과 같습니다.
이 그림은 원래 논문이 아니라, 나중에 정준영 님의 논문 ["Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling"](https://arxiv.org/abs/1412.3555)에서 다시 그린 것을 인용했습니다.

![Chung - Figure 1]({{ site.baseurl }}/media/2017-04-23-learning-phrase-representations-using-rnn-encoder-decoder-chung-fig1.jpg)

Input, forget, output gate로 구성된 LSTM과 달리, 
GRU는 **update gate** $z$와 **reset gate** $r$ 두 가지로 구성됩니다.
Update gate는 과거의 memory 값을 얼마나 유지할 것인지를 결정하고,
reset gate는 새 input과 과거의 memory 값을 어떻게 합칠지를 정합니다.

GRU는 LSTM처럼 hidden state과 분리된 별도의 memory cell을 가지지 않습니다.
다시 말하자면 외부에서 보는 hidden state 값과 내부의 메모리 값이 같습니다.
LSTM에 있었던 output gate가 없어졌기 때문입니다.

마지막으로, output을 계산할 때 non-linear 함수(``tanh()``)를 적용하지 않습니다.

GRU를 구현하기 위해 자료를 찾아보면,
계산식의 notation이 출처마다 약간씩 달라서 혼동스럽습니다.
또한 GRU의 식 자체도 나중에 정준영 님의 논문 ["Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling"](https://arxiv.org/abs/1412.3555)에서 약간 수정되기도 했습니다.
제가 찾아본 중에서 가장 그림과 식이 이해하기 쉽게 정리된 것은 역시 Christopher Olah의 블로그 post ["Understanding LSTM Networks"](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)였습니다 (아래 그림).

![Olah - GRU Figure]({{ site.baseurl }}/media/2017-04-23-learning-phrase-representations-using-rnn-encoder-decoder-olah-gru-fig.png)
<br>

## Statistical Machine Translation ##

통계 기반 기계 번역
[Statistical machine translation (SMT)](https://en.wikipedia.org/wiki/Statistical_machine_translation)는
대규모 데이터를 학습해 만든 통계적 모델을 사용해 번역하는 방법입니다.
기본적으로 단어마다 번역해서 조합하는 방식이며, 현재 가장 많이 사용되는 기계 번역 방식이기도 합니다.
언어학자 없이도 개발을 할 수 있고 데이터가 많이 쌓일수록 번역의 품질이 높아지는 장점이 있습니다.

SMT 시스템의 목표는 
주어진 source sentence $\mathbf{e}$에 대해
아래 식의 확률을 maximize하는
translation $\mathbf{f}$를 찾는 것입니다.

$$
\begin{align}
p \left( \mathbf{f} \; | \; \mathbf{e} \right) \propto p \left( \mathbf{e} \; | \; \mathbf{f} \right) p \left( \mathbf{f} \right)
\end{align}
$$

이 때, 
$p \left( \mathbf{e} \; | \; \mathbf{f} \right)$을 *translation model*,
$p \left( \mathbf{f} \right)$을 *language model*이라고 합니다.

하지만 대부분의 SMT 시스템은 아래와 같은 log-linear 모델 식으로 
대신 
$\log p \left( \mathbf{f} \; | \; \mathbf{e} \right)$를
계산합니다.

$$
\begin{align}
\log p \left( \mathbf{f} \; | \; \mathbf{e} \right) =  \sum_{n=1}^N w_n f_n \left( \mathbf{f}, \mathbf{e} \right) + \log Z \left( \mathbf{e} \right)
\end{align}
$$

여기서 $f_n$과 $w_n$은 각각 $n$번째 feature와 weight이고,
$Z \left( \mathbf{e} \right)$는 normalization constant입니다.
Weight는 [BLEU](https://en.wikipedia.org/wiki/BLEU) score를 maximize하도록 optimize 됩니다.

이 논문에서는 SMT 중에서도 phrase-based 방법을 사용하는데,
translation model $p \left( \mathbf{e} \; | \; \mathbf{f} \right)$을 
source와 target sentence에서 matching하는 phrase들의 확률로 분해해서 구합니다.
즉, 완전히 새로운 번역기 시스템을 만드는 것이 아니라
구현의 편의를 위해
기존 SMT 시스템의 phrase pair table에 점수를 매기는 부분에만 
RNN Encoder-Decoder를 적용했습니다.

## Experiments ##

실험에는 WMT'14 workshop의 English/French translation task를 사용했습니다.
이 task는 영어를 프랑스어로 번역하는 작업인데, 
이때 영어 phrase를 프랑스어 phrase로 번역하는 쌍의 확률을 학습하도록 모델을 training하고,
이 모델을 baseline 시스템에 적용해 phrase pair table에 점수를 매기도록 했습니다.

Baseline의 phrase 기반 SMT 시스템으로는
대표적인 free software SMT 엔진인 [Moses](http://www.statmt.org/moses/)를 
default setting 상태로 사용했습니다.

기타 실험에서 사용한 parameter들과 training 조건은 아래와 같습니다.

- 1000 hidden units (GRU)
- SGD with Adadelta
- 64 phrase pairs used per each update
- most frequent 15,000 words (both English and French)

또한, 실험에는
target language model을 습득하는 neural network인
Schwenk의 [CSLM](http://www-lium.univ-lemans.fr/~schwenk/papers/Schwenk.mtlm.acl2006.pdf)를
추가로 적용해 사용하기도 했습니다.

실험에서 사용한 4가지 조합은 다음과 같습니다.
- **Baseline**: Moses 기본 세팅
- **Baseline + RNN**: RNN Encoder-Decoder 적용
- **Baseline + CSLM + RNN**: CSLM 추가 적용
- **Baseline + CSLM + RNN + Word penalty**: 모르는 단어에 penalty 추가 적용

## Results ##

아래 그림은 실험 결과를 요약한 테이블입니다.

![Table 1]({{ site.baseurl }}/media/2017-04-23-learning-phrase-representations-using-rnn-encoder-decoder-table1.png)

간단히 정리하면 다음과 같습니다.

- RNN Encoder-Decoder를 적용해 baseline 대비 성능이 개선되었다.
- CSLM과 함께 적용했을 때 가장 성능이 좋았다. 즉, 두 방법이 성능 향상에 독립적으로 기여한다.
- Word penalty까지 적용한 경우는 test set에서 성능이 약간 떨어졌다.

또한, 정성적인 실험 결과에서는
제안하는 모델이 
phrase들을
의미적으로(semantically) 그리고 문법적으로(syntactically)
잘 표현하는 것을 볼 수 있었습니다.
아래 그림은 학습한 phrase representation의 2D embedding 일부를 확대해 보인 것입니다.

![Figure 7]({{ site.baseurl }}/media/2017-04-23-learning-phrase-representations-using-rnn-encoder-decoder-figure7.jpg)

이 논문에서는
임의의 길이를 가지는 sequence를 다른 (역시 임의의 길이를 가지는) sequence로 mapping하는
학습을 할 수 있는 새로운 neural network인 RNN Encoder-Decoder (seq2seq)를 제안했습니다.
이 모델은 sequence pair들에게 조건부 확률에 기반해 점수를 매기는 용도로 쓸 수도 있고,
source sequence 입력에 대한 target sequence 생성에도 쓸 수 있습니다.

또한 이 논문에서는 
reset gate와 update gate로 구성되어
개별 hidden unit들이 기억하거나 잊는 정도를 adaptive하게 제어할 수 있는
새로운 hidden unit (GRU)를 제안했습니다.
GRU는 현재까지 RNN 구현에 LSTM과 함께 필수적인 구성요소로 자리매김하고 있습니다.


-- *[Jamie](http://twitter.com/JiyangKang);*
<br>
<iframe width="560" height="315" src="https://www.youtube.com/embed/_Dp8u97_rQ0?list=PLlMkM4tgfjnJhhd4wn5aj8fVTYJwIpWkS" frameborder="0" allowfullscreen></iframe>
<br>

**References**

- K. Cho의 논문 ["Learning Phrase Representations Using RNN Encoder-Decoder for Statistical Machine Translation"](https://arxiv.org/abs/1406.1078)
- J. Chung의 논문 ["Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling"](https://arxiv.org/abs/1412.3555)
- 곽근봉 님의 슬라이드 ["Learning Phrase Representations Using RNN Encoder-Decoder for Statistical Machine Translation"](https://www.slideshare.net/keunbongkwak/learning-phrase-representations-using-rnn-encoder-decoder-for-statistical-machine-translation)
- Wikipedia의 [Long short-term memory](https://en.wikipedia.org/wiki/Long_short-term_memory)
- S. Hochreiter의 논문 ["Long Short-term Memory"](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory)
- Wikipedia의 [Gated recurrent unit](https://en.wikipedia.org/wiki/Gated_recurrent_unit)
- Wikipedia의 [Neural machine translation](https://en.wikipedia.org/wiki/Neural_machine_translation)
- TensorFlow 공식 사이트의 [Sequence-to-Sequence Models](https://www.tensorflow.org/tutorials/seq2seq) 
- C. Olah의 블로그 post ["Understanding LSTM Networks"](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- WildML의 블로그 post ["Recurrent Neural Networks Tutorial, Part 4 - Implementing a GRU/LSTM RNN with Python and Theano"](http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/)
- Wikipedia의 [Statistical machine translation](https://en.wikipedia.org/wiki/Statistical_machine_translation)
- Wikipedia의 [BLEU](https://en.wikipedia.org/wiki/BLEU)
- Schwenk의 논문 ["Continuous Space Language Models for Statistical Machine Translation"](http://www-lium.univ-lemans.fr/~schwenk/papers/Schwenk.mtlm.acl2006.pdf)