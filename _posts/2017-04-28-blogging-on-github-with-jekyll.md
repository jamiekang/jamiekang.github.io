---
layout: post
title: Jekyll로 GitHub에 blog 만들기
use_math: true
date: 2017-04-28 03:22:31
#categories: 
#- Keep Learning
tags: [jekyll, github]
#tags: [Jekyll, GitHub]
---

GitHub에 블로그를 만들려면 알아야 할 것이 많습니다. 티스토리나 WordPress를 쓰던 것과는 사뭇 다른 방식이라서 처음에 고생하게 되는데, 이 블로그를 꾸미며 배운 것들을 간단히 정리해 보겠습니다.

GitHub의 블로그는 [GitHub Pages](https://pages.github.com/)가 제공하는 기능으로, GitHub repository에서 직접 static한 웹 페이지를 호스팅합니다. 이를 위해서는 [Jekyll](https://jekyllrb.com/)이라는 정적 사이트 생성기(static site generator)를 설치해야 합니다. Jekyll 자체가 ruby 언어의 gem 형태로 번들 되기 때문에, 터미널에서 설정하고 설치하는 작업이 약간 필요합니다. 

## Jekyll Now

먼저 알아야 할 것은, GitHub pages에서 Jekyll을 사용하는 것이 일반적이지만 Jekyll의 모든 기능이 GitHub pages에서 허용되는 것은 아니라는 점입니다. 예를 들어, [Jekyll의 plugin 기능은 GitHub에서 동작하지 않습니다](http://charliepark.org/jekyll-with-plugins/). 그래서 Jekyll 자체의 문서만 보면서 작업을 하다 보면 GitHub에서 여러가지 문제가 생길 수 있습니다. 또한 ruby의 설치 등 잡다한 과정이 다소 귀찮을 수도 있습니다.

검색해 보면 Jekyll로 GitHub에 블로그를 더 쉽게 시작하도록 도와주는 프로젝트들이 있습니다. 그 중에서 검색 결과 최상단에 나오는 것이 [Jekyll Now](https://github.com/barryclark/jekyll-now)이며, 저도 사용했습니다. 역시 또 알아야 할 것은, Jekyll Now 또한 Jekyll로 GitHub에 블로그를 설치하는 여러 방법 중의 하나일 뿐이며 제약점도 많이 가지고 있다는 점입니다. 아래에서 더 설명하겠습니다.

## Theme

Jekyll의 웹페이지에는 마치 많은 Theme이 있는 것처럼 보입니다. 하지만 Jekyll Now를 사용할 경우, Jekyll Now에서 [검증된 별도의 Theme](https://github.com/barryclark/jekyll-now#other-forkable-themes)을 사용하는 것이 훨씬 쉽습니다. Jekyll Theme을 적용하는 [GitHub Pages의 공식 문서](https://help.github.com/articles/creating-a-github-pages-site-with-the-jekyll-theme-chooser/)의 설명은 Jekyll Now에서는 동작하지 않습니다. 섣불리 시도했다가 ruby의 알 수 없는 에러 메시지들을 만나는 것은 물론, 파일도 날아가는 경우가 있습니다. Jekyll Now의 새로운 Theme을 적용하는 것은 처음 GitHub에서 fork하는 것부터 시도하는 것이 가장 쉬운 방법인 것 같습니다.

## Front Matter

[Front matter](https://jekyllrb.com/docs/frontmatter/)는 각 페이지에서 Jekyll에 특별한 속성 정보(메타 데이터)를 전달하는 방법입니다. Front matter는 각 페이지(html 또는 markdown)의 시작 부분에 아래처럼 `---`로 구분된 블록 형태로 되어 있습니다. 

```html
---
layout: post
title: Blogging Like a Hacker
---
<!--
your markdown or html starts here
-->
```

Front matter는 다소 생소한 `yaml` 문법으로 되어 있지만 사용하기 쉽습니다. 위의 예에서 보인 `layout`, `title` 외에도, `date`, `permalink`, `published`, `categories`, `tags` 등을 설정해 줄 수 있습니다.

## MathJax

Jekyll에서 수학식을 표시하려면 [MathJax](https://github.com/mathjax/MathJax)를 사용합니다. 

MathJax를 지원하는 것은 Ben Lansdell의 [MathJax, Jekyll and github pages](http://benlansdell.github.io/computing/mathjax/)를 참고해 쉽게 설정할 수 있습니다.

먼저, `_config.yml`을 수정하고 `_includes` 디렉토리에 `mathjax_support.html`라는 파일을 추가합니다. 그 다음으로 `mathjax_support.html` 파일을 읽기 위해 `_layouts/default.html` (Jekyll Now의 기본 theme) 또는 `head.html`(Lanyon theme)의 `<head></head>` 사이에 아래의 코드를 추가하면 설정이 완료됩니다.

{% raw %}
```html
{% if page.use_math %}
  {% include mathjax_support.html %}
{% endif %}
```
{% endraw %}

이 때 주의할 것은, 수학식이 들어가는 페이지(예를 들면 새 post)의 `front matter` 부분에서 `use_math : true`라고 선언을 해줘야 한다는 점입니다. 방금 위에서 보인 코드라면 아래처럼 수정해주면 됩니다.

```html
---
layout: post
title: Blogging Like a Hacker
use_math: true
---
<!--
your markdown or html starts here
-->
```

## Google Analytics

티스토리, 네이버 블로그와 달리 Jekyll은 자체적으로 방문자 통계를 관리하지 않습니다. 오늘 내 블로그에 몇 명이 방문했고, 어떤 검색으로 찾아 왔는지와 같은 기본 정보를 알기 위해서는 [Google Analytics](https://analytics.google.com/) 서비스를 사용합니다. 

저는 loustler 님의 [Jekyll을 이용한 Github pages 만들기[심화/Google Analytics 적용]](http://loustler.io/2016/09/26/github_pages_blog_google_analytics/)이라는 포스트 내용을 참고해서 적용했습니다. Google Analytics에 가입할 때 빈 칸 몇 개를 채우는 것이 가장 어려운 일이었을 정도로 설정이 쉽습니다.

![Dashboard]({{ site.baseurl }}/media/2017-04-28-blogging-on-github-with-jekyll-google-analytics2.jpg)

Google Analytics의 분석 결과는 위의 그림과 같이 확인할 수 있다고 합니다. 아직 이 블로그는 보여드릴 만한 통계가 없어서, 위의 이미지는 [flickr](https://www.flickr.com)에 공개된 것을 사용했습니다. 나중에 실제 dashboard 이미지로 바꿔 보여드리겠습니다.

-- *[Jamie](http://twitter.com/JiyangKang);*

## References

**Jekyll Blog 일반**
- Jekyll Blog의 [local copy @127.0.0.1:4000](http://127.0.0.1:4000/)
- Jekyll Bootstrap의 [Jekyll QuickStart](http://jekyllbootstrap.com/usage/jekyll-quick-start.html)
- Jekyll의 [공식 Docs](https://jekyllrb.com/docs/home/)
- Stack Overflow의 [Jekyll tagged questions](https://stackoverflow.com/questions/tagged/jekyll)
- GitHub의 [Pages 공식 페이지](https://pages.github.com/)
- GitHub Help의 [Adding a Jekyll theme to your GitHub Pages site](https://help.github.com/articles/adding-a-jekyll-theme-to-your-github-pages-site/)
- 한량넷의 [초보자를 위한 Jekyll Blog 시작하기](http://www.halryang.net/Jekyll-Blogging-For-Beginners/)
- 놀부 님의 [지킬로 깃허브에 무료 블로그 만들기](https://nolboo.kim/blog/2013/10/15/free-blog-with-github-jekyll/)
- KaKao Tech Blog의 [kakao 기술 블로그가 GitHub Pages로 간 까닭은](http://tech.kakao.com/2016/07/07/tech-blog-story/)
- Anatol Broder의 [Compress HTML in Jekyll](https://github.com/penibelst/jekyll-compress-html)
- Will Koehler의 [Save 50 Hours Setting up Your Jekyll Blog](http://willkoehler.net/2014/08/26/save-50-hours-setting-up-your-jekyll-blog.html)

**Jekyll Now**
- Barry clark의 [Jekyll Now](https://github.com/barryclark/jekyll-now)
- Smashing Magazine의 [Build A Blog With Jekyll And GitHub Pages](https://www.smashingmagazine.com/2014/08/build-blog-jekyll-github-pages/)
- taehwan 님의 [GitHub 블로그 빠르게 시작하기!](http://thdev.net/653)

**Jekyll Now Theme**
- 한량넷의 [Easyjekyll](https://github.com/easyjekyll/easyjekyll.github.io/)
- Mark Otto의 [Hyde](https://github.com/poole/hyde)
- Mark Otto의 [Lanyon](https://github.com/poole/lanyon)
- Michael Rose의 [Minimal Mistakes](https://github.com/mmistakes/minimal-mistakes)
- Barry clark의 [Jekyll Now - Other forkable themes](https://github.com/barryclark/jekyll-now#other-forkable-themes)

**Front Matter**
- Jekyll의 [Front Matter](https://jekyllrb.com/docs/frontmatter/)
- Code Beauty의 [YAML Validator](http://codebeautify.org/yaml-validator)

**MathJax**
- GitHub의 [공식 MathJax repository](https://github.com/mathjax/MathJax)
- Ben Lansdell의 [MathJax, Jekyll and github pages](http://benlansdell.github.io/computing/mathjax/)

**Google Analytics**
- 티스토리 블로그 채널의 [구글 애널리틱스 설치 사용법 및 팁 노하우](http://blogchannel.tistory.com/149)
- 에르차마토리 웹로그의 [구글 아널리틱스를 이용하여 블로그를 완벽하게 분석하는 방법](http://www.erzsamatory.net/42)
- loustler 님의 [Jekyll을 이용한 Github pages 만들기[심화/Google Analytics 적용]](http://loustler.io/2016/09/26/github_pages_blog_google_analytics/)

**Markdown**
- Adam Pritchard의 [Markdown Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)

**GitHub**
- 놀부 님의 [완전 초보를 위한 깃허브](https://nolboo.kim/blog/2013/10/06/github-for-beginner/)
