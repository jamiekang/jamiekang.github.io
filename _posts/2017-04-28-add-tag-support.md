---
layout: post
title: Jekyll 블로그에 tag 지원하기
use_math: true
date: 2017-04-28 07:22:31
#categories: 
#- Keep Learning
tags: [jekyll, github]
#tags: [Jekyll, GitHub]
---

[Jekyll](https://jekyllrb.com/) 블로그는 category와 tag를 기본으로 지원하지 않습니다. 
일부 Jekyll Theme에서 지원하는 경우도 있지만 선택의 폭이 크지 않습니다. 
이미 사용하는 Theme이 있다면 그 위에 기능을 추가하는 방법도 있습니다.

Jekyll Now에 적용 가능한 tag 구현 방법을 찾아보니 크게 3가지가 있었습니다. 물론 더 있을 수 있습니다.

## Lanyon theme에 추가하는 방법

첫 번째는 GitHub [Lanyon 프로젝트](https://github.com/poole/lanyon)에 [등록된 issue](https://github.com/poole/lanyon/issues/83)에 대한 해결책이 [pull request](https://github.com/poole/lanyon/pull/85)를 거쳐 [별도의 프로젝트](https://github.com/wireddown/wireddown.github.io/tree/feature_tags)가 된 방법입니다.

적용 방법은 위의 [pull request](https://github.com/poole/lanyon/pull/85)에 잘 나와 있습니다. 간단히 정리하면 다음과 같습니다.

```html
1. _data/tags.yml 추가 -- tag 저장소
2. tags.md 추가 -- tag 전체의 index page
3. _includes/tag_collector.html 추가 -- tag lists 생성
4. _includes/tag_link_formatter.html 추가 -- tag links 생성
5. _layouts/posts_by_tag.html 추가 -- 특정 tag가 있는 모든 포스트에 대한 page layout
6. _tools/createTag 추가 -- 새 tag 추가 명령어
7. index.html 및 _layouts/post.html 업데이트 -- 현재 post에 tag 표시
8. README.md 및 Introducing Lanyon 업데이트 (option)
```

이렇게 적용한 후에, 새로운 tag(예를 들어 GitHub Pages)를 등록하려면 command line에서 아래의 명령어를 입력합니다. 

```bash
$ ./_tools/createTag "GitHub Pages"
```

그러면 아래와 같은 결과가 출력됩니다. 참고로 return되는 `github-pages`라는 string은 내가 등록한 "GitHub Pages"라는 tag의 slug이라고 부릅니다.

```
Begin using this tag by adding this line to your post's Front Matter:
  ---
  tags: [github-pages]
  ---
```

새로운 tag를 등록하는 `_tools/createTag` 명령어를 사용하려면 먼저 시스템에 `gawk` (GNU awk)이 설치되어 있어야 합니다. Mac OSX 환경이라면 terminal에서 아래와 같이 2개의 명령어를 실행하면 `gawk`이 설치됩니다.

```bash
$ ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)" < /dev/null 2> /dev/null
$ brew install gawk
```

이후에 작성한 post에서 tag를 사용하려면 front matter 부분에 아래와 같은 형식으로 적으면 됩니다.

```
---
layout: post
title: Blogging Like a Hacker
tags: [github-pages]
---
<!--
your markdown or html starts here
-->
```

여러 개의 tag를 적용하려면 쉼표로 구분해서 적어줍니다.

```
---
layout: post
title: Blogging Like a Hacker
tags: [github-pages, jekyll]
---
<!--
your markdown or html starts here
-->
```

여기서 중요한 것은, **front matter에 내가 쓰고 싶은 tag의 이름("GitHub Pages")을 적는 것이 아니라 그 tag의 slug (`github-pages`)을 적는다는 것입니다.** 등록한 tag들의 이름과 slug 정보는 `_data/tags.yml` 라는 파일에 아래와 같이 간단한 형식으로 기록되어 있습니다.

```yml
- slug: jekyll
  name: Jekyll

- slug: github-pages
  name: GitHub Pages

- slug: machine-learning
  name: Machine Learning
```

이 방법이 적용된 [Down to the Wire](http://downtothewire.io/)라는 demo 사이트를 보면, 첫 화면에 아래 스크린샷처럼 tag들이 나열된 것을 볼 수 있습니다. (첫 화면의 layout은 `index.html`을 수정해서 바꿀 수 있습니다.)

![downtothewire]({{ site.baseurl }}/media/2017-04-28-add-tag-support-tags.jpg)

## Jekyll의 Collections 기능을 사용하는 방법

이 방법은 [kakao 기술 블로그](http://tech.kakao.com/)가 GitHub pages와 Jekyll 기반으로 옮기면서 공개한 [포스트](http://tech.kakao.com/2016/07/07/tech-blog-story/)를 참고하는 것입니다. Jekyll의 [Collections](https://jekyllrb.com/docs/collections/)라는 기능을 활용했고 비교적 따라하기 쉽습니다. 

실제 코드와 함께 상세한 설명은 kakao 기술 블로그의 ["kakao 기술 블로그가 GitHub Pages로 간 까닭은"](http://tech.kakao.com/2016/07/07/tech-blog-story/)이라는 포스트를 보시길 권합니다. 아래에 간단히 핵심만 정리했습니다.

```html
1. _config.yml 파일에 collections와 defaults 설정을 추가
2. _layout 디렉토리에 tag.html 파일을 작성
3. _tags 디렉토리를 만들고 각 tag마다 하나의 md 파일(예: opensource.md)을 추가
```

검색하면 이 방법으로 tag를 구현한 블로그들을 발견할 수 있습니다. 저는 loustler 님의 블로그에서 ["Jekyll을 이용한 Github pages 만들기[심화/태그]"](http://loustler.io/2016/09/25/create_github_page_use_jekyll_2/)라는 포스트를 참고했습니다.

이 방법 또한, 새로운 tag를 추가할 때마다 수동으로 새로운 md 파일을 만드는 번거로움(위의 3번 과정)이 있습니다. GitHub pages에서 Jekyll에 plugin을 허용하지 않기 때문에 현재까지는 해결 방법이 없는 것 같습니다.

## 다른 블로그의 소스를 활용하는 방법

GitHub pages로 구현한 블로그는 repository에 모든 소스가 들어 있기 때문에 (공개된 repository라면) 소스를 찾아볼 수 있습니다. 

Lanyon theme 기반으로 가장 완성도 높은 블로그 중의 하나는 [Michael Lanyon의 블로그](https://blog.lanyonm.org/)입니다. 감춰진 왼편 sidebar에서 [TAGS](https://blog.lanyonm.org/tags.html)를 선택하면 아래 스크린샷처럼 간결하게 정리된 Tags 페이지가 나옵니다.

![lanyonm]({{ site.baseurl }}/media/2017-04-28-add-tag-support-lanyonm.jpg)

이 블로그의 Jekyll 소스는 [lanyonm/lanyonm.github.io](https://github.com/lanyonm/lanyonm.github.io) 라는 repository에 공개되어 있습니다.

한 가지 주의할 것은 해당 사이트의 라이센스 정책입니다. Jekyll 기반의 블로그는 대개 [MIT 라이센스](https://ko.wikipedia.org/wiki/MIT_%ED%97%88%EA%B0%80%EC%84%9C)를 따르지만, 작업 전에 반드시 개별 정책을 확인할 필요가 있습니다.

## References

- Mark Otto의 [Lanyon](https://github.com/poole/lanyon)
- wireddown의 [feature_tags](https://github.com/wireddown/wireddown.github.io/tree/feature_tags)
- wireddown의 [Down to the Wire](http://downtothewire.io/)
- KaKao Tech Blog의 [kakao 기술 블로그가 GitHub Pages로 간 까닭은](http://tech.kakao.com/2016/07/07/tech-blog-story/)
- loustler 님의 [Jekyll을 이용한 Github pages 만들기[심화/태그]](http://loustler.io/2016/09/25/create_github_page_use_jekyll_2/)
- Michael Lanyon의 [블로그](https://blog.lanyonm.org/)
