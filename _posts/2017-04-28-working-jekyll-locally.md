---
layout: post
title: 로컬에서 Jekyll 블로그 작업하기
use_math: true
date: 2017-04-28 11:23:41
#categories: 
#- Keep Learning
tags: [jekyll, github]
#tags: [Jekyll, GitHub]
---

[Jekyll](https://jekyllrb.com/) 블로그에 올릴 포스트를 GitHub repository로 push하지 않고, 로컬에서 preview하며 작업할 수 있습니다. Jekyll 공식 페이지의 [Basic Usage](https://jekyllrb.com/docs/usage/)에 있는 내용 중에서 가장 필요한 부분만 여기서 정리해 보겠습니다.

먼저 `jekyll serve` 명령은 개발을 위한 웹 서버를 로컬에서 실행합니다. 기본으로 auto-regeneration 기능이 활성화 되어 있어, `jekyll serve` 실행 중에 data 파일(예를 들면 `.md` 또는 `.html`)에 변동이 있으면 실시간으로 `_site` 디렉토리의 파일들을 다시 생성합니다. 그 결과 [http://localhost:4000/](http://localhost:4000/)로 접속한 웹 브라우저에서 방금 수정한 내용을 바로 확인할 수 있습니다.

```bash
$ jekyll serve
# => A development server will run at http://localhost:4000/
# Auto-regeneration: enabled. Use `--no-watch` to disable.
```

한 가지 알아둘 점은, `_config.yml` 파일은 수정해도 바로 반영되지 않습니다. 이 파일은 `jekyll serve`를 `Ctrl-C`를 입력해 종료하고 다시 실행해야 반영됩니다.

`jekyll serve` 명령은 실행하면 `_site` 디렉토리의 파일들을 전부 다시 생성합니다. 블로그의 규모가 작을 때는 문제가 되지 않지만, 커졌을 때는 생성에 걸리는 시간이 부담스러울 수 있습니다. 그럴 때는 `--skip-initial-build`라는 option을 사용하면 생성 과정을 생략하고 바로 웹 서버가 동작합니다.

```bash
$ jekyll serve --skip-initial-build
# => A development server will run at http://localhost:4000/
# Skips the initial site build which occurs before the server is started.
```

이와 반대로, 로컬 웹 서버 기능이 필요없고 단순히 `_site` 디렉토리의 파일을 다시 생성하고 싶다면 `jekyll build` 명령을 실행하면 됩니다.

```bash
$ jekyll build
# => The current folder will be generated into ./_site
```

-- *[Jamie](http://twitter.com/JiyangKang);*

## References

- Jekyll 공식 페이지의 [Basic Usage](https://jekyllrb.com/docs/usage/)
