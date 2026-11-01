# GGUF 웹 실행기

브라우저 안에서만 GGUF 모델을 돌리는 정적 웹 페이지입니다. 서버나 백엔드가 없고, 모델 파일은 사용자가 그때그때
고릅니다 (저장소나 Pages에 올리지 않음). OpenAI 형식 도구 호출(tool calling)도 됩니다.

- 런타임: [`@wllama/wllama`](https://github.com/ngxson/wllama) **3.8.1** (llama.cpp WASM). CDN(jsdelivr)에서 불러옵니다.
- 멀티스레드: GitHub Pages는 응답 헤더를 바꿀 수 없어서 `coi-serviceworker.js`(MIT, v0.1.7)가 COOP/COEP를 붙입니다.
  처음 접속하면 서비스 워커를 등록하고 **한 번 새로고침**됩니다.
- WebGPU: wllama 3.1부터 자동으로 씁니다. "WebGPU 쓰기"를 끄면 `n_gpu_layers: 0`(CPU만)으로 불러옵니다.

## 파일

| 파일 | 내용 |
| --- | --- |
| `index.html` | 화면·스타일·스크립트 전부 (빌드 도구 없음) |
| `coi-serviceworker.js` | 멀티스레드용 서비스 워커. `<head>`에서 가장 먼저 불러옴 |
| `.nojekyll` | Pages가 Jekyll로 바꾸지 않게 |

## 실행 (내 PC에서)

```bash
cd webrunner
python -m http.server 8000
```

브라우저(Chrome 권장)에서 `http://localhost:8000`을 엽니다.

1. 첫 접속 때 한 번 새로고침됩니다.
2. "모델 파일"에서 `.gguf`를 고릅니다. 여러 조각으로 나뉜 모델은 조각을 모두 함께 고르면 이름순으로 불러옵니다.
3. 상태 줄에 "준비됨"이 뜨고 아래에 스레드 수와 WebGPU 여부가 나오면 말을 입력합니다.
   - `crossOriginIsolated=true`이고 스레드가 2개 이상이면 멀티스레드로 도는 것입니다.

`file://`로 바로 열면 서비스 워커가 동작하지 않습니다. 꼭 위처럼 http로 엽니다.

## 화면

- **도구**
  - 기본 도구: `calculator`, `get_time`, `pibo_action`. 도구 결과를 넣어 다시 묻고, 최대 5번 반복합니다.
  - 더메이커 한국어·영어: 더메이커 모델이 학습한 시스템 글과 도구(`light`, `look`, `move`)를 그대로 씁니다.
    더메이커는 도구를 부를 때 앞에 짧은 말("빨간색으로 바꿀게요!")을 하고, 결과를 다시 받지 않게 배웠습니다.
    그래서 이 묶음은 기기 앱처럼 도구를 한 번 실행하고 끝냅니다.
  - 도구 없음
- **대화 기록**: 나 / 모델 / 도구 호출 / 도구 결과를 구분해 보여 줍니다. "새 대화"로 기억을 비웁니다.
- **실행·중지**: 생성은 흘려 보여 주고(스트리밍), 중지 버튼은 진행 중인 생성을 끊습니다.
- **생각 끄기**: Qwen3.5 원본처럼 생각을 먼저 하는 모델에 `chat_template_kwargs: {enable_thinking: false}`를 보냅니다.
  더메이커는 GGUF 템플릿이 늘 생각 없이 대답하므로 상관없습니다.
- 모델을 다시 고르면 앞의 모델을 `exit()`로 내리고 새로 만듭니다. 불러오기·생성 오류는 상태 줄과 대화 기록, 콘솔에 나옵니다.

## 도구 더하기

`index.html`의 `<script id="tools">` 한 곳만 고칩니다.

1. `TOOLSETS`에서 쓸 묶음(예: `basic`)의 `tools` 배열에 OpenAI 형식 정의를 넣습니다.
   ```js
   {type: "function", function: {name: "get_weather", description: "날씨를 알려 줍니다.",
     parameters: {type: "object", properties: {city: {type: "string"}}, required: ["city"]}}}
   ```
2. 같은 묶음의 `run`에 같은 이름의 함수를 넣습니다. 인자 객체를 받아 결과 객체(또는 Promise)를 돌려줍니다.
   ```js
   get_weather: ({city}) => ({city, condition: "맑음"}),
   ```
3. 목록에 없는 도구나 enum에 없는 값은 실행하지 않고 `{error: …}`를 결과로 돌려줍니다 (`executeTool`).

파이보에 실제로 동작을 보내려면 `PIBO_ACTION_ENDPOINT`에 파이보 HTTP API 주소를 넣습니다. `null`이면 흉내만 냅니다.
브라우저가 https(Pages) 페이지에서 http 주소로 보내는 것은 막을 수 있습니다 (mixed content, 확인 필요).

## GitHub Pages 배포

이 폴더만 배포하도록 워크플로(`.github/workflows/webrunner-pages.yml`)를 두었습니다. 저장소의 다른 파일과 문서는
올라가지 않습니다.

1. 저장소 **Settings → Pages → Build and deployment → Source**를 **GitHub Actions**로 바꿉니다.
2. `main` 브랜치에 `webrunner/`가 바뀌어 올라가면 자동으로 배포됩니다. Actions 탭의 `webrunner-pages`에서
   "Run workflow"로 손으로 돌릴 수도 있습니다.
3. 주소는 `https://<사용자 또는 조직>.github.io/<저장소>/`입니다 (워크플로 결과에 나옴).

- 폴더 경로를 바꾸려면 워크플로의 `paths`와 `upload-pages-artifact`의 `path` 두 곳을 고칩니다.
- `github-pages` 환경은 보통 기본 브랜치(`main`)에서만 배포를 허용합니다. 다른 브랜치에서 돌리면 거절될 수 있습니다.
- 워크플로는 `webrunner/` 안에 `.gguf`가 있으면 배포를 멈춥니다. 저장소 `.gitignore`에도 `*.gguf`가 있습니다.

## 하지 않는 것

- 모델 파일을 저장소나 Pages에 올리지 않습니다.
- 외부 API(OpenAI 등)를 부르지 않습니다. 대화는 브라우저 안에서만 돌고 저장하지 않습니다.
- 프레임워크나 번들러를 쓰지 않습니다.
