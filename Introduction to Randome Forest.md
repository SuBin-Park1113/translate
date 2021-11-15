# 랜덤포레스트
***
## 불도저를 위한 지침서
``` 
%load_ext autoreload<br>
%autoreload 2<br>
%matplotlib inline<br>
from fastai.imports import *<br>
from fastai.structured import *<br>
from pandas_summary import DataFrameSummary<br>
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier<br>
from IPython.display import display<br>
from sklearn import metrics<br>
```
<br>
데이터 과학은 소프트웨어 공학이 아닙니다 [[08:43]](https://www.youtube.com/watch?v=CzdWqFTmn0Y&t=523s). 여러분은 `PEP 8`을 따르지 않는 코드 및 `import *` 와 같은 것들 것 보게 될 것입니다. 하지만 그것은 잠깐일 뿐입니다. 우리가 지금 하고 있는 것은 모델을 프로토타이핑 하는 것이며 프로토타이핑 모델에는 어디에서도 본 적없는 남다른 모범 사례들이 있습니다. 핵심은 매우 상호적이고 반복적인 무언가를 할 수 있다는 것입니다. 주피터 노트북은 이를 쉽게 만들어줍니다. 만약 `display`가 무엇인지 궁금하다면 다음 세가지 중 하나를 하십시오.

1. 셀에 `display`를 입력하고 shift+enter 키를 누르십시오. – `<function IPython.core.display.display>`의 출처를 알 수 있습니다. <br>
2. 셀에 `?display`를 입력하고 shift+enter 키를 누르십시오. – 문서가 표시됩니다. <br>
3. 셀에 `??display`를 입력하고 shift+enter 키를 누르십시오. – 소스코드가 표시됩니다. 특히 대부분의 함수가 읽기 쉽고 5줄을 넘지 않기 때문에 fastai 라이브러리에 유용합니다.<br>
***
## 데이터 다운로드 [[12:05]](https://www.youtube.com/watch?t=12m5s&v=CzdWqFTmn0Y&feature=youtu.be)

Kaggle 대회에 참가하면 어떤 종류의 모델, 어떤 종류의 데이터에 능숙한지 알 수 있습니다. 데이터에 노이즈가 심해서 정확도가 나쁩니까? 아니면 데이터셋은 간단하지만 실수하셨습니까? 당신의 프로젝트를 진행할 때, 당신은 이런 피드백은 얻을 수 없습니다. – 우리는 그저 기준모델을 안정적으로 개발할 수 있는 효과적인 기술이 있다는 것을 알면 됩니다.<br><br>

머신러닝은 데이터셋을 예측할 뿐 아니라 이해할 수 있게 해주어야 합니다[[15:36]](https://www.youtube.com/watch?v=CzdWqFTmn0Y&t=936s). 그래서 익숙하지 않은 분야를 선택하면 이해할 수 있는지를 시험할 수 있습니다. 그렇지 않으면 데이터에 대한 직감으로 인해 데이터의 실제 의미를 파악하기 어려워집니다. <br><br>

여기 데이터를 다운로드하는 몇가지 방법이 있습니다. <br>
1. 컴퓨터에 다운로드하고 AWS로 `scp`하세요.
2. [[17:32]](https://www.youtube.com/watch?t=17m32s&v=CzdWqFTmn0Y&feature=youtu.be) Firefox에서 `ctrl + shift + I` 키를 눌러 웹 개발자 도구를 실행시킵니다. `Network` 탭으로 이동하여 `Download` 버튼을 클릭하고 대화상자를 빠져나갑니다. 그러면 네트워크 연결이 표시됩니다. 그 후 마우스 오른쪽을 클릭하고 `Copy as cURL`을 선택합니다. 명령을 붙여넣고 끝에 `-o bulldozer.zip`을 추가합니다.<br><br>

#### 데이터 알아보기

* **구조적 데이터(Structured data):** 식별자, 날짜, 크기 등 다양한 유형의 항목을 나타내는 컬럼입니다.
* **비구조적 데이터(Unstructured data):** `pandas`는 `pd'로 import된 구조적 데이터로 작업할 때 가장 중요한 라이브러리입니다.

``` 
df_raw = pd.read_csv(f'{PATH}Train.csv', low_memory=False, 
                     parse_dates=["saledate"])
```
* `parse_dates`는 해당 날짜를 포함하는 컬럼의 목록을 뜻합니다.
* `low_memory=False`는 타입을 결정하기 위해 파일을 더 읽어오도록 합니다.

``` 
def display_all(df):
    with pd.option_context("display.max_rows", 1000): 
        with pd.option_context("display.max_columns", 1000): 
            display(df)
display_all(df_raw.tail().transpose())
```
