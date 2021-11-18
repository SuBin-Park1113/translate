# 랜덤포레스트
***
## 랜덤포레스트 : 중고 불도저 가격표
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
데이터 과학은 소프트웨어 공학이 아닙니다 [[08:43]](https://www.youtube.com/watch?v=CzdWqFTmn0Y&t=523s). 여러분은 `PEP 8`을 따르지 않는 코드 및 `import *` 와 같은 것들 것 보게 될 것입니다. 하지만 그것은 잠깐일 뿐입니다. 우리가 지금 하고 있는 것은 모델을 프로토타이핑 하는 것이며 프로토타이핑 모델에는 어디에서도 본 적없는 남다른 모범 사례들이 있습니다. 핵심은 매우 상호적이고 반복적인 무언가를 할 수 있다는 것입니다. 주피터 노트북은 이를 쉽게 만들어줍니다. 만약 `display`가 무엇인지 궁금하다면 다음 세가지 중 하나를 하십시오.<br><br>

1. 셀에 `display`를 입력하고 shift+enter 키를 누르십시오. – `<function IPython.core.display.display>`의 출처를 알 수 있습니다. <br>
2. 셀에 `?display`를 입력하고 shift+enter 키를 누르십시오. – 문서가 표시됩니다. <br>
3. 셀에 `??display`를 입력하고 shift+enter 키를 누르십시오. – 소스코드가 표시됩니다. 특히 대부분의 함수가 읽기 쉽고 5줄을 넘지 않기 때문에 fastai 라이브러리에 유용합니다.<br>
***
## 데이터 다운로드 [[12:05]](https://www.youtube.com/watch?t=12m5s&v=CzdWqFTmn0Y&feature=youtu.be)

Kaggle 대회에 참가하면 어떤 종류의 모델, 어떤 종류의 데이터에 능숙한지 알 수 있습니다. 데이터에 노이즈가 심해서 정확도가 나쁩니까? 아니면 데이터셋은 간단하지만 실수하셨습니까? 당신의 프로젝트를 진행할 때, 당신은 이런 피드백은 얻을 수 없습니다. – 우리는 그저 기준모델을 안정적으로 개발할 수 있는 효과적인 기술이 있다는 것을 알면 됩니다.<br>

머신러닝은 데이터셋을 예측할 뿐 아니라 이해할 수 있게 해주어야 합니다[[15:36]](https://www.youtube.com/watch?v=CzdWqFTmn0Y&t=936s). 그래서 익숙하지 않은 분야를 선택하면 이해할 수 있는지를 시험할 수 있습니다. 그렇지 않으면 데이터에 대한 직감으로 인해 데이터의 실제 의미를 파악하기 어려워집니다. <br><br>

여기 데이터를 다운로드하는 몇가지 방법이 있습니다.<br>
1. 컴퓨터에 다운로드하고 AWS로 `scp`하세요.
2. [[17:32]](https://www.youtube.com/watch?t=17m32s&v=CzdWqFTmn0Y&feature=youtu.be) Firefox에서 `ctrl + shift + I` 키를 눌러 웹 개발자 도구를 실행시킵니다. `Network` 탭으로 이동하여 `Download` 버튼을 클릭하고 대화상자를 빠져나갑니다. 그러면 네트워크 연결이 표시됩니다. 그 후 마우스 오른쪽을 클릭하고 `Copy as cURL`을 선택합니다. 명령을 붙여넣고 끝에 `-o bulldozer.zip`을 추가합니다.<br><br>

#### 데이터 알아보기

* **구조적 데이터(Structured data):** 식별자, 날짜, 크기 등 다양한 유형의 항목을 나타내는 컬럼입니다.
* **비구조적 데이터(Unstructured data):** `pandas`는 `pd'로 import된 구조적 데이터로 작업할 때 가장 중요한 라이브러리입니다. <br><br>

``` 
df_raw = pd.read_csv(f'{PATH}Train.csv', low_memory=False, 
                     parse_dates=["saledate"])
``` 

* `parse_dates`는 해당 날짜를 포함하는 컬럼의 목록을 뜻합니다.
* `low_memory=False`는 타입을 결정하기 위해 파일을 더 읽어오도록 합니다.
<br>

``` 
def display_all(df):
    with pd.option_context("display.max_rows", 1000): 
        with pd.option_context("display.max_columns", 1000): 
            display(df)
display_all(df_raw.tail().transpose())
```


**질문**: 차원의 저주는 어떻습니까?[[38:16]](https://www.youtube.com/watch?t=38m16s&v=CzdWqFTmn0Y&feature=youtu.be) 여러분이 자주 듣는 두가지 개념 차원의 저주(curse of dimensionality)와 공짜 점심은 없다(No Free Lunch Theorem)가 있습니다.
그것들은 둘 다 대체로 무위미하고 기본적으로 어리석지만, 현장에 있는 많은 사람들이 그것을 알 뿐만 아니라 반대로 생각하므로 설명할 가치가 있습니다.
차원의 저주는 열이 많을수록 점점 더 비어 있는 공간이 생성된다는 생각입니다.
차원이 많을수록 모든 점이 그 공간의 가장자리에 더 많이 위치한다는 매혹적인 수학적 아이디어가 있습니다. 만약 사물이 무작위인 단일 차원일 경우에는 전체에 흩어져 있습니다.
다른 곳에서 만약 그것이 정사각형이라면 그것들이 중간에 있을 확률은 그것들이 어느 차원이 가장자리에 있을 수 없다는 것을 의미하므로 가장자리에 없을 가능성이 조금 더 적습니다.
각 차원을 추가하면 점이 적어도 한차원의 가장자리에 없을 확률이 곱셈적으로 낮아집니다. 따라서 높은 차원에서는 모든 것이 가장자리에 위치합니다.
이론적으로 그것이 의미하는 것은 점들 사의 거리가 훨씬 의미가 없다는 것이다
그래서 만약 우리가 그것이 중요하다고 가정한다면, 그것은 여러분이 많은 열을 가지고 있고 여러분이 신경 쓰지 않는 열을 제거하기 위해 조심하지 않고 그것들을 사용한다면, 그것이 작동하지 않을 것이라는 것을 암시할 것이다. 이것은 여러 가지 이유로 인해 사실이 아닌 것으로 밝혀졌다.

* 점들은 여전히 서로 다른 거리를 가지고 있습니다. 가장자리에 있기 때문에 서로 얼마나 멀리 떨어져 있는지에 따라 여전히 다르므로 이 지점은 저 지점보다 이 지점에서 더 유사합니다<br>
* 따라서 k-nearest neighbors과 같은 것들은 이론가들이 주장한 것과는 달리 고차원에서 실제로 잘 작동합니다. 여기서 실제로 일어난 일은 90년대에 이론이 기계 학습을 대신했다는 것입니다.
이러한 지원 벡터 머신의 개념은 이론적으로 충분히 정당화되었고, 수학적으로 분석하기도 쉬웠습니다. 그리고 여러분은 그들에 대한 것들을 증명할 수 있습니다. 그리고 우리는 10년간의 실질적인 발전을 잃었습니다. 그리고 이 모든 이론들은 차원의 저주처럼 매우 유명해졌습니다.
요즘 기계 학습의 세계는 매우 경험적이 되었고, 실제로 많은 열에 모델을 만드는 것이 정말 효과가 있다는 것이 밝혀졌습니다.<br>
* 공짜 점심은 없다(No Free Lunch Theorem)[[41:08]](https://www.youtube.com/watch?v=CzdWqFTmn0Y&t=2468s)- 그들의 주장은 어떤 데이터셋에도 잘 작동하는 유형의 모델이 없다는 것입니다.
수학적 의미에서, 정의상 임의의 데이터셋은 무작위이므로, 다른 접근 방식보다 더 유용하게 모든 가능한 무작위 데이터 셋을 볼 수 있는 방법을 없을 것이다.
실제로, 우리는 무작위가 아닌 데이터를 봅니다. 수학적으로 우리는 그것이 어떤 저차원 다양체에 위치한다고 말할 것이다. 그것은 일종의 인과 구조에 의해 만들어 졌습니다. 
여기에는 몇가지 관계가 있습니다, 사실 우리는 무작위 데이터셋을 사용하지 않기 때문에 실제로 여러분이 보고 있는 거의 모든 데이터셋은 다른 기술보다 훨씬 잘 작동하는 기술이 있습니다.  오늘날, 어떤 기술이 많은 시간 동안 효과가 있는지 연구하는 경험적 연구자들이 있습니다. 결정트리의 앙상블, 즉 무작위로 포레스트를 만드는 것은 아마도 가장 자주 맨 위에 있는 기술일 것입니다.
Fast.ai는 적절하게 전처리 하고 매개 변수를 설정하는 표준 방법을 제공합니다.<br>

## Scikit-learn[[42:54]](https://www.youtube.com/watch?v=CzdWqFTmn0Y&t=2574s)
파이썬에서 가장 인기있고 중요한 기계학습 패키지이다.
그것은 모든부분에서 가장 좋지는 않다.(예를 들면 XGBoost가 Gradient Boosting Tree보다 좋다) 하지만 거의 모든 부분에서 꽤 좋다.<br>
```
m = RandomForestRegressor(n_jobs=-1)
```
* RandomForestRegressor — 회귀 분석기는 연속 변수를 예측하는 방법(예: 회귀 분석)입니다.

* RandomForestClassifier — 분류자는 범주형 변수를 예측하는 방법입니다. (예: 분류)
```
m.fit(df_raw.drop('SalePrice', axis=1), df_raw.SalePrice)
```

Scikit-learn 의 모든 것은 같은 형태를 가지고 있습니다.
* 머신러닝 모델의 객체 인스턴스 생성 

* fit함수에 독립 변수(예측할 변수)와 종속 변수(예측할 변수)를 전달하며 콜한다.

* axis=1 은 열을 제거하는 것을 의미합니다.

* shift + tab은 Jupyter 노트북에서 함수의 파라미터 검사를 불러옵니다.

* "list-like"는 파이썬에서 색인화할 수 있는 모든 것을 의미합니다.

<img src="https://miro.medium.com/max/875/1*7tmiacXQ-X1x6KACz-L5Hw.png" width="650px" height="400px"></img><br>
위의 코드는 에러가 날것입니다. 
데이터 세트 내에 "Conventional" 값이 있으며, 해당 문자열을 사용하여 모델을 만드는 방법을 알지 못했습니다.
우리는 대부분의 기계 학습 모델과  랜덤 포레스트에 숫자를 전달해야 합니다.
그래서 1단계는 모든 것을 숫자로 변환하는 것입니다.

이 데이터 셋에는 연속형 변수와 범주형 변수가 혼합되어 있습니다.

* 연속형 – 가격과 같이 숫자를 의미합니다.
* 범주형 -  우편 번호와 같이 연속적이지 않은 숫자 또는 "크다", "중간", "작다"와 같은 문자열을 의미합니다.


날짜에서 추출할 수 있는 몇 가지 정보가 있습니다. 연도, 월, 분기, 달의 일, 주의 일, 년도의 주, 공휴일인지?, 주말인지? 비가오는지? 스포츠 행사가 있던 날인지?
그것은 정말로 당신이 무엇을 하느냐에 달려 있다.
만약 여러분이 SOMA의 탄산음료 판매를 예측하고 있다면, 여러분은 아마 그날 샌프란시스코 자이언츠 구기 경기가 있었는지 알고 싶을 것입니다
날짜 안에 있는 것은 여러분이 할 수 있는 가장 중요한 변수 가공 중 하나이고 어떤 기계 학습 알고리즘도 자이언츠가 그날 경기를 했는지와 그것이 중요했는지 알려줄 수 없습니다.


```add_datepart``` 메서드는 범주 구성을 위해 전체 날짜 시간에서 특정 날짜 필드를 추출합니다. 날짜와 시간으로 작업할 때는 항상 이 변수 추출 단계를 고려해야 합니다.
이러한 추가 필드로 날짜와 시간을 확장하지 않으면 이러한 세분화에서 시간 함수로 추세/사이클 동작을 캡처할 수 없습니다.


```
def add_datepart(df, fldname, drop=True):
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, 
                                     infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 
            'Dayofyear', 'Is_month_end', 'Is_month_start', 
            'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 
            'Is_year_start'):
        df[targ_pre+n] = getattr(fld.dt,n.lower())
    df[targ_pre+'Elapsed'] = fld.astype(np.int64) // 10**9
    if drop: df.drop(fldname, axis=1, inplace=True)
```
* ```getattr``` - 개체 내부를 살펴보고 해당 이름의 속성을 찾습니다.

* ```drop=True``` - 지정하지 않으면 숫자가 아니기 때문에 "saledate"를 직접 사용할 수 없기 때문에 날짜 시간 필드가 삭제됩니다.

```
fld = df_raw.saledate
fld.dt.year
```
* ```fld```- Pandas 시리즈

* ```dt``` – ```fld```는 시간 날짜 오브젝트인 pandas 시리즈에만 적용되기 때문에 “년도” 가 없습니다.
그래서 Pandas가 하는 일은 그것들이 무엇인지에 특정한 속성 내에서 다른 메소드들을 분리하는 것입니다.

```
add_datepart(df_raw, 'saledate')
df_raw.saleYear.head()
```

