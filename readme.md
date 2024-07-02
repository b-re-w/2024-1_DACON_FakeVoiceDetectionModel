# SW중심대학 디지털 경진대회_SW와 생성AI의 만남 : AI 부문
* Homepage: [dacon.io](https://dacon.io/competitions/official/236253/overview/description)

## 1. 대회 개요

### [주제선정 배경]
최근 생성 AI 기술의 발전으로 인해 가짜 음성 합성이 점점 더 정교해지고 있습니다. 이러한 가짜 음성은 기존의 텍스트 기반 가짜 정보 유포 문제에 더해 새로운 위협이 되고 있습니다. 가짜 음성을 통해 유명인의 음성을 모방하거나 중요 인사의 발언을 조작할 수 있기 때문입니다. 이는 개인 및 기업의 명예 실추, 금전적 피해, 사회적 혼란 등 다양한 문제를 야기할 수 있습니다.

따라서 가짜 음성을 신뢰할 수 있는 수준에서 검출하고 탐지할 수 있는 기술 개발이 시급한 상황입니다. 이를 통해 가짜 음성으로 인한 피해를 예방하고, 생성 AI 기술이 건전하게 활용될 수 있는 환경을 조성할 수 있을 것입니다.

또한 가짜 음성 탐지 기술은 음성인식, 스피커 인증, 대화 시스템 등 다양한 분야에서 활용될 수 있어 폭넓은 파급효과가 예상됩니다.

따라서 가짜 음성 검출 및 탐지 기술을 발전시킨다면 앞으로 대두될 수 있는 가짜 음성 문제에 선제적으로 대응할 수 있을 것입니다.

### [주제]
생성 AI의 가짜(Fake) 음성 검출 및 탐지

### [문제]
5초 분량의 입력 오디오 샘플에서 진짜(Real) 사람의 목소리와 생성 AI의 가짜(Fake) 사람의 목소리를 동시에 검출해내는 AI 모델을 개발해야합니다.

학습 데이터는 방음 환경에서 녹음된 진짜(Real) 사람의 목소리 샘플과 방음 환경을 가정한 가짜(Fake) 사람의 목소리로 구성되어 있으며, 각 샘플 당 사람의 목소리는 1개입니다.
평가 데이터는 5초 분량의 다양한 환경에서의 오디오 샘플로 구성되며, 샘플 당 최대 2개의 진짜(Real) 혹은 가짜(Fake) 사람의 목소리가 동시에 존재합니다.
Unlabel 데이터는 학습에 활용할 수 있지만 Label이 제공되지 않으며, 평가 데이터의 환경과 동일합니다.

---

## 2. 데이터 구성

#### Dataset Info.

### train [폴더]
55438개의 학습 가능한 오디오(ogg) 샘플
방음 환경에서 녹음된 진짜 사람 목소리(Real) 샘플과 방음 환경을 가정한 가짜 생성 목소리(Fake) 샘플
각 샘플 당 한명의 진짜 혹은 가짜 목소리가 존재


### test [폴더]
50000개의 5초 분량의 평가용 오디오(ogg) 샘플
TEST_00000.ogg ~ TEST_49999.png
방음 환경 혹은 방음 환경이 아닌 환경 모두 존재하며, 각 샘플 당 최대 2명의 진짜 혹은 가짜 목소리가 존재


### unlabeled_data [파일]
1264개의 5초 분량의 학습 가능한 Unlabeled 오디오(ogg) 샘플
평가용 오디오(ogg) 샘플과 동일한 환경에서 녹음되었지만 Unlabeled 데이터로 제공


### train.csv [파일]
id : 오디오 샘플 ID
path : 오디오 샘플 경로
label : 진짜(real) 혹은 가짜(fake) 음성의 Class


### test.csv [파일]
id : 평가용 오디오 샘플 ID
path : 평가용 오디오 샘플 경로


### sample_submission.csv [파일] - 제출 양식
id : 평가용 오디오 샘플 ID
fake : 해당 샘플에 가짜 목소리가 존재할 확률 (0~1)
real : 해당 샘플에 진짜 목소리가 존재할 확률 (0~1)

---

## 3. 평가 산식 안내
이번 'SW중심대학 디지털 경진대회_SW와 생성AI의 만남: AI 부문'에 실제 적용되고 있는 평가 산식 코드를 공개합니다.

참가자 여러분들께서는 모델 성능 개선에 해당 평가 산식 코드를 활용하실 수 있습니다.


이번 산식은 여러 평가 지표를 조합하여 분류 문제에서의 모델의 예측 성능과 예측의 신뢰도를 함께 고려하여 평가하기 위해 구성되었습니다.

산식은 AUC(Area Under the Curve), Brier Score, ECE(Expected Calibration Error) 세 가지 주요 성능 지표를 결합합니다.

<underline>산식은 0 ~ 1 의 범위로 산출되며, 산식 점수가 0 에 가까울수록 좋은 모델 성능을 뜻합니다.</underline>

```python
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
import pandas as pd


def expected_calibration_error(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    bin_totals = np.histogram(y_prob, bins=np.linspace(0, 1, n_bins + 1), density=False)[0]
    non_empty_bins = bin_totals > 0
    bin_weights = bin_totals / len(y_prob)
    bin_weights = bin_weights[non_empty_bins]
    prob_true = prob_true[:len(bin_weights)]
    prob_pred = prob_pred[:len(bin_weights)]
    ece = np.sum(bin_weights * np.abs(prob_true - prob_pred))
    return ece
    
def auc_brier_ece(answer_df, submission_df):
    # Check for missing values in submission_df
    if submission_df.isnull().values.any():
        raise ValueError("The submission dataframe contains missing values.")


    # Check if the number and names of columns are the same in both dataframes
    if len(answer_df.columns) != len(submission_df.columns) or not all(answer_df.columns == submission_df.columns):
        raise ValueError("The columns of the answer and submission dataframes do not match.")
        
    submission_df = submission_df[submission_df.iloc[:, 0].isin(answer_df.iloc[:, 0])]
    submission_df.index = range(submission_df.shape[0])
    
    # Calculate AUC for each class
    auc_scores = []
    for column in answer_df.columns[1:]:
        y_true = answer_df[column]
        y_scores = submission_df[column]
        auc = roc_auc_score(y_true, y_scores)
        auc_scores.append(auc)


    # Calculate mean AUC
    mean_auc = np.mean(auc_scores)


    brier_scores = []
    ece_scores = []
    
    # Calculate Brier Score and ECE for each class
    for column in answer_df.columns[1:]:
        y_true = answer_df[column].values
        y_prob = submission_df[column].values
        
        # Brier Score
        brier = mean_squared_error(y_true, y_prob)
        brier_scores.append(brier)
        
        # ECE
        ece = expected_calibration_error(y_true, y_prob)
        ece_scores.append(ece)
    
    # Calculate mean Brier Score and mean ECE
    mean_brier = np.mean(brier_scores)
    mean_ece = np.mean(ece_scores)
    
    # Calculate combined score
    combined_score = 0.5 * (1 - mean_auc) + 0.25 * mean_brier + 0.25 * mean_ece
    
    return combined_score
```
