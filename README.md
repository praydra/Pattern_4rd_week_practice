# 신경망 학습 문제와 해결 기법 실습 (TensorFlow)

신경망 학습에서 자주 생기는 문제(기울기 소실, 과적합, 클래스 불균형)를 직접 재현하고, 해결 기법(ReLU, Residual Connection, L1/L2 정규화, Dropout, 데이터 증강, 클래스 가중치)의 효과를 비교한 노트북 모음입니다.

## 데이터셋

모든 데이터는 `sklearn.datasets` 또는 `tf.keras.datasets`에서 불러오거나 생성합니다.

| 데이터 | 사용 노트북 |
| --- | --- |
| California Housing | 1 |
| make_classification, Iris, make_multilabel_classification | 2, 5 |
| MNIST | 2-1 |
| make_moons (100~2,000개, noise 0.2~0.25) | 3, 4-1, 4-2, 6, 7, 8 |
| CIFAR-10 (Cat vs Dog, Cat을 20%만 사용해 불균형 생성) | 9 |

## 소스 구성 및 분석 내용

| 노트북 | 주제 | 실험 내용 |
| --- | --- | --- |
| `1.nn_regression_loss_comparison` | 회귀 손실 함수 | MSE, MAE, Huber 손실로 같은 회귀 모델 학습 후 비교(20 epoch) |
| `2.nn_classification_examples` | 분류 유형 | 이진, 다중 클래스, 다중 라벨 분류의 출력층과 손실 함수 구성 |
| `2-1.nn_multiclass_classification` | 다중 클래스 분류 | MNIST, Dense 256-128-10, Adam(lr=0.001) |
| `3.vanishing_gradient_demo` | 기울기 소실 | 20층 vs 3층 Sigmoid 네트워크 비교, ReLU 적용 깊은 네트워크와 비교 |
| `4-1.overfitting_dataset_size_demo` | 과적합 (데이터 크기) | 같은 복잡한 모델을 100개 vs 2,000개 데이터로 학습 |
| `4-2.overfitting_demo` | 과적합 (모델 복잡도) | 단순 모델 vs 복잡한 모델의 학습/검증 곡선 비교 |
| `5.class_imbalanced_demo` | 클래스 불균형 | 9:1 불균형 데이터로 학습 후 Confusion Matrix, Classification Report 확인 |
| `6.residual_vs_plain_demo` | Residual Connection | Plain 깊은 네트워크 vs Residual Block 네트워크 |
| `7.l1_l2_regularization_demo` | 가중치 정규화 | 정규화 없음 / L1 / L2 / L1+L2 (0.001), 200 epoch |
| `8.dropout_demo` | Dropout | Dropout 적용 유무 비교, 200 epoch |
| `9.data_aug_class_weight_demo` | 불균형 해결 | Baseline vs 소수 클래스 증강 vs 클래스 가중치({0: 3.0, 1: 0.6}), CNN |

## 결과

| 실습 | 결과 |
| --- | --- |
| 손실 함수 비교 (검증 MAE) | MSE 0.367 / MAE 0.355 / Huber 0.352 |
| MNIST 다중 클래스 분류 | 테스트 정확도 0.977 |
| 클래스 불균형 (불균형 테스트셋 600개) | 정확도 0.950. 다수 클래스 F1 0.972, 소수 클래스 F1 0.754 |
| 클래스 불균형 (균형 테스트셋 100개) | 정확도 0.360. 소수 클래스 재현율 0.140 |
| CIFAR-10 Cat/Dog 불균형 해결 (테스트 2,000개) | Baseline 0.534 → 증강 0.669 → 클래스 가중치 0.691 (정확도) |
| CIFAR-10 소수 클래스(Cat) 재현율 | Baseline 0.080 → 증강 0.453 → 클래스 가중치 0.688 |

- 클래스 불균형 실험: 불균형 테스트셋에서는 정확도가 높게 나오지만, 균형 테스트셋에서는 성능이 크게 떨어졌습니다. 정확도만으로는 불균형 문제가 드러나지 않습니다.
- CIFAR-10 실험: 클래스 가중치를 적용했을 때 두 클래스의 F1이 약 0.69로 가장 균형 잡힌 결과가 나왔습니다.
- 기울기 소실, 과적합, Residual, 정규화, Dropout 실험의 결과는 각 노트북의 학습/검증 손실·정확도 곡선으로 확인할 수 있습니다.

## 실행 방법

```bash
pip install tensorflow scikit-learn numpy matplotlib
jupyter notebook
```
