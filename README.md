# 이미지 매칭 (Image Matching) - OpenCV

본 리드미는 OpenCV를 활용한 이미지 매칭 기법 중 **평균 해시 매칭(Average Hash Matching)**과 **템플릿 매칭(Template Matching)**을 정리한 자료입니다.  


---

## 1. 이미지 매칭(Image Matching)이란?

두 이미지에서 **유사한 객체**를 찾는 기술로, 이미지의 특징을 **특징 벡터**(디스크립터)로 추출하고 이를 비교하여 유사도를 측정합니다.

---

## 2. 평균 해시 매칭 (Average Hash Matching)

특징 벡터를 단순화하여 평균값을 기준으로 0과 1로 이진화한 후 비교합니다.  
간단하지만 회전, 스케일 변화에 민감합니다.

### 구현 절차
1. 이미지를 16x16으로 축소
2. 픽셀 평균값 계산
3. 평균보다 크면 1, 작으면 0 → 이진 벡터 생성
4. 이진값을 16진수 또는 벡터로 저장
5. **해밍 거리**로 유사도 측정

### 해밍 거리(Hamming Distance)
- 두 벡터에서 **서로 다른 비트 수**를 계산
- 값이 작을수록 유사도가 높음

### 예제 코드
- `avg_hash.py`: 단일 이미지의 평균 해시 출력
- `avg_hash_matching.py`: 101_ObjectCategories 이미지와 비교하여 유사한 이미지를 출력

### 실행 예시
```python
# 해밍 거리가 25% 이하인 경우 유사한 이미지로 간주
if hamming_distance(query_hash, target_hash) / 256 < 0.25:
    cv2.imshow(path, img)
````

---

## 3. 템플릿 매칭 (Template Matching)

작은 이미지를 기준으로 큰 이미지에서 해당 객체가 **어디에 있는지 위치를 찾는 방식**입니다.
**크기, 방향이 동일**한 경우에만 잘 작동합니다.

### 사용 함수

```python
cv2.matchTemplate(img, templ, method)
cv2.minMaxLoc(result)
```

### 주요 매칭 방법

| 메서드             | 의미              | 완벽 매칭 값 |
| --------------- | --------------- | ------- |
| `cv2.TM_SQDIFF` | 제곱 차이 (작을수록 좋음) | 0       |
| `cv2.TM_CCORR`  | 상관관계 (클수록 좋음)   | 1       |
| `cv2.TM_CCOEFF` | 상관계수 (클수록 좋음)   | 1       |
| `_NORMED` 버전    | 정규화된 방식         | -       |

### 예제 코드

* `template_matching.py`: 태권 V 이미지를 기준으로 전체 이미지에서 위치 탐색

```python
# 매칭 사각형 그리기
cv2.rectangle(img_draw, top_left, bottom_right, (0,0,255), 2)
cv2.putText(img_draw, str(match_val), top_left, ...)
```

---

## 4. 주의사항 및 한계점

* 평균 해시 매칭:

  * 단순하지만 밝기와 구조적 유사도만 판단
  * 회전/변형에 약함
* 템플릿 매칭:

  * 정확한 위치 찾기에 효과적
  * **크기나 회전 변화**에는 매우 민감
  * 느릴 수 있음

---

