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


# 이미지 특징점 및 특징점 검출기 - OpenCV

이 리드미는 OpenCV를 활용한 다양한 **이미지 특징점(Keypoints)** 검출 알고리즘과 사용법을 정리한 문서입니다.  

---

## 1. 특징점(Keypoints)이란?

- 이미지 내 **모서리, 꼭짓점, 특정 패턴** 등 **특징이 되는 점**을 의미
- 매칭, 추적, 객체 인식 등 다양한 컴퓨터 비전 작업의 핵심
- 보통 코너 검출(Corner Detection)을 바탕으로 함

---

## 2. 해리스 코너 검출 (Harris Corner Detection)

고전적인 코너 검출 알고리즘으로, 이미지 경계의 **수직/수평/대각선 변화량**을 바탕으로 코너를 감지합니다.

```python
cv2.cornerHarris(src, blockSize, ksize, k)
````

* `blockSize`: 코너 검출 시 주변 픽셀 블록 크기
* `ksize`: 소벨(Sobel) 필터 크기
* `k`: 코너 응답 계산 상수 (보통 0.04\~0.06)

✅ 결과는 코너 강도 값 배열이며, 지역 최대값이 코너로 판단됨

---

## 3. 시-토마시 검출 (Shi & Tomasi Detection)

해리스 방법을 개선하여 **강한 코너만 선별적으로** 검출하는 방식.

```python
cv2.goodFeaturesToTrack(img, maxCorners, qualityLevel, minDistance)
```

* `qualityLevel`: 전체 중 코너로 간주할 최소 품질 비율
* `minDistance`: 코너 간 최소 거리
* `useHarrisDetector=True` 시 해리스 방식 사용

📌 결과는 좌표값만 제공 (`N x 1 x 2` 배열)

---

## 4. 특징점 검출기 (Keypoints Detector)

OpenCV는 다양한 검출기를 제공하며, 아래 함수로 공통적으로 사용합니다:

```python
keypoints = detector.detect(img[, mask])
```

결과는 `cv2.KeyPoint` 객체 리스트이며, 각 특징점은 다음 속성을 가짐:

* `pt`: (x, y) 좌표
* `size`, `angle`: 크기와 방향
* `response`, `octave`, `class_id`: 검출기별 정보

📍 시각화: `cv2.drawKeypoints(img, keypoints, ...)`

---

## 5. GFTTDetector

`goodFeaturesToTrack()` 기반의 검출기.

```python
cv2.GFTTDetector_create()
```

* 단순히 코너 좌표만 반환
* 속도 빠름, 비교적 많은 특징점 검출

---

## 6. FAST (Features from Accelerated Segment Test)

**속도 중심의 검출기**, 미분 연산 없이 픽셀 밝기 패턴으로 특징점 판단

```python
cv2.FastFeatureDetector_create(threshold, nonmaxSuppression, type)
```

* `threshold`: 중심 픽셀 대비 임계 밝기 차
* `type`: 코너 판단 픽셀 수 (TYPE\_9\_16, TYPE\_7\_12 등)
* `nonmaxSuppression`: 비최대 억제 사용 여부

특징점 주변에 원을 그리며, 밝기 변화 패턴으로 검출

---

## 7. SimpleBlobDetector

특정 크기 이상의 연속된 픽셀 블록(BLOB)을 검출
자잘한 노이즈는 무시하며, **큰 객체에 적합**

```python
cv2.SimpleBlobDetector_create([params])
```

### 주요 파라미터 (`cv2.SimpleBlobDetector_Params()`):

* `filterByArea`, `minArea`, `maxArea`: 면적 기반 필터링
* `filterByCircularity`: 원형 비율 필터링
* `filterByColor`, `blobColor`: 밝기 조건
* `filterByConvexity`, `filterByInertia`: 형태 기반 필터

💡 필터를 적절히 설정하면 **특정 구조**에 더 잘 대응 가능

---

## 8. 요약 비교표

| 검출기          | 특징점 수 | 속도    | 특징 정보       | 회전/스케일 불변성 |
| ------------ | ----- | ----- | ----------- | ---------- |
| Harris       | 보통    | 보통    | 위치만         | ❌          |
| Shi-Tomasi   | 많음    | 보통    | 위치만         | ❌          |
| GFTTDetector | 많음    | 빠름    | 위치만         | ❌          |
| FAST         | 매우 많음 | 매우 빠름 | 위치/반응       | ❌          |
| SimpleBlob   | 선택적   | 느림    | 크기 등 다양한 속성 | ❌          |

---

