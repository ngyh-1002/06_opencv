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
네, 이 자료도 특징 디스크립터(SIFT, SURF, ORB)에 대한 아주 체계적인 정리입니다. 내용을 요약하면 다음과 같습니다:

---

## 📌 핵심 요약

### 🧠 **특징점(keypoint)이란?**

* 이미지에서 **두드러지는 지점**, 예: 코너, 경계 등.
* 매칭할 때 서로 비교되는 "기준점".

### 🔍 **특징 디스크립터(descriptor)란?**

* 특징점 주변 픽셀의 **방향, 밝기, 경사도** 등의 정보를 **벡터**로 표현.
* 예: 4x4 블록 × 8방향 = 128차원의 벡터 (SIFT 기준)

---

## 💡 특징 디스크립터 검출 알고리즘 비교

| 항목       | SIFT                              | SURF                       | ORB                        |
| -------- | --------------------------------- | -------------------------- | -------------------------- |
| 정식 이름    | Scale-Invariant Feature Transform | Speeded Up Robust Features | Oriented and Rotated BRIEF |
| 특성       | 크기, 회전 불변<br>정확도 높음               | 속도 개선, 필터 크기 변화            | 속도 빠름, 이진 디스크립터            |
| 속도       | 느림                                | 보통                         | 빠름                         |
| 디스크립터 길이 | 128 (float)                       | 64 또는 128 (float)          | 32 (binary)                |
| 사용 가능 여부 | OpenCV-contrib 필요                 | OpenCV-contrib 필요          | 기본 OpenCV 포함               |
| 라이선스     | 특허 문제 있음                          | 특허 문제 있음                   | 오픈소스, 자유 사용 가능             |

---

## 🧪 예제 코드 주요 흐름

공통 단계:

1. 이미지 읽기
2. `gray = cv2.cvtColor(...)` → 그레이스케일 변환
3. `detector.detectAndCompute()` → 특징점 + 디스크립터 추출
4. `cv2.drawKeypoints()` → 시각화
5. `cv2.imshow()` → 출력

각 알고리즘 별 생성자 예시:

### ✅ SIFT

```python
sift = cv2.xfeatures2d.SIFT_create()
keypoints, descriptor = sift.detectAndCompute(gray, None)
```

### ✅ SURF

```python
surf = cv2.xfeatures2d.SURF_create(1000, 3, True, True)
keypoints, descriptor = surf.detectAndCompute(gray, None)
```

### ✅ ORB

```python
orb = cv2.ORB_create()
keypoints, descriptor = orb.detectAndCompute(gray, None)
```

---

## ⚠️ 주의 사항

* **SIFT와 SURF는 OpenCV의 `contrib` 모듈**을 설치해야 사용 가능:

  ```bash
  pip install opencv-contrib-python
  ```
* ORB는 기본 OpenCV로도 작동.

---

## 📌 실제 프로젝트 적용 팁

* 실시간 시스템, 속도 중요 → **ORB**
* 정확도, 복잡한 환경 대응 → **SIFT 또는 SURF**
* GPU 지원 필요시 OpenCV CUDA 또는 Deep Learning 기반 대안 고려 (e.g. SuperPoint, D2-Net 등)

---


---

## 🧠 OpenCV Feature Matching Summary

이 문서는 OpenCV를 활용한 **특징 매칭(Feature Matching)** 기법을 간단히 정리한 것입니다. 주로 `SIFT`, `SURF`, `ORB` 디스크립터를 사용하며, 매칭 알고리즘은 `BFMatcher`와 `FlannBasedMatcher` 두 가지를 중심으로 다룹니다.

---

### 📌 특징 매칭이란?

두 이미지에서 특징점과 디스크립터를 추출하고, 이들을 비교하여 유사한 부분을 짝지어 객체를 식별하는 기법입니다.

---

### 🔧 특징 매칭 함수

OpenCV에서 제공하는 주요 매칭 함수:

| 함수                         | 설명                 |
| -------------------------- | ------------------ |
| `match()`                  | 가장 유사한 하나의 매칭 반환   |
| `knnMatch(k=2)`            | k개의 최근접 이웃 반환      |
| `radiusMatch(maxDistance)` | 특정 거리 이내의 디스크립터 반환 |

---

### 🔍 DMatch 객체

각 매칭은 `cv2.DMatch`로 표현됨:

* `queryIdx`: 기준 이미지 디스크립터 인덱스
* `trainIdx`: 대상 이미지 디스크립터 인덱스
* `distance`: 디스크립터 간 거리

---

### 🖼️ 매칭 시각화

```python
cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=...)
```

* `DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS`: 한쪽만 있는 매칭 생략
* `DRAW_RICH_KEYPOINTS`: 키포인트 크기/방향 표시

---

## 💥 매칭 방법별 예제

---

### ✅ 1. Brute-Force Matcher (`BFMatcher`)

모든 디스크립터를 전수 비교하는 방식. 정확도 높지만 느림.

#### 🎯 사용법:

```python
matcher = cv2.BFMatcher(normType, crossCheck=True)
matches = matcher.match(desc1, desc2)
```

* `SIFT`, `SURF`: `NORM_L1`, `NORM_L2`
* `ORB`: `NORM_HAMMING`, `NORM_HAMMING2`



---

### 🚀 2. FLANN Matcher (`FlannBasedMatcher`)

근사 최근접 검색을 위한 빠른 매칭. 대용량 이미지에 유리.

#### 🎯 사용법:

```python
matcher = cv2.FlannBasedMatcher(indexParams, searchParams)
matches = matcher.match(desc1, desc2)
```

#### 🛠️ 설정값

| 디스크립터       | index\_params 설정                                                            |
| ----------- | --------------------------------------------------------------------------- |
| SIFT / SURF | `algorithm=1` (KDTree), `trees=5`                                           |
| ORB         | `algorithm=6` (LSH), `table_number=6`, `key_size=12`, `multi_probe_level=1` |


---

## ⚠️ 잘못된 매칭 제거

* 단순히 매칭된 결과 중 일부는 오류일 수 있음
* 예: 전혀 관련 없는 객체 간 매칭
* **후처리 (예: Lowe’s ratio test 등)** 필요 → 다음 단계에서 다룸

---

## 🧠 Feature Matching with OpenCV

본 문서는 OpenCV에서 제공하는 **특징 매칭(Feature Matching)** 알고리즘의 핵심 내용을 정리한 것입니다. 특히, SIFT, SURF, ORB 등의 디스크립터와 함께 `BFMatcher`, `FlannBasedMatcher`를 사용하여 두 이미지 간의 유사한 부분을 효과적으로 찾는 방법을 다룹니다.

---

### 📌 특징 매칭이란?

두 이미지에서 특징점(Feature point)을 추출하고, 각 특징점의 디스크립터를 기반으로 유사한 점끼리 매칭하여 물체나 장면을 식별하는 기법입니다.

---

### 🔧 주요 함수 설명

| 함수                         | 설명                        |
| -------------------------- | ------------------------- |
| `match()`                  | 가장 유사한 하나의 매칭을 반환         |
| `knnMatch(k=2)`            | 각 디스크립터에 대해 k개의 최근접 이웃 반환 |
| `radiusMatch(maxDistance)` | 지정된 거리 이내의 매칭만 반환         |

---

### 🔎 DMatch 객체란?

각 매칭은 `cv2.DMatch` 객체로 표현되며 다음과 같은 정보를 포함합니다:

* `queryIdx`: 기준 이미지의 디스크립터 인덱스
* `trainIdx`: 대상 이미지의 디스크립터 인덱스
* `distance`: 디스크립터 간 거리 (값이 작을수록 유사)

---

### 🖼️ 매칭 결과 시각화

```python
cv2.drawMatches(img1, kp1, img2, kp2, matches, outImg, flags=...)
```

* `DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS`: 매칭되지 않은 점 생략
* `DRAW_RICH_KEYPOINTS`: 특징점의 크기와 방향 표시

---

## ✅ Brute-Force Matcher (BFMatcher)

* 모든 디스크립터를 비교하여 가장 유사한 것 찾기
* 정확도는 높지만 느림
* `crossCheck=True`로 설정하면 양방향 검증을 통해 매칭 품질 향상

### 📌 normType 설정

| 디스크립터     | normType                                  |
| --------- | ----------------------------------------- |
| SIFT/SURF | `cv2.NORM_L1` 또는 `cv2.NORM_L2`            |
| ORB       | `cv2.NORM_HAMMING` 또는 `cv2.NORM_HAMMING2` |

### 💻 사용 예시

```python
bf = cv2.BFMatcher(normType, crossCheck=True)
matches = bf.match(desc1, desc2)
matches = sorted(matches, key=lambda x: x.distance)
```

---

## 🚀 FLANN-Based Matcher (FlannBasedMatcher)

* 큰 이미지나 많은 디스크립터를 다룰 때 빠르고 효율적
* **근사 최근접 이웃(Approximate Nearest Neighbor)** 탐색 사용

### 📌 index\_params 설정

| 디스크립터     | indexParams 설정                                                                                 |
| --------- | ---------------------------------------------------------------------------------------------- |
| SIFT/SURF | `{'algorithm':1, 'trees':5}` (`algorithm=1`: KDTree)                                           |
| ORB       | `{'algorithm':6, 'table_number':6, 'key_size':12, 'multi_probe_level':1}` (`algorithm=6`: LSH) |

### 💻 사용 예시

```python
flann = cv2.FlannBasedMatcher(indexParams, searchParams)
matches = flann.match(desc1, desc2)
matches = sorted(matches, key=lambda x: x.distance)
```

---

## ⚠️ 잘못된 매칭 제거

특징 매칭만으로는 잘못된 결과가 포함될 수 있습니다. 따라서 **후처리**가 필요합니다:

* **Lowe's Ratio Test** 적용
* **RANSAC**을 통한 정합성 확인
* **거리 임계값**으로 필터링

---

### 📌 Background Subtraction (배경 제거) - OpenCV

배경 제거(Background Subtraction)는 **영상에서 움직이는 객체(전경)를 검출하기 위해 배경을 제거**하는 기법입니다.
이 방법은 객체 추적(Object Tracking)의 전처리 과정으로 매우 중요하며, OpenCV에서는 다음과 같은 두 가지 대표적인 알고리즘이 사용됩니다:

---

### 🔧 1. `BackgroundSubtractorMOG`

* **함수**: `cv2.bgsegm.createBackgroundSubtractorMOG()`
* **기반 논문**: KadewTraKuPong and Bowden (2001)
* **설정 가능 파라미터**:

  * `history=200`: 히스토리 프레임 수
  * `nmixtures=5`: 가우시안 혼합 개수
  * `backgroundRatio=0.7`: 배경 비율
  * `noiseSigma=0`: 노이즈 강도 (0이면 자동)
* **특징**:

  * 간단하고 빠름
  * 그림자 탐지 불가
* **사용 예시**:

```python
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
fgmask = fgbg.apply(frame)
```

---

### 🔧 2. `BackgroundSubtractorMOG2`

* **함수**: `cv2.createBackgroundSubtractorMOG2()`
* **기반 논문**: Zivkovic (2004, 2006)
* **설정 가능 파라미터**:

  * `history=500`: 히스토리 프레임 수
  * `varThreshold=16`: 픽셀 분산 임계값
  * `detectShadows=True`: 그림자 탐지 여부
* **특징**:

  * 그림자 탐지 가능 (회색으로 표시됨)
  * 빛 변화에 강함
* **사용 예시**:

```python
fgbg = cv2.createBackgroundSubtractorMOG2()
fgmask = fgbg.apply(frame)
```

---

### ▶️ 공통 사용 방법

```python
import cv2

cap = cv2.VideoCapture('walking.avi')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()  # 또는 createBackgroundSubtractorMOG2()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    fgmask = fgbg.apply(frame)
    cv2.imshow('Original', frame)
    cv2.imshow('Foreground Mask', fgmask)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키로 종료
        break

cap.release()
cv2.destroyAllWindows()
```

---

### ✅ 참고

* `MOG2`는 `MOG`보다 더 진보된 방식이며, **조명 변화**와 **그림자 처리**가 필요한 환경에서 유리합니다.
* 간단한 처리 속도를 원한다면 `MOG`, 정교한 추적이 필요하다면 `MOG2`를 추천합니다.

--

---
# 🎥 Optical Flow (광학 흐름) - OpenCV

> 객체 추적(Object Tracking)을 위한 영상 처리 기법  
> 참고 자료: Baek Kyun Shin 블로그, *파이썬으로 만드는 OpenCV 프로젝트* (이세우 저)

---

## 📌 Optical Flow란?

- **광학 흐름(Optical Flow)**: 연속된 영상에서 **픽셀의 이동 방향과 크기**를 추정하는 기술  
- 예: 사람이 걷는 영상에서, 각 프레임마다 사람의 위치 변화량을 추적 가능

### ✅ 가정 조건
1. **픽셀 밝기 불변**: 시간 흐름에도 픽셀의 밝기(intensity)는 일정하다.
2. **이웃 픽셀의 유사 움직임**: 인접한 픽셀은 비슷하게 움직인다.

---

## 🧠 Optical Flow 방식

### 🔹 1. 희소 광학 흐름 (Sparse Optical Flow)
- **일부 특징점에 대해서만** 움직임을 추적
- 대표 알고리즘: `Lucas-Kanade`
- 특징점 검출: `cv2.goodFeaturesToTrack()`
- 흐름 계산: `cv2.calcOpticalFlowPyrLK()`

#### 📄 주요 코드

```python
prevPt = cv2.goodFeaturesToTrack(prevImg, maxCorners=200, qualityLevel=0.01, minDistance=10)
nextPt, status, err = cv2.calcOpticalFlowPyrLK(prevImg, nextImg, prevPt, None, criteria=term_criteria)
````

* **status==1**: 추적 성공한 점
* **term\_criteria**: 반복 종료 조건 설정

#### 📌 장점 / 단점

| 장점     | 단점                          |
| ------ | --------------------------- |
| 빠름     | 전체 장면 추적 불가                 |
| 적은 계산량 | 큰 움직임에 약함 (이미지 피라미드로 보완 가능) |

---

### 🔹 2. 밀집 광학 흐름 (Dense Optical Flow)

* **영상의 모든 픽셀**을 대상으로 흐름 추정
* 대표 알고리즘: `Farneback` (cv2.calcOpticalFlowFarneback)

#### 📄 주요 코드

```python
flow = cv2.calcOpticalFlowFarneback(prev, next, None,
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.1,
    flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
```

* `flow[y, x]`: (dx, dy), 픽셀 이동 벡터

#### 📌 시각화

```python
cv2.line(frame, (x, y), (x+dx, y+dy), (0,255,0), 2)
```

* 격자(grid) 간격을 두고 화살표(선)로 시각화

#### 📌 장점 / 단점

| 장점            | 단점     |
| ------------- | ------ |
| 전체 움직임 파악 가능  | 속도 느림  |
| 별도 특징점 검출 불필요 | 계산량 많음 |

---

## 🔍 정리 비교

| 항목        | Lucas-Kanade                 | Farneback                        |
| --------- | ---------------------------- | -------------------------------- |
| 방식        | 희소 (Sparse)                  | 밀집 (Dense)                       |
| 계산 대상     | 특징점                          | 모든 픽셀                            |
| 속도        | 빠름                           | 느림                               |
| 정밀도       | 중간                           | 높음                               |
| OpenCV 함수 | `cv2.calcOpticalFlowPyrLK()` | `cv2.calcOpticalFlowFarneback()` |

---

## ▶️ 예제 영상: `walking.avi`

* **희소 광학 흐름**: 움직이는 사람의 특징점만 추적
* **밀집 광학 흐름**: 전체 프레임 내 픽셀 이동량 시각화

---

## 🏁 Tip

* **추적 리셋**: ESC 키 또는 Backspace 키 활용 (예제 코드 내 구현)
* **윈도우 크기 조절**: `winSize`, `maxLevel` 등을 조절해 성능 최적화 가능

---


