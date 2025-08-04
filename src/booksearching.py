# 책 이미지 중에서 일치하는 책 찾기 (avg_hash_matching.py)

import cv2
import numpy as np
import glob
import time


# 영상 읽기 및 표시

# 비교할 영상들이 있는 경로 ---①
search_dir = '../img/books'

results = {}
# 초기 설정
img1 = None #ROI로 선택할 이미지
win_name = 'Camera Matching' # 윈도우 이름
MIN_MATCH = 10 # 최소 매칭점 개수(이 값 이사면 매칭실패로 간주)

# ORB 검출기 생성
# ORB_create(1000)은 이미지에서 1000개의 특징점을 찾는 알고리즘
detector = cv2.ORB_create(5000)

# Flann 추출기 생성
# 두 이미지의 특징점을 빠르게 매칭
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                   table_number=6,
                   key_size=12,
                   multi_probe_level=1)
search_params = dict(checks=32)
matcher = cv2.FlannBasedMatcher(index_params, search_params)

# 카메라 캡쳐 연결 및 프레임 크기 축소
cap = cv2.VideoCapture(0)              
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
break_outer_loop = False

while cap.isOpened() and not break_outer_loop:       
    ret, frame = cap.read() 
    if not ret:
        break
        
    if img1 is None:  # 등록된 이미지 없음, 카메라 바이패스
        res = frame
    else:             # 등록된 이미지 있는 경우, 매칭 시작

        start_time = time.time()

        # [step1]
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # 참조 이미지
        
        #[step2]
        # 키포인트와 디스크립터 추출
        # kp : keypoint 특징점의 위치정보
        # desc: 특징점의 특성을 숫자로 표현
        kp1, desc1 = detector.detectAndCompute(gray1, None) # 참조 이미지의 특징점

        # 이미지 데이타 셋 디렉토리의 모든 영상 파일 경로 ---⑤
        img_path = glob.glob(search_dir+'/*.jpg')
        for path in img_path:
            # 데이타 셋 영상 한개 읽어서 표시 ---⑥
            compared_img = cv2.imread(path)
            cv2.imshow('searching...', compared_img)
            cv2.waitKey(5)
            gray2 = cv2.cvtColor(compared_img, cv2.COLOR_BGR2GRAY) # 비교 이미지
            kp2, desc2 = detector.detectAndCompute(gray2, None) # 비교 이미지의 특징점

            matches = matcher.knnMatch(desc1, desc2, 2)
            
            # [step4]
            # 이웃 거리의 75%로 좋은 매칭점 추출
            ratio = 0.75
            # Lowe's 비율 테스트로 좋은 매칭 선별 ---⑨

            good_matches = [m[0] for m in matches \

                        if len(m) == 2 and m[0].distance < m[1].distance * ratio]
            
            
            # 좋은 매칭점 최소 갯수 이상인 경우
            if len(good_matches) > MIN_MATCH: 
                # 좋은 매칭점으로 원본과 대상 영상의 좌표 구하기
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # 원근 변환 행렬 구하기
                # RANSAC은 잘못된 매칭점들 outline 제거
                mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if mtrx is not None:
                    accuracy = float(mask.sum()) / mask.size

                    results[path] = accuracy

                        # 정확도 기준으로 결과 정렬

        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        for i, (img_path, accuracy) in enumerate(sorted_results):

            print(f"{i}: {img_path} - 정확도: {accuracy:.2%}")

            

            if i == 0:  # 가장 높은 정확도의 결과 표시

                cover = cv2.imread(img_path)

                cv2.putText(cover, f"Accuracy: {accuracy*100:.2f}%", 

                           (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, 

                           (0,255,0), 2, cv2.LINE_AA)

                cv2.imshow('Result', cover)
                search_time = time.time() - start_time
                cv2.waitKey(0)  # 사용자가 아무 키나 누를 때까지 대기
                break_outer_loop = True
                print(f"검색 시간: {search_time:.2f}초")
                break





                   

    
    # 결과 출력
    cv2.imshow(win_name, res)
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:    # Esc, 종료
        break          
    elif key == ord(' '):  # 스페이스바를 누르면 ROI로 img1 설정
        x, y, w, h = cv2.selectROI(win_name, frame, False)
        if w and h:
            img1 = frame[y:y+h, x:x+w]
            print("ROI 선택됨: (%d, %d, %d, %d)" % (x, y, w, h))

cap.release()                          
cv2.destroyAllWindows()
