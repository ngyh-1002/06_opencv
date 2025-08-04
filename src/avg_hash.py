# 권총 이미지를 평균 해시로 변환
import cv2

img = cv2.imread('../img/pistol.jpg')
# 이미지를 읽어서 그레이스케일로 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 16x16 크기로 축소
gray = cv2.resize(gray, (16,16))

# 영상의 평균값을 구하기
avg = gray.mean()
#평균값을 기준으로 0과 1로 변환
bin = 1 * (gray >avg)
print(bin)

# 2진수 문자여ㅑㄹ을 16진수 문자열로 변환
dhash = []
for row in bin.tolist():
    s = ''.join([str(i) for i in row])
    dhash.append('%02x'%(int(s,2)))
dhash.append(''.join(dhash))
print(dhash)

cv2.namedWindow('pistol', cv2.WINDOW_GUI_NORMAL)
cv2.imshow('pistol',img)
cv2.waitKey(0)