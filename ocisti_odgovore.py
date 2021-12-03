import cv2
import numpy as np

img = cv2.imread('ispit.png')
kvacica = cv2.imread('kvacica.png')
bezkvacice = cv2.imread('bezkvacice.png')
w, h = kvacica.shape[:-1]

res = cv2.matchTemplate(img, kvacica, cv2.TM_CCOEFF_NORMED) + cv2.matchTemplate(img, bezkvacice, cv2.TM_CCOEFF_NORMED)
threshold = .8
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (255, 255, 255), -10)

cv2.imwrite('result.png', img)