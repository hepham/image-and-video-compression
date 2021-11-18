import cv2
import numpy as np
import matplotlib.image as mpimg
def dct(input_matrix):
    return cv2.dct(input_matrix)
img_gray = cv2.imread('lena256color.jpg', 0)
print(img_gray)
img_float = img_gray.astype('float')
img_dct = cv2.dct(img_float)
img_dct_log = np.log(abs(img_dct))
print(img_dct_log)
img_idct1 = cv2.idct(img_dct)
print(img_idct1)
inf = 50
recor_temp = img_dct[0:inf, 0:inf]
recor_temp2 = np.zeros(img_gray.shape)
recor_temp2[0:inf, 0:inf] = recor_temp
img_idct2 = cv2.idct(recor_temp2)
img_idct3 = np.zeros(img_gray.shape)
print(recor_temp2.shape)
keep_info = 4
(h, w) = img_gray.shape
for i in range(0, h, 8):
    for j in range(0, w, 8):
        newblock = np.zeros((8, 8))
        newblock[0:min(8, h-i), 0:min(8, w-j)] = img_gray[i:min(i + 8, h), j:min(j + 8, w)]
        newblock_dct = cv2.dct(newblock)
        for k in range(0, 8):
            for l in range(0, 8):
                if k >= keep_info or l >= keep_info:
                    newblock_dct[k, l] = 0
        newblock_idct = cv2.idct(newblock_dct)
        img_idct3[i:min(i + 8, h), j:min(j + 8, w)] = newblock_idct[0:min(8, h-i), 0:min(8, w-j)]
print(img_idct2)
cv2.imwrite('apple_gray.jpg', img_gray)
cv2.imwrite('apple_dct.jpg', img_dct)
cv2.imwrite('apple_idct1.jpg', img_idct1)
cv2.imwrite('apple_idct2.jpg', img_idct2)
cv2.imwrite('apple_idct3.jpg', img_idct3)