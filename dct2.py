import cv2

import numpy as np

from matplotlib import pyplot as plt

from matplotlib.colors import Normalize

import matplotlib.cm as cm




# cut an image up into blocks of 8x8 pixels blocksize

# print height and width

# the image is cropped, such that its height and width is a multiple of blocksize


B=8

fn3= 'apple.png'

img1 = cv2.imread(fn3, cv2.IMREAD_GRAYSCALE)
print(img1.shape)

h,w=np.array(img1.shape[:2])/B * B
print(h,w)
h=int(h)
w=int(w)


img1=img1[:h,:w]

blocksV=int(h/B)
print(blocksV)

blocksH=int(w/B)
print(blocksH)
vis0 = np.zeros((h,w), np.float32)


trans = np.zeros((h,w), np.float32)

vis0[:h, :w] = img1

for row in range(blocksV):

    for col in range(blocksH):

        currentblock = cv2.dct(vis0[int(row*B):int((row+1)*B),int(col*B):int((col+1)*B)])

        trans[row*B:(row+1)*B,col*B:(col+1)*B]=currentblock
# convert DCT
cv2.imwrite('Transform.jpg',trans)

plt.imshow(trans,cmap="gray")
plt.show()
# point=plt.ginput(1)

# hien 1 block 8*8
point=[(10,10),(18,18)]
block=np.floor(np.array(point)/B) 


col=int(block[0,0])


row=int(block[0,1])


plt.plot([B*col,B*col+B,B*col+B,B*col,B*col],[B*row,B*row,B*row+B,B*row+B,B*row])

plt.axis([0,w,h,0])

plt.title("Original Image")



plt.figure()

plt.subplot(1,2,1)

selectedImg=img1[row*B:(row+1)*B,col*B:(col+1)*B]

N255=Normalize(0,255) 

plt.imshow(selectedImg,cmap="gray",norm=N255,interpolation='nearest')

plt.title("Image in selected Region")


plt.subplot(1,2,2)

selectedTrans=trans[row*B:(row+1)*B,col*B:(col+1)*B]

plt.imshow(selectedTrans,cmap=cm.jet,interpolation='nearest')

plt.colorbar(shrink=1)

plt.title("DCT transform of selected Region")

# nen anh 75%
# nếu mỗi khối 8x8 sau khi biến đổi DCT ta chỉ giữ lại block 4x4 ứng với góc phần tư thứ nhất (của block 8x8) để khôi phục lại ảnh (tương đương quá trình nén 65%) ==> Hãy lập trình để thu được kết quả, nhận xét chất lượng ảnh sau khi khôi phục  

keep_info = 4
for row in range(blocksV):
    
    for col in range(blocksH):
        for k in range(0, 8):
                for l in range(0, 8):
                    if k >= keep_info or l >= keep_info:
                        trans[k+8*row, l+8*col] = 0
back0 = np.zeros((h,w), np.float32)
for row in range(blocksV):

    for col in range(blocksH):        
        currentblock = cv2.idct(trans[row*B:(row+1)*B,col*B:(col+1)*B])
        
        back0[row*B:(row+1)*B,col*B:(col+1)*B]=currentblock
print('kich thuoc: ',trans)
print('trans',trans.shape)
cv2.imwrite('BackTransformed.jpg', back0)



#tinh sai khac

print(back0)
diff=back0-img1

print(diff.max()) 

print (diff.min())

MAD=np.sum(np.abs(diff))/float(h*w)

print ("Mean Absolute Difference: ",MAD)

plt.figure()

plt.imshow(back0,cmap="gray")

plt.title("Backtransformed Image")

plt.show()

