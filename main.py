import cv2 
import config
from dct import dct_2d
from idct import idct_2d


def main():
	
	img = cv2.imread(config.imageToRead)
	numberCoefficients = 9
	print("Numero de coeficiente	" + str(numberCoefficients) + " !" + "\n")

	print('*************** DCT_img ****************')
	imgResult = dct_2d(img,numberCoefficients)
	cv2.imwrite('dct256.jpg',imgResult)
	
	print('*************** IDCT_img ****************')
	idct_img = idct_2d(imgResult)
	cv2.imwrite('idct256.jpg',idct_img)	

if __name__ == '__main__':
	main()