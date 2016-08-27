import cv2
import numpy as np

class BlackLaneDetector:
	def _init_(self, camera):
		self.angle = 0;

	def detect(self, frame, show):
		frame = cv2.resize(frame, (320, 180))
		blur = cv2.blur(frame, (3 , 3))
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 65, 65)
		zhang = self.ZhangAlgorithm(adapt, 50)
		"""
		if show:
			cv2.imshow("original", frame)
			cv2.imshow("adapt", adapt)
			cv2.imshow("Zhang", zhang)
		"""

	def ZhangAlgorithm(self, src, iteration) :
		dst=src.copy()
		height=np.size(dst,0)-1
		width=np.size(dst,1)-1


		isFinished=False
		print "height:",height
		print "width:",width

		for n in range(iteration) :
			print "iter : ",n
			tmpImg=dst
			isFinished=False
			for i in range(1,height) :
				pU=i-1
				pC=i
				pD=i+1
				for j in range(1,width) :
					if tmpImg[pC,j] > 0 :
						ap=0
						p2=tmpImg[pU,j]/255
						p3=tmpImg[pU,j+1]/255
						p4=tmpImg[pC,j+1] /255
						p5=tmpImg[pD,j+1]/255
						p6=tmpImg[pD,j]/255
						p7=tmpImg[pD,j-1]/255
						p8=tmpImg[pC,j-1]/255
						p9=tmpImg[pU,j-1]/255
						if p2==0 and p3==1 :
							ap=ap+1
						if p3==0 and p4==1 :
							ap=ap+1
						if p4==0 and p5==1 :
							ap=ap+1
						if p5==0 and p6==1 :
							ap=ap+1
						if p6==0 and p7==1 :
							ap=ap+1
						if p7==0 and p8==1 :
							ap=ap+1
						if p8==0 and p9==1 :
							ap=ap+1
						if p9==0 and p2==1 :
							ap=ap+1
						if p2+p3+p4+p5+p6+p7+p8+p9 >1 and p2+p3+p4+p5+p6+p7+p8+p9 < 7 :
							if ap==1 :
								if p2*p4*p6 ==0 and p4*p6*p8 ==0 :
									dst[i,j]=0
									isFinished=True
			tmpImg=dst
			for i in range(1,height) :
				pU=i-1
				pC=i
				pD=i+1
				for j in range(1,width) :
					if tmpImg[pC,j] > 0 :
						ap=0
						p2=tmpImg[pU,j]/255
						p3=tmpImg[pU,j+1]/255
						p4=tmpImg[pC,j+1]/255
						p5=tmpImg[pD,j+1]/255
						p6=tmpImg[pD,j]/255
						p7=tmpImg[pD,j-1]/255
						p8=tmpImg[pC,j-1]/255
						p9=tmpImg[pU,j-1]/255
						if p2==0 and p3==1 :
							ap=ap+1
						if p3==0 and p4==1 :
							ap=ap+1
						if p4==0 and p5==1 :
							ap=ap+1
						if p5==0 and p6==1 :
							ap=ap+1
						if p6==0 and p7==1 :
							ap=ap+1
						if p7==0 and p8==1 :
							ap=ap+1
						if p8==0 and p9==1 :
							ap=ap+1
						if p9==0 and p2==1 :
							ap=ap+1
						if p2+p3+p4+p5+p6+p7+p8+p9 >1 and p2+p3+p4+p5+p6+p7+p8+p9 < 7 :
							if ap==1 :
								if p2*p4*p8 ==0 and p2*p6*p8 ==0 :
									dst[i,j]=0
									isFinished=True
			if isFinished==False :
				break
		return dst
