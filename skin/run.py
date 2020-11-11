import cv2
import numpy as np
from sklearn.cluster import KMeans
import argparse

def get_dominant_color(image_file):
    # Apply k-Means Clustering.
    image = cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters = 4)
    clt.fit(image)

    def centroid_histogram(clt):
    	# Grab the number of different clusters and create a histogram
    	# based on the number of pixels assigned to each cluster.
    	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    	(hist, _) = np.histogram(clt.labels_, bins = numLabels)

    	# Normalize the histogram, such that it sums to one.
    	hist = hist.astype("float")
    	hist /= hist.sum()

    	# Return the histogram.
    	return hist

    def get_color(hist, centroids):

    	# Obtain the color with maximum percentage of area covered.
    	maxi=0
    	COLOR=[0,0,0]

    	# Loop over the percentage of each cluster and the color of
    	# each cluster.
    	for (percent, color) in zip(hist, centroids):
    		if(percent>maxi):
    			if(skin(color)):
    				COLOR=color
    	return COLOR

    # Obtain the color and convert it to HSV type
    hist = centroid_histogram(clt)
    skin_color = get_color(hist, clt.cluster_centers_)
    skin_temp2 = np.uint8([[skin_color]])
    skin_color = cv2.cvtColor(skin_temp2,cv2.COLOR_RGB2HSV)
    skin_color=skin_color[0][0]

    # Return the color.
    return skin_color

def skinRange(H,S,V):
    e8 = (H<=25) and (H>=0)
    e9 = (S<174) and (S>58)
    e10 = (V<=255) and (V>=50)
    return (e8 and e9 and e10)

def skin(color):
	temp = np.uint8([[color]])
	color = cv2.cvtColor(temp,cv2.COLOR_RGB2HSV)
	color=color[0][0]
	return skinRange(color[0],color[1],color[2])

# This function is meant to give the skin color of the person by detecting face and then
# applying k-Means Clustering.
def get_skin_color(img):

	# Load the face detector.
	face_cascade =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

	# Convert to grayscale image.
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Detect face in the image.
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	# If a face is detected.
	if(len(faces)>0):
		for (x,y,w,h) in faces:
			print(x, y, w, h)

			# Take out the face from the image.
			image=img[y:y+h,x:x+h]
            skin_color = get_dominant_color(image)
            return skin_color
	else return None


def transfer_color(img,want_color1,skin_color,size):
    print(skin_color)
    diff01=want_color1[0]/skin_color[0]
    diff02=(255-want_color1[0])/(255-skin_color[0])
    diff03=(255*(want_color1[0]-skin_color[0]))/(255-skin_color[0])
    diff11=want_color1[1]/skin_color[1]
    diff12=(255-want_color1[1])/(255-skin_color[1])
    diff13=(255*(want_color1[1]-skin_color[1]))/(255-skin_color[1])
    diff21=want_color1[2]/skin_color[2]
    diff22=(255-want_color1[2])/(255-skin_color[2])
    diff23=(255*(want_color1[2]-skin_color[2]))/(255-skin_color[2])
    diff1=[diff01,diff11,diff21]
    diff2=[diff02,diff12,diff22]
    diff3=[diff03,diff13,diff23]
    for i in range(size[0]):
        for j in range(size[1]):
            helper(img,i,j,skin_color,diff1,diff2,diff3)

def helper(img,i,j,skin_color,diff1,diff2,diff3):
    for k in range(3):
        if(img[i,j,k]<skin_color[k]):
            img[i,j,k]*=diff1[k]
        else:
            img[i,j,k]=(diff2[k]*img[i,j,k])+diff3[k]

def get_lower_upper_range(skin_color,Hue,Saturation,Value):
    if(skin_color[0]>Hue):
    	if(skin_color[0]>(180-Hue)):
    		if(skin_color[1]>Saturation+10):
    			lower1=np.array([skin_color[0]-Hue, skin_color[1]-Saturation,Value], dtype = "uint8")
    			upper1=np.array([180, 255,255], dtype = "uint8")
    			lower2=np.array([0, skin_color[1]-Saturation,Value], dtype = "uint8")
    			upper2=np.array([(skin_color[0]+Hue)%180, 255,255], dtype = "uint8")
    			return (True,lower1,upper1,lower2,upper2)
    		else:
    			lower1=np.array([skin_color[0]-Hue, 10,Value], dtype = "uint8")
    			upper1=np.array([180, 255,255], dtype = "uint8")
    			lower2=np.array([0, 10,Value], dtype = "uint8")
    			upper2=np.array([(skin_color[0]+Hue)%180, 255,255], dtype = "uint8")
    			return (True,lower1,upper1,lower2,upper2)
    	else:
    		if(skin_color[1]>Saturation+10):
    			lower=np.array([skin_color[0]-Hue, skin_color[1]-Saturation,Value], dtype = "uint8")
    			upper=np.array([skin_color[0]+Hue, 255,255], dtype = "uint8")
    			return (False,lower,upper)
    		else:
    			lower=np.array([skin_color[0]-Hue, 10,Value], dtype = "uint8")
    			upper=np.array([skin_color[0]+Hue, 255,255], dtype = "uint8")
    			return (False,lower,upper)
    else:
    	if(skin_color[1]>Saturation+10):
    		lower1=np.array([0, skin_color[1]-Saturation,Value], dtype = "uint8")
    		upper1=np.array([skin_color[0]+Hue, 255,255], dtype = "uint8")
    		lower2=np.array([180-Hue+skin_color[0], skin_color[1]-Saturation,Value], dtype = "uint8")
    		upper2=np.array([180, 255,255], dtype = "uint8")
    		return (True,lower1,upper1,lower2,upper2)
    	else:
    		lower1=np.array([0, 10,Value], dtype = "uint8")
    		upper1=np.array([skin_color[0]+Hue, 255,255], dtype = "uint8")
    		lower2=np.array([180-Hue+skin_color[0], 10,Value], dtype = "uint8")
    		upper2=np.array([180, 255,255], dtype = "uint8")
    		return (True,lower1,upper1,lower2,upper2)

def change_skin(image_file,want_color1,output_path):

    # Input the image, convert it to the HSV color space,
    # and determine the HSV pixel intensities that fall into
    # the speicifed upper and lower boundaries
    if(isinstance(image_file,str)):
        img=cv2.imread(image_file,1)
    else:
        img=cv2.imdecode(np.fromstring(image_file.read(), np.uint8),1)
    converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img1=np.float32(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    size=img.shape

    # Define the upper and lower boundaries of the HSV pixel
    # intensities to be considered 'skin'
    skin_color=get_skin_color(img)
    if(skin_color is None):
    	lower=np.array([0, 58,50], dtype = "uint8")
    	upper=np.array([25, 173,255], dtype = "uint8")
    	skinMask=cv2.inRange(converted, lower, upper)
    	tmpImage=cv2.bitwise_and(img,img,mask=skinMask)
    	skin_color=get_dominant_color(tmpImage)
    else:
    	Hue=10
    	Saturation=65
    	Value=50
    	result=get_lower_upper_range(skin_color,Hue,Saturation,Value)
    	if(result[0]):
    		lower1=result[1]
    		upper1=result[2]
    		lower2=result[3]
    		upper2=result[4]
    		skinMask1=cv2.inRange(converted, lower1, upper1)
    		skinMask2=cv2.inRange(converted, lower2, upper2)
    		skinMask=cv2.bitwise_or(skinMask1,skinMask2)
    	else:
    		lower=result[1]
    		upper=result[2]
    		skinMask = cv2.inRange(converted, lower, upper)

    skinMaskInv=cv2.bitwise_not(skinMask)
    skin_color = np.uint8([[skin_color]])
    # print("ssss ", skin_color)
    skin_color = cv2.cvtColor(skin_color,cv2.COLOR_HSV2RGB)
    skin_color=skin_color[0][0]
    skin_color=np.int16(skin_color)
    want_color1=np.int16(want_color1)

    # Change the color maintaining the texture.
    transfer_color(img1,want_color1,skin_color,size)
    img2=np.uint8(img1)
    img2=cv2.cvtColor(img2,cv2.COLOR_RGB2BGR)

    # Get the two images ie. the skin and the background.
    imgLeft=cv2.bitwise_and(img,img,mask=skinMaskInv)
    skinOver = cv2.bitwise_and(img2, img2, mask = skinMask)
    skin = cv2.add(imgLeft,skinOver)

    res=cv2.imencode('.jpg',skin)[1].tostring()
    return res

if __name__ == "__main__":
	parser=argparse.ArgumentParser()
	parser.add_argument("--img", help="Image to be changed")
	parser.add_argument("--race", help="Required skin color race")
	parser.add_argument("--res", help="Path where image is to be saved")
	args=parser.parse_args()

	with open(args.img,'rb') as inputImage:
		if args.race=="Asian":
			b = [180, 103, 71]
		elif args.race=="European":
			b = [220, 209, 194]
		else:
			b = [111, 79, 29]
		result=change_skin(inputImage,b,args.res)
	with open(args.res,'wb') as resultFile:
		resultFile.write(result)
