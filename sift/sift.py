import cv2
import numpy as np
 
TrainSetInfo = {
	"car"		:	40,
	"city"		:	20,
	"dog"		:	30,
	"earth"		:	15,
	"fireworks"	:	20,
	"flowers"	:	20,
	"fruits"	:	20,
	"glass"		:	20,
	"gold"		:	15,
	"gun"		:	20,
	"plane"		:	40,
	"sky"		:	30,
	"worldcup"	:	40
}
 
TestSetInfo = {
	"car"		:	119,
	"city"		:	59,
	"dog"		:	49,
	"earth"		:	24,
	"fireworks"	:	54,
	"flowers"	:	63,
	"fruits"	:	78,
	"glass"		:	52,
	"gold"		:	44,
	"gun"		:	44,
	"plane"		:	102,
	"sky"		:	78,
	"worldcup"	:	131
}
 
def calcSiftFeature(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	sift = cv2.SIFT(200) # max number of SIFT points is 200
	kp, des = sift.detectAndCompute(gray, None)
	return des
 
def calcFeatVec(features, centers):
	featVec = np.zeros((1, 50))
	for i in range(0, features.shape[0]):
		fi = features[i]
		diffMat = np.tile(fi, (50, 1)) - centers
		sqSum = (diffMat**2).sum(axis=1)
		dist = sqSum**0.5
		sortedIndices = dist.argsort()
		idx = sortedIndices[0] # index of the nearest center
		featVec[0][idx] += 1	
	return featVec
 
def initFeatureSet():
	for name, count in TrainSetInfo.items():
		dir = "TrainSet/" + name + "/"
		featureSet = np.float32([]).reshape(0,128)
 
		print "Extract features from TrainSet " + name + ":"
		for i in range(1, count + 1):
			filename = dir + name + " (" + str(i) + ").jpg"
			img = cv2.imread(filename)
			des = calcSiftFeature(img)
			featureSet = np.append(featureSet, des, axis=0)
		
		featCnt = featureSet.shape[0]
		print str(featCnt) + " features in " + str(count) + " images\n"
		
		# save featureSet to file
		filename = "Temp/features/" + name + ".npy"
		np.save(filename, featureSet)
 
def learnVocabulary():
	wordCnt = 50
	for name, count in TrainSetInfo.items():
		filename = "Temp/features/" + name + ".npy"
		features = np.load(filename)
		
		print "Learn vocabulary of " + name + "..."
		# use k-means to cluster a bag of features
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
		flags = cv2.KMEANS_RANDOM_CENTERS
		compactness, labels, centers = cv2.kmeans(features, wordCnt, criteria, 20, flags)
		
		# save vocabulary(a tuple of (labels, centers)) to file
		filename = "Temp/vocabulary/" + name + ".npy"
		np.save(filename, (labels, centers))
		print "Done\n"
 
def trainClassifier():
	trainData = np.float32([]).reshape(0, 50)
	response = np.float32([])
	
	dictIdx = 0
	for name, count in TrainSetInfo.items():
		dir = "TrainSet/" + name + "/"
		labels, centers = np.load("Temp/vocabulary/" + name + ".npy")
		
		print "Init training data of " + name + "..."
		for i in range(1, count + 1):
			filename = dir + name + " (" + str(i) + ").jpg"
			img = cv2.imread(filename)
			features = calcSiftFeature(img)
			featVec = calcFeatVec(features, centers)
			trainData = np.append(trainData, featVec, axis=0)
		
		res = np.repeat(np.float32([dictIdx]), count)
		response = np.append(response, res)
		dictIdx += 1
		print "Done\n"
 
	print "Now train svm classifier..."
	trainData = np.float32(trainData)
	response = response.reshape(-1, 1)
	svm = cv2.SVM()
	svm.train_auto(trainData, response, None, None, None) # select best params
	svm.save("svm.clf")
	print "Done\n"
	
def classify():
	svm = cv2.SVM()
	svm.load("svm.clf")
	
	total = 0; correct = 0; dictIdx = 0
	for name, count in TestSetInfo.items():
		crt = 0
		dir = "TestSet/" + name + "/"
		labels, centers = np.load("Temp/vocabulary/" + name + ".npy")
		
		print "Classify on TestSet " + name + ":"
		for i in range(1, count + 1):
			filename = dir + name + " (" + str(i) + ").jpg"
			img = cv2.imread(filename)
			features = calcSiftFeature(img)
			featVec = calcFeatVec(features, centers)
			case = np.float32(featVec)
			if (dictIdx == svm.predict(case)):
				crt += 1
			
		print "Accuracy: " + str(crt) + " / " + str(count) + "\n"
		total += count
		correct += crt
		dictIdx += 1
		
	print "Total accuracy: " + str(correct) + " / " + str(total)
 
if __name__ == "__main__":	
	initFeatureSet()
	learnVocabulary()
	trainClassifier()
	classify()