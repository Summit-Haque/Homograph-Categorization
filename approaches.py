from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression

trainPath1 = r'F:\ML Lab\AI Lab work'
trainPath = r'F:\ML Lab\AI Lab work(stemmed)'
testPath1 = r'F:\ML Lab\AI lab Test data'
testPath = r'F:\ML Lab\AI lab Test data(stemmed)'

dataset = load_files(trainPath, shuffle= False, decode_error='ignore', random_state=None,load_content=True)
#trainData,testData,trainTarget,testTarget = train_test_split(dataset.data,dataset.target,train_size  = 0.7, test_size=0.3,random_state=42);
vectorizer=TfidfVectorizer( use_idf=True)
#trainData=vectorizer.fit_transform(trainData)
trainData = dataset.data
trainTarget = dataset.target
trainData = vectorizer.fit_transform(trainData)
trainData=trainData.toarray()

#clf= MultinomialNB()

#clf = MLPClassifier(hidden_layer_sizes=(50, ), random_state=1, activation='relu')

clf = LogisticRegression(C=1e6)

#clf = svm.SVC(kernel='linear', C=200)

clf.fit(trainData,trainTarget)

datatest = load_files(testPath, shuffle= False, decode_error='ignore', random_state=None,load_content=True)
testData = datatest.data;
testTarget = datatest.target
testData = vectorizer.transform(testData)
testData = testData.toarray()

#testData= vectorizer.transform(testData)
#testData=testData.toarray()
#pr = clf.predict(x_Test)
#print('Prediction', clf.predict(x_Test))
acuracy= clf.score(testData,testTarget)
print("Acuracy is",acuracy)

#for item in document:
#	with io.open(item,'r',encoding='utf-8') as f:
#		text=f.read()
#	with io.open('test2.txt','w',encoding='utf-8') as f1:
#		 f1.write(text)
        
	     	
