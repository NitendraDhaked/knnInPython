# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:57:38 2016
@student Number: 16203776
@author: nitendra
"""

from random import randint
from operator import itemgetter

#dictionary for complete set     
vector_dict = {}
#dictionary for trained data
train_vect ={}
#dictionary for test data
test_vect ={}

print("Data is loading in data structures... ")
#matrix data is loded
word_dic={}
with open("news_articles.mtx") as mtxFile:
    i=0
    prev=1
    for lne in mtxFile.readlines(): #read data line by lines
        i=i+1
        if(i>2):                    #skipped first two lines
            n,word,freq = lne.strip().split(' ')
            if(int(n)==int(prev)):
                word_dic[word] =int(freq)
            else:
                m=int(n)-1          #int(n-1) is wriiten value in string and n is incresed two n+1 and word_dic contains data of prev documnet
                vector_dict[m]= word_dic
                randValue = randint(1,1839)     #random number is genrated
                if(randValue <1839*0.67):       #compare and asign 
                    train_vect[m]=word_dic
                else:
                    test_vect[m]=word_dic
                word_dic ={}
                word_dic[word] =int(freq)
                prev=int(n)
    else:#python allow programers to write else of for loop ,included to add las
        m=int(n)                      
        vector_dict[int(n)]= word_dic
        randValue1 = randint(1,1839)  #ran
        if(randValue1 <1839*0.67):
            train_vect[m]=word_dic
        else:
            test_vect[m]=word_dic

print("Matrix loded...")

labelsData =[]
with open("news_articles.labels") as labFile:
    for lne in labFile.readlines():
        n,label = lne.strip().split(',')
        labelsData.append([n,label])
print("Labels loded...")        

#method for calculation of cosine similarity
def get_cosineSimilarity(trainingVect, testVect):
    #Here keys are the words in single training and test document
    crossProduct= set(trainingVect.keys()) & set(testVect.keys())
    
    #numerator is sum of crossproduct of two vectors frequency values
    numeratorVal =sum([trainingVect[key] * testVect[key] for key in crossProduct])
    
    #Sum of trainingvect values and testVect values
    _trainingSum = sum([trainingVect[key]**2 for key in trainingVect.keys()])
    _testSum = sum([testVect[key]**2 for key in testVect.keys()])

    #sqrt of both vectors
    _tringSumSqrt= _trainingSum**(0.5)
    _testSumSqrt= _testSum**(0.5)
    
    #multiplication of two vectors
    _denominatorVal = _tringSumSqrt * _testSumSqrt
    
    if not _denominatorVal:
        return 0.0
    else:
        #cosine value by dividing numerator to denominator
        return float(numeratorVal)/_denominatorVal

def get_kNeighborsValue(trainingVect, testVect, k):
    #knn similarity measure for each document
    _similarityMeasure = []
    
    #weighted distance measure
   # 
    for docKey in (trainingVect.keys()):
        #print(docKey)
        _similarityVal=get_cosineSimilarity(trainingVect[docKey],testVect)
        
        #document and their distance from test dataset is stored in list
        _similarityMeasure.append((docKey,_similarityVal))
        
    _similarityMeasure.sort(key=itemgetter(1),reverse=True)
    
    #_similarityMeasureWeighted.sort(key=itemgetter(1),reverse=True)
    _neighbourDocs=[]
    #_weight_neighbourDocs=[]
    for i in range(k):
        _neighbourDocs.append(_similarityMeasure[i])
        #_weight_neighbourDocs.append(_similarityMeasureWeighted[i])
    
    return _neighbourDocs

    
def get_WieghtedNeighbourValue(trainingVect, testVect, k):
    #print("finding Neighbours wait....")
    _similarityMeasure=[]
    #dockey is the document number
    for docKey in (trainingVect.keys()):
        #print(docKey)
        _similarityVal=get_cosineSimilarity(trainingVect[docKey],testVect)
        
        #document and their distance from test dataset is stored in list
        _similarityMeasure.append((docKey,_similarityVal))
        
    _similarityMeasure.sort(key=itemgetter(1),reverse=True)
    #print("finish",time.time()-starttime)
    _neighbourDocs=[]
    for i in range(k):
        _distance=1-_similarityMeasure[i][-1]
        #normalize the value between range 0  to 1  and weight calculation
        _weight=1/(1+_distance)
        
        _neighbourDocs.append((_similarityMeasure[i][0],_weight))
    return _neighbourDocs  

#get Response with class label
def get_VotedResponse(neighboursWithVal, labelledData):
    #list of labbelled data
    labelledOut=[]
    for docNum, docCosVal  in neighboursWithVal:
        #print(docNum,labelledData[docNum][1])
        labelledOut.append((docNum,docCosVal,labelledData[docNum-1][1]))
    VoteCollection ={}  
    for x in range(len(labelledOut)): #loop for counting votes
        voteFor = labelledOut[x][-1]
        if voteFor in VoteCollection:
            VoteCollection[voteFor] += 1
        else:
            VoteCollection[voteFor] = 1
    
    sortedVotes = sorted(VoteCollection.items(), key=itemgetter(1), reverse=True)
    return sortedVotes[0][0]

#Method for weighted KNN
def get_weightedVotedResponse(neighboursWithWeightedVal, labelledData):
    #list of labbelled data
    labelledOut =[]
    for docNum, docWeight in neighboursWithWeightedVal:
        labelledOut.append((docNum,docWeight,labelledData[docNum-1][1]))
    VoteCollection ={}
    for x in range(len(labelledOut)):
       # print(x)
        voteFor = labelledOut[x][-1]
        if voteFor in VoteCollection:
            VoteCollection[voteFor] += labelledOut[x][-2]

        else:
            VoteCollection[voteFor] = labelledOut[x][-2]
    
    sortedVotes = sorted(VoteCollection.items(), key=itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def get_Accuracy(test_vect,predictons):    # method for accuracy calculation 
    correct =0
    for docNumber in test_vect.keys():
        """
        #since test vect is labelled from labelled data, -1 is taken 
        #because, labeldatacount start from 0-1838 total=1839 starts 
        """
        if(labelsData[docNumber-1][1]==predictons[docNumber]):
            correct+=1
    return (correct/float(len(test_vect)))*100.0
            
def KNN():  #method for knn 
    predictionData={}
    _predict={}
    k = input("please enter value of K between 1 to 10 :\n")
    print("calculating accuracy for unweighted knn wait...")
    for testDocKey in test_vect: #each test set is compareed with all the traing sets
        neighboursWithValue = get_kNeighborsValue(train_vect, test_vect[testDocKey], int(k))
        #predicted response
        _predictedOut=get_VotedResponse(neighboursWithValue , labelsData)
        #print("> Test Document Number: "+ repr(testDocKey) + ", Actual:" + labelsData[testDocKey-1][1] + ", Predicted:" + _predictedOut)
        _predict[testDocKey]=_predictedOut
    predictionData=_predict
    #KNN accuracy
    print("for K:"+ k +" KNN Accuracy : ", get_Accuracy(test_vect,predictionData))      

        
def weighted_KNN(): # this is the method for weighted knn
    predictionData={} 
    predict={}
    k=input("please enter value of K between 1 to 10 :\n")
    print("calculating accuracy for weighted knn wait...")
    for testDocKey in test_vect: #each test set is compareed with all the traing sets
        #return neighbours with weighted value
        neighboursWithWeightedValue = get_WieghtedNeighbourValue(train_vect, test_vect[testDocKey], int(k))
        #predicted output
        predictedOut=get_weightedVotedResponse(neighboursWithWeightedValue,labelsData)
        #print("> Test Document Number: "+ repr(testDocKey) + ", Actual:" + labelsData[testDocKey-1][1] + ", Predicted:" + predictedOut)
        predict[testDocKey]=predictedOut
    predictionData=predict
    print("for K:"+ k +" Weighted KNN Accuracy: ", get_Accuracy(test_vect,predictionData))

def algoSelection(ini):
    ini=input("please enter 1 for KNN and 2 for weighted KNN :\n")
    if(int(ini)== 1):
        KNN()
        algoSelection(ini)
    elif(int(ini)== 2):
        weighted_KNN()
        algoSelection(ini)
    else:
        print("invalid selection, please try again:")
        algoSelection(ini)

ini=0
algoSelection(ini)