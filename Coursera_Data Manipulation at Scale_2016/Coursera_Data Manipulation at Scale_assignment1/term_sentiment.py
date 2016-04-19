"""
- create vector of unique words in all tweets, init to zeros
- fill in valence values for known words in sentiment file

- create matrix of co-occurence, init to zeros
- fill in co-occurence counts (leave diagonal zero)
- convert to frequency (normalize columns by column total)

1. multiply unique words by co-occurence vector to get new valence vector

2. repeat 1. until no change in valence vector

3. print out each key, value pair in valence vector  (what to do with unknown words?)

"""

import sys, json
import numpy as np
from scipy.sparse import lil_matrix

# return dictionary of scores
def sentimentScores(afinnfile):
    scores = {} # initialize an empty dictionary
    for line in afinnfile:
        term, score  = line.split("\t")  # The file is tab-delimited. "\t" means "tab character"
        scores[term] = int(score)  # Convert the score to an integer.
    return scores

# parse twitter json file into json tweet objects
def getTweets(fp_filename):
    myTweets = []

    with open(fp_filename) as f:
        content = f.readlines()
        
    for i in range(len(content)):
        try:
            myTweets.append(json.loads(content[i]))
        except:
            continue
        
    return myTweets

# return the tweet text as a list
def getTweetText(tweet):
    try:
        myText = tweet["text"]
    except:
        return([])
        
    tweetTextList = myText.split()
    return tweetTextList
    
# return a set of unique words    
def getGlobalTextSet(tweets):

    globalTextList = [] 

    for tweet in tweets:

        # get the text
        tweetTextList = getTweetText(tweet)
        globalTextList.extend(tweetTextList)

    # get the set of unique words    
    uniqueWords = set(globalTextList)    
    
    return uniqueWords    

# return a list of the unique words and numpy array with zeros except where valence scores are defined,
# same size and order
# fill in valence values for known words in sentiment file
def getUniqueWordsListAndValenceVector(uniqueWordsSet, scores):
         
        # make copy
        uniqueWordsDict = dict(scores)

        # fill in all the unique words        
        for word in uniqueWordsSet:
            # use lower case word
            uniqueWordsDict[word.lower()] = 0
        
#        scoreVector = np.zeros(len(uniqueWordsDict))

        # finally, return a list and numpy array in the same order
        uniqueWordsList = uniqueWordsDict.keys()            
        
        # create a numpy vector        
        scoreVector = np.array(uniqueWordsDict.values())            

        return(uniqueWordsList, scoreVector)

#- create matrix of co-occurence, init to zeros
#- fill in co-occurence counts (leave diagonal zero)
#- convert to frequency (normalize columns by column total)
def createCooccurenceFrequencyMatrix(uniqueWordList, tweets):
    
    # init to zeros   
    # cooccurenceFrequencyMatrix = np.zeros([len(uniqueWordList), len(uniqueWordList)])
    
    cooccurenceFrequencyMatrix = lil_matrix((len(uniqueWordList), len(uniqueWordList)))
    
    # get co-occurence counts  (probability that B occurs in tweet given A occurs)
    for tweet in tweets:
        
        tweetText = getTweetText(tweet)
        
        # look through the whole tweet
        for i in range(len(tweetText)):
            
            # find the word in the unique word list
            # index into the cooccurence matrix
            wordFrom = tweetText[i]
            indexFrom = uniqueWordList.index(wordFrom)
            
            # look through the rest of the tweet
            for j in range(i+1 , len(tweetText)):
                
                # find the word in the unique word list
                # index into the cooccurence matrix
                wordTo = tweetText[j]
                indexTo = uniqueWordList.index(wordTo)

                # increment the co-occurence count (keep it symmetric)
                cooccurenceFrequencyMatrix[i,j] += 1
                cooccurenceFrequencyMatrix[j,i] += 1
    
    
    # now finished, normalize columns        
    sums = np.sum(cooccurenceFrequencyMatrix, axis=0)
        
    # rows
    for i in range(cooccurenceFrequencyMatrix[0]):
        # cols
        for j in range(cooccurenceFrequencyMatrix[1]):

            # normalize (L-1 norm) by column sum            
            if sums[j] != 0:
                cooccurenceFrequencyMatrix[i][j] /= float(sums[j])

            # else, nothing to do, already normalized
    
    return(cooccurenceFrequencyMatrix)            

# check that the two vectors are (nearly) equal
# inputs: two numpy vectors of same length, a tolerance
# returns: True (converged) or False
def checkConvergence(lastScores, newScores, tolerance = 1e-08):
    
     return(np.allclose(lastScores, newScores, tolerance))
    
    
# return converged valence vector
#1. multiply unique words by co-occurence vector to get new valence vector
#2. repeat 1. until no change in valence vector
# inputs:  numpy vector of valence values (M x 1), co-occurence freqency matrix (M x M) 
# outputs: numpy vector of new (converged) valence values (M x 1)
# note:  can't we use eigenvalues of matrix to do this?
def findConvergedValenceScores(valenceVector, cooccurenceFrequencyMatrix): 
    
    # copy the input vector
    newScores = np.array(valenceVector)
    
    converged = False
    
    while(not converged):
        
        lastScores = np.array(newScores)
        
        #1. multiply unique words by co-occurence vector to get new valence vector
        newScores = np.linalg.linalg.product(valenceVector, cooccurenceFrequencyMatrix)    
    
        #2. repeat 1. until no change in valence vector
        converged = checkConvergence(lastScores, newScores, tolerance)
        
    
    
def main():
    #afinnfile = open(sys.argv[1])
    afinnfile = open("AFINN-111.txt")

    #tweet_file_name = sys.argv[2]
    tweet_file_name = "output.json"
    
    tweets = getTweets(tweet_file_name)

    scores = sentimentScores(afinnfile)

    #- create vector of unique words in all tweets, init to zeros
    
    # get unique words in a set 
    uniqueWordsSet = getGlobalTextSet(tweets)

    # create a list of the words and numpy array with zeros except where valence scores are defined
    #- fill in valence values for known words in sentiment file
    (uniqueWordList, valenceVector) = getUniqueWordsListAndValenceVector(uniqueWordsSet, scores)

    #- create matrix of co-occurence, init to zeros
    #- fill in co-occurence counts (leave diagonal zero)
    #- convert to frequency (normalize columns by column total)
    cooccurenceFrequencyMatrix = createCooccurenceFrequencyMatrix(uniqueWordList, tweets)    
  
    newScores = findConvergedValenceScores(valenceVector, cooccurenceFrequencyMatrix) 
      
#    while():
#        #1. multiply unique words by co-occurence vector to get new valence vector
#        newScores = np.linalg.linalg.product(valenceVector, cooccurenceFrequencyMatrix)    
    
        #2. repeat 1. until no change in valence vector
    
    #
    #3. print out each key, value pair in valence vector  (what to do with unknown words?)





if __name__ == '__main__':
    main()
