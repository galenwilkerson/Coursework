# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 15:47:09 2016

@author: username

Program to compute the top ten hash tags by frequency

usage: $ python top_ten.py <tweet_file>


Algorithm:

- load tweets
- get overall hash tags
- get unique hash tags
- compute frequency, store in dictionary
- sort
- print in order


"""
import sys, json

# parse twitter json file into json tweet objects
# optional filter by countrycode 
def getTweets(fp_filename, countryCode = None, exclusive = False):
    myTweets = []

    with open(fp_filename) as f:
        content = f.readlines()
    
    # use json to get all tweets    
    for i in range(len(content)):
        try:
            myTweets.append(json.loads(content[i]))
        except:
            continue

    resultTweets = []
    
    # if we are filtering by countryCode
    if (countryCode != None):

        # exclusive: remove all tweets that are not explicitely in countryCode
        if (exclusive):
            for tweet in myTweets:
                try:
                    if countryCode == "US":
                        resultTweets.append(tweet)
                    else:
                        continue                        
                except:
                    continue

        # not exclusive: remove only tweets that are definitely not in countryCode
        else: 
            for tweet in myTweets:
                try:
                    if countryCode != "US":
                        continue
                    else:
                        resultTweets.append(tweet)
                except:
                    resultTweets.append(tweet)
                    
    else: # country code == None
    
        resultTweets = myTweets        

    return resultTweets

# return the tweet text as a list
def getTweetText(tweet):
    try:
        myText = tweet["text"]
    except:
        return([])
        
    tweetTextList = myText.split()
    return tweetTextList
    
# return the tweet hashtags as a list
def getTweetHashTags(tweet):

    resultList = []
    
    try:
        tweetHashTagList = tweet["entities"]["hashtags"]#["text"]
        #print tweetHashTagList
        for elt in tweetHashTagList:
            resultList.append(elt["text"])#.encode('utf-8') )

#        print type(tweetHashTagList[0])
    except:
        return(resultList)
        
    return resultList
    
    
def main():    
    
#- load tweets

    # input files
    #afinnfile = open(sys.argv[1])
    tweet_file_name = sys.argv[1]
 #   afinnfile = open("AFINN-111.txt") #open(sys.argv[1])
 #   tweet_file_name = "output.json" #sys.argv[2]
        
    tweets = getTweets(tweet_file_name)        

#- get overall hash tags

    allHashTags = []
    for tweet in tweets:
        thisHashTagList = getTweetHashTags(tweet)
        allHashTags.extend(thisHashTagList)

#- get unique hash tags
#    print type(allHashTags)
    #print allHashTags
    uniqueHashTags = set(allHashTags)

    #print uniqueHashTags
    
    hashTagCountDict = dict.fromkeys(uniqueHashTags, 0)

#- compute frequency, store in dictionary

    for hashTag in allHashTags:
        hashTagCountDict[hashTag] += 1

#- sort

    import operator
    sortedHashTagCounts = sorted(hashTagCountDict.items(), key=operator.itemgetter(1))
    
#- print in order
    # first reverse
    sortedHashTagCounts.reverse()

    for i in range(10):
        (hashTag, count) = sortedHashTagCounts[i]
        print hashTag.encode('utf-8') + " " + str(count)


if __name__ == '__main__':
    main()