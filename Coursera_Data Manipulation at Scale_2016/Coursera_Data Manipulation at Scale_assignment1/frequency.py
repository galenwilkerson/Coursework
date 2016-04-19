# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 18:32:49 2016

@author: username

frequency.py

compute the term frequency for unique terms in tweets

usage:  python frequency.py <tweet.json file>
output:  <term><space><frequency>

where frequency is the number of times that unique term appears divided by the 
total term counts for the entire file

"""

import json, sys, operator

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


def main():
    
    tweetFileName = sys.argv[1]
  #  tweetFileName = "output.json"
    tweets = getTweets(tweetFileName)

    # dictionary to keep count of words
    # wordFreqCount = {}
    
    globalTextList = [] 

    # get the total text length
    totalTextLength = 0
    for tweet in tweets:

        # get the text
        tweetTextList = getTweetText(tweet)
        globalTextList.extend(tweetTextList)

        # get the length        
        totalTextLength += len(tweetTextList)

    # get the set of unique words    
    unique_words = set(globalTextList)

    # build a dictionary using unique words as keys
    wordFreqCount = dict.fromkeys(unique_words, 0)

    # go through text and keep word cound in wordFreqCount dictionary
    for word in globalTextList:
        wordFreqCount[word] += 1

    sorted_freq_count = sorted(wordFreqCount.items(), key=operator.itemgetter(1))

    totalCount = float(sum(wordFreqCount.values()))
    
    for item in sorted_freq_count:
        (term, count) = item
        freq = count/totalCount
        print(term.encode('utf-8') + " " + str(freq))


if __name__ == '__main__':
    main()

