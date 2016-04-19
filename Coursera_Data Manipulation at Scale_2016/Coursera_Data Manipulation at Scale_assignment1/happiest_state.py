# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 07:47:30 2016

@author: username

usage: $ python happiest_state.py <sentiment_file> <tweet_file>

from the twitter file and sentiment file, determine the happiest state


- Read the tweets into a list of json objects
- Read the sentiment file into a dictionary

- Create a state dictionary with scores (init to 0) 
- Create a state dictionary with number of words (init to 0)


- Go through each tweet and determine the total valence score and state
- Add the valence score to the dictionary
- Add the number of words to the other dictionary
- Find the maximum score
- Print that state


Possible useable fields:


place  DONE
user location  DONE
text (if contains state name or abbreviation in CAPS only)  DONE
coordinates
ip address

For coordinates, use k-nearest neighbors:

First find tweets whose states have been ID'd (using place, user or text)
    - store their coordinates
    
Once all states have had coordinates stored:
    - do k-nearest neighbors
    
This is imperfect, but fun and a good first-pass attempt.
    

********************************
Note, we could treat this as a supervised learning classification.

Input data are words and other available tweet data, output as the ~50 state classes.
********************************


"""

import json, sys
import random as rnd

# return a dictionary with uppercase KEYS (MD, AK, etc.) and lowercase values (maryland, alaska, etc.)
def getStateDict():
    stateDict = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
#        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
        }
    
    
    # change state name values to all lowercase (Alaska -> alaska, etc)
    for key in stateDict:
        val = stateDict[key]
        lowercaseVal = val.lower()
        stateDict[key] = lowercaseVal

    return stateDict


# return dictionary of scores
def sentimentScores(afinnfile):
    scores = {} # initialize an empty dictionary
    for line in afinnfile:
        term, score  = line.split("\t")  # The file is tab-delimited. "\t" means "tab character"
        scores[term] = int(score)  # Convert the score to an integer.
    return scores


# parse twitter json file into json tweet objects
# optional filter by countrycode 
def getTweets(fp_filename, countryCode = None, exclusive = False):
    myTweets = []

    with open(fp_filename) as f:
        content = f.readlines()
        
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
            

    return resultTweets

# return the tweet text as a list
def getTweetText(tweet):
    try:
        myText = tweet["text"]
    except:
        return([])
        
    tweetTextList = myText.split()
    return tweetTextList

# return the state using the tweet's place field. 
# Return None if not found
def getTweetPlaceState(tweet, stateDict):
    try:
        myPlace = tweet["place"]
    except:
        return(None)
    
    if myPlace == None:
        return (None)
        
    # try to use the full_name field (lowercase)
    try:
        fullName = myPlace["full_name"].lower()
        
        # search for one of the states as a substring in the user location
        for stateKey in stateDict.keys():
            
            # look for the stateKey (AK, MD, NY, etc.)
            stateIDStringIndex = fullName.find(stateKey)
    
            # check if found
            if (stateIDStringIndex != -1):
                return(stateKey)
            
            # look for a match of the name (change user location to lowercase)
            stateFullNameLowercase = stateDict[stateKey]
            stateNameStringIndex = fullName.find(stateFullNameLowercase)
    
            # check if found
            if (stateNameStringIndex != -1):
                return(stateKey)        
        
        
    except:
    
        # try to use the name field    
        try:
            name = myPlace["name"].lower()
            
            # search for one of the states as a substring in the user location
            for stateKey in stateDict.keys():
                
                # look for the stateKey (AK, MD, NY, etc.)
                stateIDStringIndex = name.find(stateKey)
        
                # check if found
                if (stateIDStringIndex != -1):
                    return(stateKey)
                
                # look for a match of the name (change user location to lowercase)
                stateFullNameLowercase = stateDict[stateKey]
                stateNameStringIndex = name.find(stateFullNameLowercase)
        
                # check if found
                if (stateNameStringIndex != -1):
                    return(stateKey)        
            
        except:
            return None
    

    

# return the abbreviation of the state in the tweet user's profile
# return None if not found
def getUserState(tweet, stateDict):
    try:
        myUser = tweet["user"]
    except:
        return(None)
         
    try:
        myLocation = myUser["location"].encode('utf-8')
    except:
        return(None)
    
    # search for one of the states as a substring in the user location
    for stateKey in stateDict.keys():
        
        # look for the stateKey (AK, MD, NY, etc.)
        stateIDStringIndex = myLocation.find(stateKey)

        # check if found
        if (stateIDStringIndex != -1):
            return(stateKey)
        
        # look for a match of the name (change user location to lowercase)
        lowercaseLocation = myLocation.lower()
        stateFullNameLowercase = stateDict[stateKey]
        stateNameStringIndex = lowercaseLocation.find(stateFullNameLowercase)

        # check if found
        if (stateNameStringIndex != -1):
            return(stateKey)

    # failing that, look for one of the states as a substring in the text
    for stateKey in stateDict.keys():
 
        ###############################################################
        # now look in the text by converting the list to a lowercase string
        tweetTextList = getTweetText(tweet)
        allTweetText = "".join(tweetTextList).lower()
        
        # look for the stateKey (AK, MD, NY, etc.)
        stateIDStringIndex = allTweetText.find(stateKey)

        # check if found
        if (stateIDStringIndex != -1):
            return(stateKey)
            
        stateNameStringIndex = allTweetText.find(stateFullNameLowercase)

        # check if found
        if (stateNameStringIndex != -1):
            return(stateKey)    

    return(None)


# return the total score of this tweet
def getTweetScore(tweet, scoreDict):

    totalScore = 0
    
    # try to get the text, if not, skip to next
    tweetTextList = getTweetText(tweet)
    if len(tweetTextList) == 0:
        return totalScore
        
    for word in tweetTextList:
        try:
            myScore = scoreDict[word]
            totalScore += myScore
        except:
            continue

    return totalScore


# return the total score of this tweet
def getTweetLength(tweet):

    totalLength = 0
    
    # try to get the text, if not, skip to next
    tweetTextList = getTweetText(tweet)
    totalLength = len(tweetTextList)

    return totalLength


# return the coordinates of the tweet, using the "coordinates" field 
# or the "user" "coordinates" field
def getTweetCoordinates(tweet):
    try:
        myCoords = tweet["coordinates"]["coordinates"]
#        print myCoords
    except:
        try:
            myCoords = tweet["user"]["coordinates"]
#            print myCoords
        except:
            return(None)


# try to get coordinates and IDs for all states
def getCoordsAndStates(tweets, stateDict):
        
    for tweet in tweets:
 
        # get the coordinates for all available tweets (look in coords and user fields)
        myCoords = getTweetCoordinates(tweet)
        
        # add to the tweet in an easy to find place
        tweet["foundCoords"] = myCoords

    
        stateID = getTweetPlaceState(tweet, stateDict)

        # if we couldn't use the place, try the user data and the text content        
        if (stateID == None):
            tweet["stateID"] = None
            stateID = getUserState(tweet, stateDict)
    
        # state identified? if not, set the state to none, skip this tweet
        if (stateID == None):
            tweet["stateID"] = None
            continue
        else: 
            # add to the tweet in an easy to find place
            tweet["stateID"] = stateID     
                            
    return tweets


# for a list of tweets and a score dictionary of terms,
# return the total score for each tweet 
def getTotalScoresAndCounts(tweets, stateDict, scoreDict):

    # dictionaries to store total score and number of words per state
    stateTotalScores = dict.fromkeys(stateDict, 0)
    stateTotalWordCount = dict.fromkeys(stateDict, 0)
    
    for tweet in tweets:
        stateID = tweet["stateID"]
        if stateID == None:
            continue
        
        tweetScore = getTweetScore(tweet, scoreDict)
        tweetLength = getTweetLength(tweet)
        stateTotalScores[stateID] += tweetScore
        stateTotalWordCount[stateID] += tweetLength

    return (stateTotalScores, stateTotalWordCount)


# return the average score per state    
def getMaxStateIDandScore(stateDict, stateTotalScores, stateTotalWordCount):

    # for average values
    stateAverages = dict.fromkeys(stateDict, 0)
    
    maxStateID = ""
    maxStateScore = -10000000  # init low, since scores can be negative
    for stateID in stateDict.keys():
        totalScore = stateTotalScores[stateID]
        totalCount = stateTotalWordCount[stateID]
        if totalCount != 0:
            average = totalScore/float(totalCount)
        else:
            average = 0.0
            
        stateAverages[stateID] = average
        if average > maxStateScore:
            maxStateID = stateID
            maxStateScore = average    
            
    return(maxStateID, maxStateScore, stateAverages)


def findLatLonBounds(tweets):

    # find the min and max lat and lon
    minLat = -10000
    maxLat = 10000
    minLon = -10000
    maxLat = 10000    
    for tweet in tweets:
        [lat, lon] = tweet["foundCoords"]        
        if lat < minLat:
            minLat = lat
        elif(lat > maxLat):
            maxLat = lat
        if lon < minLon:
            minLon = lon
        elif(lon > maxLon):
            maxLon = lon

    return([minLat, maxLat, minLon, maxLon])

# return the first K coordinates of tweets, also return their indices     
def getFirstKLocations(tweets, K):

    i = 0
    found = 0
    coords = []
    indices = []
    # loop until found K coordinates
    while(found < K):
        tweet = tweets[i]
        try:
            location = tweets["foundCoords"]
            if location != None:
                coords.append(location)
                indices.append(i)                
                found += 1
            else:
                i += 1                
        except:
            i += 1
            
    return (coords, indices)

# get the tweets having coordinates    
def getTweetsWithCoords(tweets):

    tweetsWithCoords = []
    
    for tweet in tweets:
        try:
            location = tweets["foundCoords"]
            if location != None:
                tweetsWithCoords.append(tweet)
        except:
            continue

    return tweetsWithCoords


# given the geo-tagged "coordinates" tweets and K centroids
# return K centroids, where a single centroid is (<lat>, <lon>)
# take the 1st K points in the US (or wherever we're looking) and use them as centroids
def kMeansInitCentroids(tweets, countryCode, K):
       
    centroids = []

    # get the tweets exclusively in this country
    exclusiveTweets = getTweets(tweets, countryCode, True)

    # get the first K (assumes input data is randomized and within region of interest)    
    (centroids, indices) = getFirstKLocations(exclusiveTweets, K)    
    
    return centroids


def findClosestCentroids(tweets, centroids):

    return index


def computeMeans(tweets, index, K):
    
    return centroids
    
# use k-means to guess the states for tweets having coordinate information
# inputs:  tweets, the state dictionary, and the number of iterations
def kmeansLabelStates(tweets, stateDict, numIterations):

    # the number of centroids (states)
    K = len(stateDict)
    centroids = kMeansInitCentroids(tweets , K)
    
    # note, can also use a movement threshold
    for iter in range(numIterations):
        
        # Cluster assignment step: Assign each data point to the
        # closest centroid. idx(i) corresponds to cˆ(i), the index
        # of the centroid assigned to example i
        index = findClosestCentroids(tweets, centroids)

        # Move centroid step: Compute means based on centroid assignments
        centroids = computeMeans(X, idx, K);
    
    return tweets

            
def main():    
    # input files
    afinnfile = open(sys.argv[1])
    tweet_file_name = sys.argv[2]
#    afinnfile = open("AFINN-111.txt") #open(sys.argv[1])
#    tweet_file_name = "output.json" #sys.argv[2]
    
    # get the tweets, remove tweets definitely not in US
    countryCode = "US"
    tweets = getTweets(tweet_file_name, countryCode)        
        
    # get the score dictionary
    scoreDict = sentimentScores(afinnfile)
    
    # get the dictionary of states
    stateDict = getStateDict()

    # try to get coordinates and IDs for all tweets, 
    # and add the fields foundCoords and stateID (= None if nothing found)
    tweets = getCoordsAndStates(tweets, stateDict)       

#    # get the tweets having coordinates    
#    tweetsWithCoords = getTweetsWithCoords(tweets)
#
#    #look at the set difference between tweets with foundCoords and tweets with stateID
#    tweetsWithCoords = set()
#    tweetsWithState = set()
#
#    i = 0
#    for tweet in tweets:
#        if tweet["stateID"] != None:
#            tweetsWithState.add(i)
#        if tweet["foundCoords"] != None:
#            tweetsWithCoords.add(i)
#
#    symmetricDiffTweets = tweetsWithCoords ^ tweetsWithState
#
#    print tweetsWithCoords - tweetsWithState
#    print tweetsWithState - tweetsWithCoords

    # use k-means to try to get states for remaining tweets that have locations
    # problem, all those with states also have coords, so nothing remaining...
    #tweets = kmeansLabelStates(tweets, stateDict)    
    
    # get the total scores and counts for each state
    (stateTotalScores, stateTotalWordCount) = getTotalScoresAndCounts(tweets, stateDict, scoreDict)

    # get the state ID for the state having maximum average score
    (maxStateID, maxStateScore, stateAverages) = getMaxStateIDandScore(stateDict, stateTotalScores, stateTotalWordCount)

    #print stateAverages
#    print
    print maxStateID#, " ", maxStateScore

    



if __name__ == '__main__':
    main()