import sys
import json

'''
read twitter .json file and parse into list of json objects

read sentiment analysis file into dictionary

lookup tweet text into dictionary and print out, one sentiment per line
'''

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


def main():
    afinnfile = open(sys.argv[1])
#    afinnfile = open("AFINN-111.txt")

    tweet_file_name = sys.argv[2]
#    tweet_file_name = "output.json"
    
    tweets = getTweets(tweet_file_name)

    scores = sentimentScores(afinnfile)

    # for each tweet, get the text, split by space, and get the sentiment
    # sum up the sentiments for each tweet.  unfound sentiments = 0
    for tweet in tweets:

        totalScore = 0
        
        # try to get the text, if not, skip to next
        tweetTextList = getTweetText(tweet)
        if len(tweetTextList) == 0:
            print totalScore
            continue

        for word in tweetTextList:
            try:
                myScore = scores[word]
                totalScore += myScore
            except:
                continue
            
        print totalScore

if __name__ == '__main__':
    main()


