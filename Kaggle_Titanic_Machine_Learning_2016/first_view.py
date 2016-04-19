import pandas as pd
import matplotlib.pyplot as plt

import sys

# plot scores vs. predictors
def plotScores(predictors, scores):
    
    plt.bar(range(len(predictors)), scores)
    plt.xticks(range(len(predictors)), predictors, rotation='vertical')
    plt.show()
    

# input data, predictors, true values, and K, the number to select    
# return the k best scores, where the scores are the log10(pvalues)
def getKBestScoresFromFeatures(data, trueValues, predictors, K):
    # Perform feature selection
    selector = SelectKBest(f_classif, K)
    selector.fit(data,trueValues)
    
    # Get the raw p-values for each feature, and transform from p-values into scores
    scores = -np.log10(selector.pvalues_)
    
    return scores
    
    

# We can use the pandas library in python to read in the csv file.
# This creates a pandas dataframe and assigns it to the titanic variable.
titanic = pd.read_csv("Data/train.csv")
print

print titanic.columns
print

# understand column types
print "column types:"
print titanic.dtypes

print

# count NaN values
print "null values:"
print titanic.isnull().sum()
 
#titanic["Sex_cat"] = titanic["Sex"].astype('category')
#plt.hist(titanic.Sex_cat.cat.codes)



# Print the first 5 rows of the dataframe.
#print(titanic.head(5))

#print titanic["Age"].median()

# quick summary
#print(titanic.describe())


#titanic.hist(figsize = (7,7))

# for histogram, set NA values to -1

#titanic.hist(bins = 30, figsize = (7,7))

#titanic.fillna(sys.maxint)

#titanic.hist(bins = 30, figsize = (7,7))


#print(titanic.head(5))
#titanic.to_csv("Data/train_filledNAs.csv")





# try to automatically plot all categories

import collections

numSubplots = len(titanic.columns)

i = 1

for col in titanic.columns:

    titanic[col].value_counts().hist(bins = 30, figsize = (7,7))
#    print col
#    counts = collections.Counter(titanic[col].values)
#
#    plt.plot(counts.values())
#    plt.title(col)
    plt.subplot(3, 4, i)
#
    i+= 1
    
#plt.show()
