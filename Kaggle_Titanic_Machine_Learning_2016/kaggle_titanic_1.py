'''
Basic Titanic Competition submission for kaggle
https://www.kaggle.com/c/titanic

Much of this comes originally from:
https://www.dataquest.io/mission/74/getting-started-with-kaggle

The point of this is to go through all of the steps from reading data to submitting to kaggle

Functions are written as generally as possible, so we don't care what the data set is,
as a step toward automation.


including:


read data

data summary

clean data

change non-numeric values to numeric

train (various models) on data using k-fold training sets

evaluate accuracy scores of results

submit
'''

import pandas as pd
import sys

# Import the linear regression classifier
from sklearn.linear_model import LinearRegression

# logistic regression classifier
from sklearn.linear_model import LogisticRegression

# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold

# helper to get scores from cross validation
from sklearn import cross_validation

import numpy as np

# random forests classifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

# import regular expressions
import re

# (python standard library)
import operator

# select K best features
from sklearn.feature_selection import SelectKBest, f_classif

# for gradient boosting
from sklearn.ensemble import GradientBoostingClassifier


# read in data using pandas
def readData(filename):
    
    # read into dataframe
    data = pd.read_csv(filename)
    
    return data


# summarize data using pandas
def summarizeData(data):
    # Print the first 5 rows of the dataframe.
    print(data.head(5))

    print(data.describe())
    

# fill in NA values
def cleanData(data):
    
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Embarked"] = data["Embarked"].fillna("S")
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())
    
    return data
    
    
# make non-numeric columns numerical
def makeNumeric(data):
    # Find all the unique genders -- the column appears to contain only male and female.
    #print(data["Sex"].unique())

    # Replace all the occurences of male with the number 0.
    data.loc[data["Sex"] == "male", "Sex"] = 0
    data.loc[data["Sex"] == "female", "Sex"] = 1
    
    # Embarked column
    #print(data["Embarked"].unique())

    data.loc[data["Embarked"] == "S", "Embarked"] = 0
    data.loc[data["Embarked"] == "C", "Embarked"] = 1
    data.loc[data["Embarked"] == "Q", "Embarked"] = 2

    return data


# perform kFoldLinearRegression on data
# inputs: data, the ML algorithm, the predictors, K (num folds)    
def kFoldTraining(data, inputAlgorithm, predictors, K):
    
    
    # Initialize our algorithm class
    alg = inputAlgorithm
    
    # Generate cross validation folds for the data dataset.  It return the row indices corresponding to train and test.
    # We set random_state to ensure we get the same splits every time we run this.
    kf = KFold(data.shape[0], n_folds=K, random_state=1)
    
    predictions = []
    for train, test in kf:
        # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
        train_predictors = (data[predictors].iloc[train,:])
        # The target we're using to train the algorithm.
        train_target = data["Survived"].iloc[train]
        # Training the algorithm using the predictors and target.
        alg.fit(train_predictors, train_target)
        # We can now make predictions on the test fold
        test_predictions = alg.predict(data[predictors].iloc[test,:])
        predictions.append(test_predictions)

    return predictions
    
# evaluate the prediction accuracy against the true values     
# enter the predictions trueValues, and threshold, return accuracy
# Note, this does almost the same as sklearn.cross_validation.cross_val_score, without the kFoldTraining
def evaluateAccuracy(predictions, trueValues, threshold):

    # The predictions are in three separate numpy arrays.  Concatenate them into one.  
    # We concatenate them on axis 0, as they only have one axis.
    predictions = np.concatenate(predictions, axis=0)
    
    # Map predictions to outcomes (only possible outcomes are 1 and 0)
    predictions[predictions > threshold] = 1
    predictions[predictions <= threshold] = 0
    
    #print(predictions == titanic_train["Survived"])
    #print(predictions[predictions == titanic_train["Survived"]])
    #print(sum(predictions[predictions == titanic_train["Survived"]]))
#    accuracy = sum(predictions[predictions == titanic_train["Survived"]]) / len(predictions)
    accuracy = sum(predictions[predictions == trueValues]) / len(predictions)
    
    return accuracy


# plot the performance of a learning algorithm by iterating over various parameters
# basically, this will try to create an N+1 dimensional surface, where N is the number of parameters, and the height is accuracy
#
# try to implement this generically for any number of parameters
# the user is responsible for inputting a good, runnable range of parameters
# inputs: the classification algorithm, the data, the true values to cross-validate, 
#         a list of parameters, the parameter ranges in tuples of (min, max), 
#         and the step size to iterate over the parameters
def plotPerformance(alg, data, params, paramRanges, paramStepSize):
    
    #alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
    
    classificationAlgorithm = alg    
    
    # for each parameter, iterate through the paramRanges by paramStepSize
    # the number of iterators = len(params)
    for i in len(params):

    # kfold cross validation
    #kf = KFold(titanic.shape[0], n_folds=3, random_state=1)
    #  sklearn.cross_validation.cross_val_score(estimator, X, y=None, scoring=None, cv=None, n_jobs=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs')
    
        scores = cross_validation.cross_val_score(alg, titanic_train[predictors], titanic_train["Survived"], cv=3)
        #        print(scores.mean())
        meanScore = scores.mean()
   
    return

# A function to get the title from a name.
def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
    
    
# A function to get the family id given a row
# output: dictionary 
    
family_id_mapping = {}

def get_family_id(row):

    # Find the last name by splitting on a comma
    last_name = row["Name"].split(",")[0]
    
    # Create the family id
    # this concatenates the first part ({0}) of the format() (last_name)
    # with the second part ({1}) of the format() (row["FamilySize"])
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    
    # Look up the id in the mapping
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            # Get the maximum id from the mapping and add one to it if we don't have an id
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
        
    return family_id_mapping[family_id]
    
    
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

    
def main():    
    
#    # try to read first argument
#    try:
#        filename = sys.argv[1]
#    except:
#        #filename = "Data/train.csv"
#        filename = "Data/test.csv"
    
    ###############################################    
    # run on training data
    
    filename = "Data/train.csv"    
    
    # read into dataframe
    titanic_train = readData(filename) 

    # print a summary
    #summarizeData(titanic_train)

    # clean the data, fix NA values, etc
    titanic_train = cleanData(titanic_train)
    
    # change non-numeric values to numeric (necessary?) 
    titanic_train = makeNumeric(titanic_train)

    # The columns we'll use to predict the target
    predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    inputAlgorithm = LinearRegression()
    numFolds = 3

    ###############################################
    # get k-fold predictions using linear regression

    print "linear regression"
    
    predictions = kFoldTraining(titanic_train, inputAlgorithm, predictors, numFolds)

    #print predictions
    
    # evaluate accuracy
    trueValues = titanic_train["Survived"]
    threshold = 0.5
    accuracy = evaluateAccuracy(predictions, trueValues, threshold)
    print(accuracy)
    print

    ###############################################
    # now try logistic regression

    print "logistic regression"

    # set algorithm to logistic regression
    inputAlgorithm = LogisticRegression(C = .3, random_state=1)
    
    # Compute the accuracy score for all the cross validation folds.
    scores = cross_validation.cross_val_score(inputAlgorithm, titanic_train[predictors], titanic_train["Survived"], cv=3)
    
    # Take the mean of the scores (because we have one for each fold)
    #    print(scores)
    print(scores.mean())    
    print    
    
    ###############################################    
    # run on test data, make 1st submission
   
    print "running on test data, generating submission"
    
    filename = "Data/test.csv"
    
        # read into dataframe
    titanic_test = readData(filename) 

    # print a summary
    #summarizeData(titanic_test)

    # clean the data, fix NA values, etc
    titanic_test = cleanData(titanic_test)
    
    # change non-numeric values to numeric (necessary?) 
    titanic_test = makeNumeric(titanic_test)

    # Initialize the algorithm class
    alg = LogisticRegression(random_state=1)

    # Train the algorithm using all the training data
    alg.fit(titanic_train[predictors], titanic_train["Survived"])
    
    # Make predictions using the test set.
    predictions = alg.predict(titanic_test[predictors])
    
    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    submission = pd.DataFrame({
            "PassengerId": titanic_test["PassengerId"],
            "Survived": predictions
        })
    
    #print(submission)    
    submission.to_csv("kaggle_titanic_submission_1.csv", index=False)
    
    print
    
    ###############################################    
    # improve submission using random forests
    
    print "random forests 1"
    
    predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    
    # Initialize our algorithm with the default paramters
    # random_state is for seeding the randomizer, for reproducability
    # n_estimators is the number of trees we want to make
    # min_samples_split is the minimum number of rows we need to make a split
    # min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends (the bottom points of the tree)
    alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
    
    # kfold cross validation
    #kf = KFold(titanic.shape[0], n_folds=3, random_state=1)
    #  sklearn.cross_validation.cross_val_score(estimator, X, y=None, scoring=None, cv=None, n_jobs=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs')
    
    scores = cross_validation.cross_val_score(alg, titanic_train[predictors], titanic_train["Survived"], cv=3)
    print(scores.mean())
    print
    
    
    
    ###############################################    
    # try again using random forests, with more trees
    print "random forests 2"

    alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)

    scores = cross_validation.cross_val_score(alg, titanic_train[predictors], titanic_train["Survived"], cv=3)
    print(scores.mean())
    print
    
    
        
    ###############################################    
    # 
    # Generating a familysize column
    print "creating new features:"
    print "FamilySize"
    print
    
    titanic_train["FamilySize"] = titanic_train["SibSp"] + titanic_train["Parch"]
    
    print "NameLength"
    print
    
    # The .apply method generates a new series
    titanic_train["NameLength"] = titanic_train["Name"].apply(lambda x: len(x))


    
    ###############################################    
    #   
    # Get all the titles and print how often each one occurs.
    print "Title"
    print
    
    titles = titanic_train["Name"].apply(get_title)
    #print(pd.value_counts(titles))
    
    # Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
    for k,v in title_mapping.items():
        titles[titles == k] = v
    
    # Verify that we converted everything.
    #print(pd.value_counts(titles))
    
    # Add in the title column.
    titanic_train["Title"] = titles


    ###############################################    
    #   
    # Get the family ids with the apply method
    print "FamilyId"
    print

    # dictionary for results    
    #family_id_mapping = {}

    # apply to each row, passing in the family_id_mapping dictionary as a parameter
    family_ids = titanic_train.apply(get_family_id, axis=1)
    
    # There are a lot of family ids, so we'll compress all of the families under 3 members into one code.
    family_ids[titanic_train["FamilySize"] < 3] = -1
    
    # Print the count of each unique id.
    #print(pd.value_counts(family_ids))
    
    titanic_train["FamilyId"] = family_ids
    

    ###############################################    
    #   
    print "Get K best features"
    print

    predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]
    
    # Perform feature selection and get the k best scores
    K = 5
    scores = getKBestScoresFromFeatures(titanic_train[predictors], titanic_train["Survived"], predictors, K)
    
    # Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
    plotScores(predictors, scores)
    
    # Pick only the four best features.
    predictors = ["Pclass", "Sex", "Fare", "Title"]
    
    alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=8, min_samples_leaf=4)
    
    scores = cross_validation.cross_val_score(alg, titanic_train[predictors], titanic_train["Survived"], cv=3)
    print(scores.mean())
    print    
    
    
    
    ###############################################    
    #   
    print "Boosting and Ensembling"
    print        
    
    # The algorithms we want to ensemble.
    # We're using the more linear predictors for the logistic regression, and everything with the gradient boosting classifier.
    algorithms = [
        [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],
        [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]#,
        #[RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=8, min_samples_leaf=4), ["Pclass", "Sex", "Fare", "Title"]]
    ]
    
    # Initialize the cross validation folds
    kf = KFold(titanic_train.shape[0], n_folds=3, random_state=1)
    
    predictions = []
    for train, test in kf:
        train_target = titanic_train["Survived"].iloc[train]
        full_test_predictions = []
        
        # Make predictions for each algorithm on each fold
        for alg, predictors in algorithms:
            
            # Fit the algorithm on the training data.
            alg.fit(titanic_train[predictors].iloc[train,:], train_target)
            # Select and predict on the test fold.  
            # The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
            test_predictions = alg.predict_proba(titanic_train[predictors].iloc[test,:].astype(float))[:,1]
            full_test_predictions.append(test_predictions)
        # Use a simple ensembling scheme -- just average the predictions to get the final classification.
        test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
        # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
        test_predictions[test_predictions <= .5] = 0
        test_predictions[test_predictions > .5] = 1
        predictions.append(test_predictions)
    
    # Put all the predictions together into one array.
    predictions = np.concatenate(predictions, axis=0)
    
    # Compute accuracy by comparing to the training data.
    accuracy = sum(predictions[predictions == titanic_train["Survived"]]) / len(predictions)
    print(accuracy)
    print

#    titanic_train.to_csv("Data/new_titanic_train.csv")
    

    ###############################################    
    #
    print "Working on test set"
    print

    # First, we'll add titles to the test set.
    titles = titanic_test["Name"].apply(get_title)
    # We're adding the Dona title to the mapping, because it's in the test set, but not the training set
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}
    for k,v in title_mapping.items():
        titles[titles == k] = v
    titanic_test["Title"] = titles
    # Check the counts of each unique title.
    print(pd.value_counts(titanic_test["Title"]))
    
    # Now, we add the family size column.
    titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]

    # Now we can add family ids.
    # We'll use the same ids that we did earlier.
    #print(family_id_mapping)
    
    family_ids = titanic_test.apply(get_family_id, axis=1)
    family_ids[titanic_test["FamilySize"] < 3] = -1
    titanic_test["FamilyId"] = family_ids        
    
    # add Namelength    
    titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))


    ###############################################    
    #
    print "predicting on test set"
    print
    
    predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]
    
    algorithms = [
        [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
        [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
    ]
    
    full_predictions = []
    for alg, predictors in algorithms:
        # Fit the algorithm using the full training data.
        alg.fit(titanic_train[predictors], titanic_train["Survived"])
        # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
        predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:,1]
        full_predictions.append(predictions)
    
    # The gradient boosting classifier generates better predictions, so we weight it higher.
    predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4

    predictions[predictions <= .5] = 0
    predictions[predictions > .5] = 1
    predictions = predictions.astype(int)

    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    submission = pd.DataFrame({
            "PassengerId": titanic_test["PassengerId"],
            "Survived": predictions
        })
    
    #print(submission)    
    submission.to_csv("kaggle_titanic_submission_2.csv", index=False)
    

#    Try using features related to the cabins.
#    See if any family size features might help -- do the number of women in a family make the whole family more likely to survive?
#    Does the national origin of the passenger's name have anything to do with survival?
#
#There's also a lot more we can do on the algorithm side:
#
#    Try the random forest classifier in the ensemble.
#    A support vector machine might work well with this data.
#    We could try neural networks.
#    Boosting with a different base classifier might work better.
#
#And with ensembling methods:
#
#    Could majority voting be a better ensembling method than averaging probabilities?


if __name__ == "__main__":
    main()