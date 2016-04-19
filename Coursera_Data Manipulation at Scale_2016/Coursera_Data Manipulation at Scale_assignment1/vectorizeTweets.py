from sklearn.feature_extraction.text import *
tfidf_vectorizer = TfidfVectorizer(min_df=100)
X_train_tfidf = tfidf_vectorizer.fit_transform(data)
