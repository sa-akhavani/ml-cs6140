# https://www.kaggle.com/mohamedabdullah/disaster-tweets-solution#Real-or-Not?-NLP-with-Disaster-Tweets
import numpy as np
import pandas as pd
import re

# Read Data
training_data = pd.read_csv('./train.csv')
print(training_data.head())

# Find Missing Data
number_of_rows = training_data['id'].size

keywords = training_data['keyword']
existing_keywords_count = 0
for word in keywords:
    if word and not pd.isnull(word):
        existing_keywords_count += 1
missing_keywords_percentage = ((number_of_rows - existing_keywords_count) / number_of_rows) * 100
print("Total Keywords: {} - Existing Keywords: {} - Missing Percentage: %{}".format(
    number_of_rows, existing_keywords_count, missing_keywords_percentage))


locations = training_data['location']
existing_locations_count = 0
for location in locations:
    if location and not pd.isnull(location):
        existing_locations_count += 1
missing_locations_percentage = ((number_of_rows - existing_locations_count) / number_of_rows) * 100
print("Total Keywords: {} - Existing Keywords: {} - Missing Percentage: %{}".format(
    number_of_rows, existing_locations_count, missing_locations_percentage))
# print(keywords)



# Handling Missing Data by Removing Columns
# Dataset without extra columns
data = training_data.drop(['id', 'keyword', 'location'], axis=1)
print(data.head())


tweet_set = []
# Data and Text Cleaning
for tweet in data['text']:
    # Remove Mentions and Links

    # Remove numbers, marks, and unwanted words
    tweet = re.sub("[^a-zA-Z]", ' ', tweet)

    # Changing to lowercase
    tweet = tweet.lower()

    # Extracting Tweet Words
    tweet = tweet.split()

    # #Remove stopwords then Stemming it
    # tweet = [pstem.stem(word) for word in tweet if not word in set(stopwords.words('english'))]

    #Generate the cleaned tweet again
    tweet = ' '.join(tweet)
    print('### - {}'.format(tweet))
    tweet_set.append(tweet)