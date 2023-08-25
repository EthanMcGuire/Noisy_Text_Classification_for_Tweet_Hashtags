# Description
A noisy text classifier that classifies the hashtags of tweets. This program is written in Python and uses the Vowpalwabbit machine learning library. 

My main method of classifying hashtags from tweets was to preprocess the tweets, use some specific features, and tweek the hyperparameters. 
I also undersampled the training data due to a lack of tweets for hashtags 3, 5, and 6. Around 100 tweets were used for training on each hashtag (except hashtag 6 which had 17 tweets). 
All this combined led to a pretty solid model with around a 45% success rate on the validation set. Once I learned to add OAA (One against all) with 6 classifiers, my success rate jumped to about 85%.

For preprocessing I removed stop words, punctuation, URLs, and stemmed the words in each tweet.

For features I obtained the complexity of each tweet along with using the TF-IDF vectorizer. Complexity was obtained using the method from assignment 4.

For hyperparameters I also used an n-grams of 2.

How to run:
    $ python3 classifier.py -d train.txt -v val.txt -t test.txt

    -d for training data
    -v for validation data
    -t for test data