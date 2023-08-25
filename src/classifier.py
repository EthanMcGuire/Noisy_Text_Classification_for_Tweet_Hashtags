import random
import re
import sys, getopt

import nltk

import vowpalwabbit.pyvw as vw

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

#These only need to be downloaded once
nltk.download('stopwords')
nltk.download('punkt')

OUTPUT_DIR = "predictions.txt"
COMMON_DIR = "top_5k_twitter_2015.txt"

#Reads all of the example files in directory and returns a preprocessed array of all the examples
def readTweetData(file):
    tweet_list = []

    with open(file, "r", encoding='UTF-8') as f:
        lines = f.readlines()

        #Read each tweet, strip new lines characters, and replace : and |
        tweets = [l.strip().replace(':','COLON').replace('|','PIPE') for l in lines]

    for tweet in tweets:
        tweet_data = tweet.split('\t')

        tweet_id = tweet_data[0]
        hashtag_id = tweet_data[1]
        tweet = tweet_data[2]

        #Preprocess the tweet
        tweet = preprocess(tweet)

        tweet_list.append((tweet_id, hashtag_id, tweet))

    return tweet_list

def prepareTrainingData(tweets):
    #The training set will consist of a balanced number tweets for each hashtag (excluding 6)
    split_tweets = splitByHashtag(tweets)

    #Replace the training_tweets with a balanced list of tweets
    hashtag_count = []
    tweets = []

    # Hashtag 1 count: 1181
    # Hashtag 2 count: 1172
    # Hashtag 3 count: 109
    # Hashtag 4 count: 1024
    # Hashtag 5 count: 171
    # Hashtag 6 count: 17

    #Using undersampling

    hashtag_count.append(125)
    hashtag_count.append(125)
    hashtag_count.append(109)
    hashtag_count.append(125)
    hashtag_count.append(125)
    hashtag_count.append(17)

    for i in range(len(split_tweets)):
        #Get the number of tweets for this hashtag
        for j in range(hashtag_count[i]):
            #Grab a tweet by random
            tweets.append(split_tweets[i].pop(random.randrange(len(split_tweets[i]))))

    #Organize training tweets in order of complexity
    complexity = []

    for tweet in tweets:
        complexity.append(getTweetComplexity(tweet[2]))

    tweets = [x for _, x in sorted(zip(complexity, tweets))]

    return tweets

#Preprocesses a tweet for the model
def preprocess(tweet):
    #Remove stop words and punctuation
    if (True):
        #Tokenize the tweet
        words = word_tokenize(tweet)

        #Get stop words
        stop_words = set(stopwords.words('english'))

        #Remove stop words
        filtered_words = [word for word in words if not word.lower() in stop_words]

        tweet = " ".join(filtered_words)

    #Remove URLS and punctuation
    if (True):
        #Remove URLs
        tweet = re.sub(r"http\S+", "", tweet)

    #Stem the tweet
    if (True):
        #Create the Porter Stemmer
        stemmer = PorterStemmer()

        #Tokenize the tweet
        words = word_tokenize(tweet)

        #Apply stemming to each word in the tweet
        stemmed_words = [stemmer.stem(word) for word in words]

        tweet = " ".join(stemmed_words)


    return tweet

#Get the hashtag count of each hashtag for a list of tweets
def getHashtagCounts(tweet_list):
    hashtag_counts = [0] * 6

    for tweet in tweet_list:
        hashtag_counts[int(tweet[1]) - 1] += 1

    return hashtag_counts

#Splits a list of tweets into lists of corresponding hashtags
def splitByHashtag(tweet_list):
    split_list = [[] for i in range(6)]

    #Append the tweet to the list of its hashtag index
    for tweet in tweet_list:
        split_list[int(tweet[1]) - 1].append(tweet)

    return split_list

#Returns a list of examples from an array of tweets in the format (id, hashtag, tweet)
def getExamples(tweet_list, useLabel):
    #Get tweets and hashtags
    tweets = [tweet[2] for tweet in tweet_list]
    hashtags = [tweet[1] for tweet in tweet_list]

    #Initialize the TF-IDF vectorizer and fit it to the tweets
    vectorizer = TfidfVectorizer(max_features=4)
    tfidf = vectorizer.fit_transform(tweets)

    examples = []

    #Get vw examples from hashtags and tweet features
    for i in range(len(tweets)):
        #Get tweet complexity
        complexity = getTweetComplexity(tweets[i])

        tfidf_features = ' | '.join([str(feature) for feature in tfidf[i].toarray()[0]])
        
        features = str(complexity) + ' | ' + tfidf_features
        features = tweets[i] + ' | ' + features #Add the tweet

        if (useLabel):
            examples.append(hashtags[i] + ' | ' + features)
        else:
            examples.append(' | ' + features)

    return examples

#Returns the complexity of a tweet based on how many words are in the common_words list
def getTweetComplexity(tweet):
    n_slex = 0
    n = 0

    words = word_tokenize(tweet)

    for word in words:
        n += 1

        if (not word.lower() in common_words):
            n_slex += 1
    
    try:
        return float(n_slex) / float(n)
    except:
        print("Divide by zero? Does your tweet file have empty tweets?")

        return -1

#Read common words
common_words = []

with open(COMMON_DIR, 'r', encoding='UTF-8') as f:
    while line := f.readline():
        common_words.append(line.strip().split('\t')[0])

# The main program
def main(argv):
    training_file = ''
    validation_file = ''
    test_file = ''

    opts, args = getopt.getopt(argv,"d:v:t:")

    # Get data files
    for opt, arg in opts:
      if opt == '-d':
         training_file = arg
      elif opt == '-v':
         validation_file = arg
      elif opt == '-t':
         test_file = arg

    # All data given?
    if training_file == '':
        print("No training file entered.")
        sys.exit()
    
    if validation_file == '':
        print("No validation file entered.")
        sys.exit()

    if test_file == '':
        print("No test file entered.")
        sys.exit()

    # Read data
    training_tweets = readTweetData(training_file)
    validation_tweets = readTweetData(validation_file)
    test_tweets = readTweetData(test_file)

    print(f"total training tweets: {len(training_tweets)}")
    print(f"total validation tweets: {len(validation_tweets)}")
    print(f"total test tweets: {len(test_tweets)}")

    # Prepare training data
    prepareTrainingData(training_tweets)

    #Shuffle and split the data
    random.seed()

    random.shuffle(validation_tweets)
    random.shuffle(test_tweets)

    #Training set
    training_set = getExamples(training_tweets, True)

    # Create the model
    model = vw.Workspace(
        loss_function='hinge',
        oaa=6,
        learning_rate=0.45,
        power_t=0.5,
        initial_t=0,
        quadratic='ss',
        cubic='sbc',
        bit_precision=24,
        interactions='ff',
        passes=1000,
        ngram=2,
        quiet=True,
        random_seed=1,
        cache_file='cache_file'
    )

    # Train the model
    for tweet in training_set:
        model.learn(tweet)

    # Test on the validation set
    predictions = []
    correct = 0.0

    validation_set = getExamples(validation_tweets, False)

    tweet_count = len(validation_set)

    for i in range(len(validation_set)):
        predicted_hashtag = round(model.predict(validation_set[i]), 0)  #Starting at 0 instead of 1 for some reason
        tweet_id = validation_tweets[i][0]

        predictions.append(predicted_hashtag)

    #Validation set accuracy
    for i in range(len(validation_tweets)):
        if (int(validation_tweets[i][1]) == predictions[i]):
            correct += 1.0

    print(f"Accuracy on validation set is {float(correct)/tweet_count * 100.0}%. {correct} tweets correct.")

    #TESTING PHASE
    predictions = []

    test_set = getExamples(test_tweets, False)

    #Test the model
    for tweet in test_set:
        predicted_hashtag = round(model.predict(tweet) + 1)  #Starting at 0 instead of 1 for some reason
        tweet_id = test_tweets[test_set.index(tweet)][0]

        predictions.append((tweet_id, predicted_hashtag))

    with open(OUTPUT_DIR, "w") as f:
        for prediction in predictions:
            f.write(f"{prediction[0]}\t{prediction[1]}\n")

if __name__ == "__main__":
   main(sys.argv[1:])