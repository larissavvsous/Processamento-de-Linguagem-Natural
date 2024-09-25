from src.utils import load_pfizer_tweet_dataset

tweets = load_pfizer_tweet_dataset()

import matplotlib.pyplot as plt

## WORD COUNT

# a) Function that returns number of words in a string
def count_words(string):
    # Split the string into words
    words = string.split()

    # Return the number of words
    return len(words)


# Create a new feature word_count
tweets['word_count'] = tweets['text'].apply(count_words)

# Print the average word count of the talks
print("Word count mean: ", tweets['word_count'].mean())

## CHAR COUNT

# Create a feature char_count from "text" feature. Use len function to count
tweets['char_count'] = tweets['text'].apply(len)

# Print the average character count
print("Char count mean: ", tweets['char_count'].mean())


## HASHTAGS COUNT

# Function that returns numner of hashtags in a string
def count_hashtags(string):
    # Split the string into words
    words = string.split()

    # Create a list of words that are hashtags
    hashtags = [word for word in words if word.startswith('#')]

    # Return number of hashtags
    return (len(hashtags))


# Create a feature hashtag_count and display distribution
tweets['hashtag_count'] = tweets['text'].apply(count_hashtags)
tweets['hashtag_count'].hist()
plt.title('Distribuição da contagem de hashtags')
plt.show()


# Function that returns number of mentions in a string
def count_mentions(string):
    # Split the string into words
    words = string.split()

    # Create a list of words that are mentions
    mentions = [word for word in words if word.startswith('@')]

    # Return number of mentions
    return (len(mentions))


# Create a feature mention_count and display distribution
tweets['mention_count'] = tweets['text'].apply(count_mentions)
tweets['mention_count'].hist()
plt.title('Distribuição de contagem de menções')
plt.show()
