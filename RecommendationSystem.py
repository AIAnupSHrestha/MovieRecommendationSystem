import string
import csv
import math
from collections import defaultdict

def loadCsvToDict(filename, encoding="utf-8"):
    data = defaultdict(str)
    
    with open(filename, 'r', encoding=encoding) as csvfile:
        df = csv.reader(csvfile)
        next(df, None)
        for row in df:
            title, description = row[0].lower().strip(), row[1].lower()
            description = preprocessing(description)
            data[title] = description
    return data

def preprocessing(data):
    stopWords = loadStopwords("StopWords.txt")
    data = removeStopwords(data, stopWords)
    data = removePunctuation(data)
    data = stemming(data)
    return data

def removeStopwords(text, stopWords):
    filteredText = " ".join([word for word in text.split() if word not in stopWords])
    return filteredText

def loadStopwords(filename):
    stopWords = set()
    with open(filename, 'r') as stopfile:
        for line in stopfile:
            word = line.strip().lower()
            if word:
                stopWords.add(word)
    return stopWords

def removePunctuation(text):
    punctuation = set(string.punctuation)
    filteredText = "".join([char for char in text if char not in punctuation])
    return filteredText

def stemming(text):
    suffixes = {
        "ing" : "",
        "ed" : "",
        "es" : "",
        "s" : ""
        }
    words = text.split()
    stemmed_words = []
    for word in words:
        for suffix in suffixes:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if stem not in suffixes:
                    stemmed_words.append(stem)
                    break
        else:
            stemmed_words.append(word)
    return " ".join(stemmed_words)

def calculateTf(data):
    tfData = {}
    for title, description in data.items():
        wordCount = {}

        for word in description.split():
            wordCount[word] = wordCount.get(word, 0) + 1
        total_words = sum(wordCount.values())
        tfData[title] = {word: count / total_words for word, count in wordCount.items()}
    return tfData

def calculateIdf(tfValue, data):
    idfData = {}
    uniqueWords = set()
    for movieTitle, wordCount in tfValue.items():
        uniqueWords.update(wordCount.keys())
    
    total_documents = len(data)
    for word in uniqueWords:
        document_count = sum(word in word_counts for _, word_counts in tfValue.items())
        idfData[word] = 1 + math.log(total_documents / (document_count + 1))
    return idfData

def calcTfidfValue(data):
    tfValue = calculateTf(data)
    idfValue = calculateIdf(tfValue, data)
    
    tfidfValue = {}
    for title, wordcount in tfValue.items():
        tfidfValue[title] = {word: tf * idfValue[word] for word, tf in wordcount.items()}
    return tfidfValue

def cosineSimilarity(vec1, vec2):
  dotProduct = sum(v1 * v2 for v1, v2 in zip(vec1.values(), vec2.values()))
  mag1 = math.sqrt(sum(v**2 for v in vec1.values()))
  mag2 = math.sqrt(sum(v**2 for v in vec2.values()))
  if not mag1 or not mag2:
    return 0
  return dotProduct / (mag1 * mag2)

def findSimilarMovies(movieTitle, testTfidf, trainTfidf):
    movieTfidf = testTfidf[movieTitle]
    similarities = {}
    for title, _ in trainTfidf.items():
        if title != movieTitle:
            similarities[title] = cosineSimilarity(movieTfidf, trainTfidf[title])
      
    sorted_similarities = dict(sorted(similarities.items(), key=lambda item: item[1], reverse=True))
  
    return list(sorted_similarities.keys())[:3]

def main():
    trainData = loadCsvToDict("train.csv")
    trainTfidf = calcTfidfValue(trainData)
    
    # testData = loadCsvToDict("test.csv")
    # testTfidf = calcTfidfValue(testData)
  
    # for movie, _ in testData.items():
    #     recommendations = findSimilarMovies(movie, testTfidf, trainTfidf)
    #     print(f"Recommendations for {movie}: {recommendations}")


    movieTitle = input("Enter movie title").lower()
    moviedescription = input("Enter movie description").lower()
    
    moviedescription = preprocessing(moviedescription)
    testMovie = { movieTitle: moviedescription}
    movieTfidf = calcTfidfValue(testMovie)
    for movie, _ in testMovie.items():
        recommendations = findSimilarMovies(movie, movieTfidf, trainTfidf)
        print(f"Recommendations for {movie}: {recommendations}")
  


if __name__ == "__main__":
  main()
