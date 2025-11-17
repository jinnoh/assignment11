import gzip
import csv
import string
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
import random
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer



def setup_nltk():
    """
    Downloads necessary NLTK data files.
    This must be called BEFORE any NLTK tools are initialized.
    """
    print("  Checking NLTK data...")
    try:
        # Try to load stopwords, if it fails, download
        stopwords.words('english')
    except LookupError:
        print("    Downloading NLTK 'stopwords' data...")
        nltk.download('stopwords', quiet=True)
    
    try:
        # Try to use WordNetLemmatizer, if it fails, download
        WordNetLemmatizer().lemmatize("testing")
    except LookupError:
        print("    Downloading NLTK 'wordnet' data...")
        nltk.download('wordnet', quiet=True)
        print("    Downloading NLTK 'omw-1.4' data...")
        nltk.download('omw-1.4', quiet=True)
    
    print("  NLTK setup complete.")


def readCSV(filename, gzipped=False):
    """Reads a CSV file, handling gzipped files if specified."""
    data = []
    if gzipped:
        f = gzip.open(filename, 'rt', encoding='utf-8')
    else:
        f = open(filename, 'r', encoding='utf-8')
    cr = csv.reader(f)
    for row in cr:
        data.append(row)
    f.close()
    return data

def readGzJSON(filename):
    """Reads a gzipped JSON-lines file safely."""
    data = []
    with gzip.open(filename, 'rt', encoding='utf-8') as f:
        for line in f:
            data.append(ast.literal_eval(line)) # safe eval
    return data

def writeCSV(filename, rows, header=None):
    """Writes data to a CSV file."""
    with open(filename, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        for r in rows:
            writer.writerow(r)

def Jaccard(s1, s2):
    """Jaccard similarity function."""
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return (numer/denom) if denom > 0 else 0


def getGlobalAverage(trainRatingsList):
    if not trainRatingsList: return 0
    return float(sum(trainRatingsList)) / len(trainRatingsList)

def alphaUpdate(allRatings, alpha, betaU, betaI):
    tot = 0.0
    for(user, book, rating) in allRatings:
        tot += rating - betaU.get(user, 0.0) - betaI.get(book, 0.0)
    return tot / len(allRatings) if allRatings else alpha

def betaUUpdate(ratingsPerUser, alpha, betaU, betaI, lamb):
    newBetaU = {}
    for (user, items) in ratingsPerUser.items():
        numerator = 0.0
        for (book, rating) in items:
            numerator += rating - alpha - betaI.get(book, 0.0)
        # Check for division by zero if user has no ratings (shouldn't happen in loop)
        newBetaU[user] = numerator / (lamb + len(items)) if (lamb + len(items)) != 0 else 0
    return newBetaU

def betaIUpdate(ratingsPerItem, alpha, betaU, betaI, lamb):
    newBetaI = {}
    for (book, users) in ratingsPerItem.items():
        numerator = 0.0
        for (user, rating) in users:
            numerator += rating - alpha - betaU.get(user, 0.0)
        # Check for division by zero if item has no ratings (shouldn't happen in loop)
        newBetaI[book] = numerator / (lamb + len(users)) if (lamb + len(users)) != 0 else 0
    return newBetaI

def train_rating_model(allRatings, ratingsPerUser, ratingsPerItem, trainRatingsList):
    print("  Training advanced rating model (alpha + bu + bi)...")
    lambdaRegularization = 10.0
    iterations = 10
    alpha = getGlobalAverage(trainRatingsList)
    betaU = defaultdict(float)
    betaI = defaultdict(float)
    for i in range(iterations):
        alpha = alphaUpdate(allRatings, alpha, betaU, betaI)
        betaU = betaUUpdate(ratingsPerUser,  alpha, betaU, betaI, lambdaRegularization)
        betaI = betaIUpdate(ratingsPerItem,  alpha, betaU, betaI, lambdaRegularization)
    print("  Rating model training complete.")
    return alpha, betaU, betaI


def train_read_model(allRatings, ratingsPerUser, ratingsPerItem, betaU, betaI):
    """
    Trains a Logistic Regression model for read prediction.
    """
    print("  Generating positive/negative samples for read model...")
    X_train = []
    y_train = []
    
    allBooks = list(ratingsPerItem.keys())
    # Handle edge case where there are no books
    if not allBooks:
        print("  Warning: No books found for read model training.")
        return LogisticRegression() # Return an untrained model
    
    user_read_books = {u: set(b for b,r in items) for u, items in ratingsPerUser.items()}

    for u, b, r in allRatings:
        # Positive sample
        X_train.append([betaU.get(u, 0.0), 
                        betaI.get(b, 0.0), 
                        len(ratingsPerUser.get(u, [])), 
                        len(ratingsPerItem.get(b, []))])
        y_train.append(1)
        
        # Negative sample
        # Pick a random book b' that user u has not read
        while True:
            b_neg = random.choice(allBooks)
            # Check if user 'u' exists in user_read_books and if b_neg is not read
            if u not in user_read_books or b_neg not in user_read_books[u]:
                break
        
        X_train.append([betaU.get(u, 0.0), 
                        betaI.get(b_neg, 0.0), 
                        len(ratingsPerUser.get(u, [])), 
                        len(ratingsPerItem.get(b_neg, []))])
        y_train.append(0)

    print(f"  Training read prediction model (Logistic Regression) on {len(X_train)} samples...")
    model = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
    model.fit(X_train, y_train)
    print("  Read model training complete.")
    return model

def predict_reads(pairs, model, betaU, betaI, ratingsPerUser, ratingsPerItem):
    """
    Generates read predictions using the trained classifier.
    """
    predictions = []
    for u, b in pairs:
        # Create the feature vector [betaU, betaI, user_count, item_count]
        features = [betaU.get(u, 0.0), 
                    betaI.get(b, 0.0), 
                    len(ratingsPerUser.get(u, [])), 
                    len(ratingsPerItem.get(b, []))]
        
        pred = model.predict([features])[0]
        predictions.append([u, b, pred])
    return predictions

def clean_text(text):
    """Step 1: Clean text (punct, non-alpha, lowercase)"""
    text = re.sub("\'", "", str(text)) # Added str() for safety
    text = re.sub("[^a-zA-Z]"," ",text) 
    text = ' '.join(text.split()) 
    text = text.lower() 
    return text

def remove_stopwords(text, stop_words_set):
    """Step 2: Remove stopwords"""
    return ' '.join([w for w in text.split() if w not in stop_words_set])

def lemmatizing(text, lemma):
    """Step 3: Lemmatize"""
    stemSentence = ""
    for word in text.split():
        stem = lemma.lemmatize(word)
        stemSentence += stem + " "
    return stemSentence.strip()

def stemming(text, stemmer):
    """Step 4: Stem"""
    stemSentence = ""
    for word in text.split():
        stem = stemmer.stem(word)
        stemSentence += stem + " "
    return stemSentence.strip()

def preprocess_text_pipeline(text, stop_words_set, lemma, stemmer):
    """Runs the full NLP pipeline from the Colab notebook."""
    text = clean_text(text)
    text = remove_stopwords(text, stop_words_set)
    text = lemmatizing(text, lemma)
    text = stemming(text, stemmer)
    return text

def train_category_model(catTrain, nlp_tools):
    """
    Trains the advanced SVC model on TF-IDF features.
    nlp_tools is a dict containing: {'stopwords', 'lemma', 'stemmer'}
    """
    print("  Running advanced NLP pre-processing on training data...")
    # This will take some time
    preprocessed_reviews = [
        preprocess_text_pipeline(
            d['review_text'], 
            nlp_tools['stopwords'], 
            nlp_tools['lemma'], 
            nlp_tools['stemmer']
        ) for d in catTrain
    ]
    y_train = [d['genreID'] for d in catTrain]

    print("  Fitting TfidfVectorizer (10,000 features)...")
    # Use the parameters from the Colab notebook
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(preprocessed_reviews)
    
    print("  Training category model (SVC)...")
    # Use the best model from the Colab notebook
    model = SVC(kernel='rbf', gamma=1)
    model.fit(X_train_tfidf, y_train)
    
    print("  Category model training complete.")
    # Return both the trained model and the fitted vectorizer
    return model, tfidf_vectorizer

def predict_categories(pairs, catTest, model, vectorizer, nlp_tools):
    """
    Generates category predictions using the trained SVC and TF-IDF vectorizer.
    """
    print("  Running advanced NLP pre-processing on test data...")
    # Create a lookup for test reviews
    testReviewDict = {d['review_id']: d for d in catTest}
    
    # Pre-process all test reviews at once
    test_review_ids = [rID for u, rID in pairs]
    reviews_to_process = []
    for rID in test_review_ids:
        if rID in testReviewDict:
            reviews_to_process.append(testReviewDict[rID]['review_text'])
        else:
            reviews_to_process.append("") # Empty string for missing reviews
            
    preprocessed_test_reviews = [
        preprocess_text_pipeline(
            text, 
            nlp_tools['stopwords'], 
            nlp_tools['lemma'], 
            nlp_tools['stemmer']
        ) for text in reviews_to_process
    ]
    
    print("  Transforming test data and predicting...")
    X_test_tfidf = vectorizer.transform(preprocessed_test_reviews)
    predictions = model.predict(X_test_tfidf)
    
    # Combine pairs with predictions
    final_predictions = []
    for i in range(len(pairs)):
        u, rID = pairs[i]
        pred = predictions[i]
        final_predictions.append([u, rID, pred])
        
    return final_predictions


if __name__ == "__main__":
    
    # --- 0. NLTK Setup ---
    # Call this FIRST, before anything tries to use NLTK
    print("Setting up NLTK...")
    setup_nltk()
    print("NLTK setup finished.\n")
    
    # --- 1. Load All Data ---
    print("Loading all data files...")
    train_data = readCSV("train_Interactions.csv.gz", gzipped=True)[1:]
    pairsRating = readCSV("pairs_Rating.csv")[1:]
    pairsRead = readCSV("pairs_Read.csv")[1:]
    pairsCategory = readCSV("pairs_Category.csv")[1:]
    catTrain = readGzJSON("train_Category.json.gz")
    catTest = readGzJSON("test_Category.json.gz")
    
    # --- 2. Process Interaction Data ---
    print("Processing interaction data...")
    allRatings = [] 
    ratingsPerUser = defaultdict(list)
    ratingsPerItem = defaultdict(list)
    trainRatingsList = [] 
    
    for u, b, r_str in train_data:
        r = int(r_str)
        allRatings.append((u, b, r))
        ratingsPerUser[u].append((b, r))
        ratingsPerItem[b].append((u, r))
        trainRatingsList.append(r)
    print("Data loading and processing complete.\n")

    # --- 3. TASK 3: RATING PREDICTION ---
    print("Starting Task 3: Rating Prediction")
    alpha, betaU, betaI = train_rating_model(allRatings, ratingsPerUser, ratingsPerItem, trainRatingsList)
    
    rating_predictions = []
    for u, b in pairsRating:
        bu = betaU.get(u, 0.0)
        bi = betaI.get(b, 0.0)
        pred = alpha + bu + bi
        if pred > 5: pred = 5
        if pred < 0: pred = 0
        rating_predictions.append([u, b, pred])
        
    writeCSV("predictions_Rating.csv", rating_predictions, header=["userID", "bookID", "prediction"])
    print("Task 3 Complete: 'predictions_Rating.csv' written.\n")

    # --- 4. TASK 1: READ PREDICTION ---
    print("Starting Task 1: Read Prediction")
    # We use the biases (betaU, betaI) from the rating model as features
    read_model = train_read_model(allRatings, ratingsPerUser, ratingsPerItem, betaU, betaI)
    read_predictions = predict_reads(pairsRead, read_model, betaU, betaI, ratingsPerUser, ratingsPerItem)
    
    writeCSV("predictions_Read.csv", read_predictions, header=["userID", "bookID", "prediction"])
    print("Task 1 Complete: 'predictions_Read.csv' written.\n")

    # --- 5. TASK 2: CATEGORY PREDICTION ---
    print("Starting Task 2: Category Prediction")
    
    # Initialize NLTK tools AFTER setup and pass them to functions
    nlp_tools = {
        'stopwords': set(stopwords.words('english')),
        'lemma': WordNetLemmatizer(),
        'stemmer': PorterStemmer()
    }
    
    # Train the new advanced model
    category_model, tfidf_vectorizer = train_category_model(catTrain, nlp_tools)
    
    # Make predictions
    category_predictions = predict_categories(pairsCategory, catTest, category_model, tfidf_vectorizer, nlp_tools)

    writeCSV("predictions_Category.csv", category_predictions, header=["userID", "reviewID", "prediction"])
    print("Task 2 Complete: 'predictions_Category.csv' written.\n")

    print("All tasks complete. Your prediction files are ready for submission.")