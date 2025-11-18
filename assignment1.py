import gzip
import csv
from collections import defaultdict, Counter
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import ast
import random
import re
import math

from scipy.sparse import hstack, csr_matrix

# NLTK imports
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

##################################################
#  ====== NLTK Setup Function ======             #
##################################################

def setup_nltk():
    """
    Downloads necessary NLTK data files.
    """
    print("  Checking NLTK data...")
    try:
        stopwords.words('english')
    except LookupError:
        print("    Downloading NLTK 'stopwords' data...")
        nltk.download('stopwords', quiet=True)

    try:
        WordNetLemmatizer().lemmatize("testing")
    except LookupError:
        print("    Downloading NLTK 'wordnet' data...")
        nltk.download('wordnet', quiet=True)
        print("    Downloading NLTK 'omw-1.4' data...")
        nltk.download('omw-1.4', quiet=True)

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("    Downloading NLTK 'punkt' tokenizer data...")
        nltk.download('punkt', quiet=True)

    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("    Downloading NLTK 'punkt_tab' tokenizer data...")
        nltk.download('punkt_tab', quiet=True)

    print("  NLTK setup complete.")

##################################################
#  ====== Helpers to read/write files ======     #
##################################################

def readCSV(filename, gzipped=False):
    data = []
    f = gzip.open(filename, 'rt', encoding='utf-8') if gzipped else open(filename, 'r', encoding='utf-8')
    cr = csv.reader(f)
    for row in cr:
        data.append(row)
    f.close()
    return data

def readGzJSON(filename):
    data = []
    with gzip.open(filename, 'rt', encoding='utf-8') as f:
        for line in f:
            data.append(ast.literal_eval(line))
    return data

def writeCSV(filename, rows, header=None):
    with open(filename, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        for r in rows:
            writer.writerow(r)

##################################################
#  ====== TASK 3: RATING PREDICTION (TUNED) ==#
#  (NO CHANGE - This model is optimized)         #
##################################################

def getGlobalAverage(trainRatingsList):
    return float(sum(trainRatingsList)) / len(trainRatingsList) if trainRatingsList else 0.0

def alphaUpdate(allRatings, alpha, betaU, betaI):
    tot = sum(
        rating - betaU.get(user, 0.0) - betaI.get(book, 0.0)
        for (user, book, rating) in allRatings
    )
    return tot / len(allRatings) if allRatings else alpha

def betaUUpdate(ratingsPerUser, alpha, betaU, betaI, lamb):
    newBetaU = {}
    for user, items in ratingsPerUser.items():
        numerator = sum(
            rating - alpha - betaI.get(book, 0.0)
            for (book, rating) in items
        )
        newBetaU[user] = numerator / (lamb + len(items))
    return newBetaU

def betaIUpdate(ratingsPerItem, alpha, betaU, betaI, lamb):
    newBetaI = {}
    for book, users in ratingsPerItem.items():
        numerator = sum(
            rating - alpha - betaU.get(user, 0.0)
            for (user, rating) in users
        )
        newBetaI[book] = numerator / (lamb + len(users))
    return newBetaI

def train_rating_model(allRatings, ratingsPerUser, ratingsPerItem, trainRatingsList):
    """
    Your tuned model: lambda=5.0 and 20 iterations.
    """
    print("  Training rating model (alpha + bu + bi)...")

    lambdaRegularization = 5.0
    iterations = 20

    alpha = getGlobalAverage(trainRatingsList)
    betaU = defaultdict(float)
    betaI = defaultdict(float)

    for i in range(iterations):
        alpha = alphaUpdate(allRatings, alpha, betaU, betaI)
        betaU = betaUUpdate(ratingsPerUser, alpha, betaU, betaI, lambdaRegularization)
        betaI = betaIUpdate(ratingsPerItem, alpha, betaU, betaI, lambdaRegularization)

    print("  Rating model training complete.")
    return alpha, betaU, betaI

##################################################
#  ====== TASK 1: READ PREDICTION (LOCKED) ====#
#  (This is your high-scoring V6 model)          #
##################################################

def compute_user_item_means(ratingsPerUser, ratingsPerItem):
    user_mean = {}
    for u, items in ratingsPerUser.items():
        if items: user_mean[u] = np.mean([r for (_, r) in items])
    item_mean = {}
    for b, users in ratingsPerItem.items():
        if users: item_mean[b] = np.mean([r for (_, r) in users])
    return user_mean, item_mean

def compute_popular_books(ratingsPerItem, percentile=90):
    counts = [len(users) for _, users in ratingsPerItem.items()]
    if not counts:
        return set()
    threshold = np.percentile(counts, percentile)
    popular = {b for b, users in ratingsPerItem.items() if len(users) >= threshold}
    return popular

def build_read_features(u, b, alpha,
                        betaU, betaI,
                        ratingsPerUser, ratingsPerItem,
                        user_mean, item_mean,
                        global_mean,
                        popular_books):
    """
    Construct the 10-dimensional feature vector.
    """
    f1_beta_u = betaU.get(u, 0.0)
    f2_beta_i = betaI.get(b, 0.0)
    u_cnt = len(ratingsPerUser.get(u, []))
    i_cnt = len(ratingsPerItem.get(b, []))
    f3_log_user_cnt = math.log(u_cnt + 1.0)
    f4_log_item_cnt = math.log(i_cnt + 1.0)
    f5_user_mean = user_mean.get(u, global_mean)
    f6_item_mean = item_mean.get(b, global_mean)
    f7_is_popular = 1.0 if b in popular_books else 0.0
    f8_beta_interaction = f1_beta_u * f2_beta_i
    f9_pred_rating = alpha + f1_beta_u + f2_beta_i
    f10_pred_rating_dup = f9_pred_rating

    return [
        f1_beta_u, f2_beta_i, f3_log_user_cnt, f4_log_item_cnt,
        f5_user_mean, f6_item_mean, f7_is_popular, f8_beta_interaction,
        f9_pred_rating, f10_pred_rating_dup
    ]

def train_read_model(allRatings, ratingsPerUser, ratingsPerItem,
                     betaU, betaI, alpha):
    """
    Train Logistic Regression for read prediction using enriched features.
    """
    print("  Preparing enriched features for read model...")
    global_mean = alpha
    user_mean, item_mean = compute_user_item_means(ratingsPerUser, ratingsPerItem)
    popular_books = compute_popular_books(ratingsPerItem, percentile=90)

    X_train = []
    y_train = []

    allBooks = list(ratingsPerItem.keys())
    if not allBooks:
        return LogisticRegression(), StandardScaler(), user_mean, item_mean, popular_books

    user_read_books = {u: set(b for b, r in items)
                       for u, items in ratingsPerUser.items()}

    for u, b, r in allRatings:
        # Positive sample
        features_pos = build_read_features(
            u, b, alpha, betaU, betaI,
            ratingsPerUser, ratingsPerItem,
            user_mean, item_mean, global_mean, popular_books
        )
        X_train.append(features_pos)
        y_train.append(1)

        # Negative sample
        while True:
            b_neg = random.choice(allBooks)
            if u not in user_read_books or b_neg not in user_read_books[u]:
                break
        features_neg = build_read_features(
            u, b_neg, alpha, betaU, betaI,
            ratingsPerUser, ratingsPerItem,
            user_mean, item_mean, global_mean, popular_books
        )
        X_train.append(features_neg)
        y_train.append(0)

    print(f"  Scaling features for {len(X_train)} samples...")
    X_train_np = np.array(X_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_np)

    print("  Training read prediction model (Logistic Regression)...")
    model = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
    model.fit(X_train_scaled, y_train)
    print("  Read model training complete.")

    return model, scaler, user_mean, item_mean, popular_books

def predict_reads(pairs, model, scaler,
                  betaU, betaI, alpha,
                  ratingsPerUser, ratingsPerItem,
                  user_mean, item_mean,
                  popular_books):
    """
    Generate read predictions using enriched features and trained classifier.
    """
    print("  Generating enriched test features for read model...")
    global_mean = alpha
    X_test = []

    for u, b in pairs:
        features = build_read_features(
            u, b, alpha, betaU, betaI,
            ratingsPerUser, ratingsPerItem,
            user_mean, item_mean, global_mean, popular_books
        )
        X_test.append(features)

    X_test_scaled = scaler.transform(np.array(X_test))

    print("  Predicting read status...")
    all_preds = model.predict(X_test_scaled)

    predictions = []
    for i in range(len(pairs)):
        predictions.append([pairs[i][0], pairs[i][1], int(all_preds[i])])

    return predictions

##################################################
#  ====== TASK 2: CATEGORY PREDICTION (V12) ======
#  (Pipeline: clean+stop+lemma, 60k features, LinearSVC)
##################################################

def clean_text(text):
    text = re.sub("\'", "", str(text))
    text = re.sub("[^a-zA-Z]", " ", text)
    text = ' '.join(text.split())
    return text.lower()

def remove_stopwords(text, stop_words_set):
    return ' '.join([w for w in text.split() if w not in stop_words_set])

def lemmatizing(text, lemma):
    stemSentence = ""
    for word in text.split():
        stemSentence += lemma.lemmatize(word) + " "
    return stemSentence.strip()

def preprocess_text_pipeline(text, nlp_tools):
    """
    V12 NLTK-based preprocessing:
      - clean text
      - remove stopwords
      - lemmatize
    (Stemming has been removed as it was too aggressive)
    """
    text = clean_text(text)
    text = remove_stopwords(text, nlp_tools['stopwords'])
    text = lemmatizing(text, nlp_tools['lemma'])
    return text

def extract_numeric_features(review_dict):
    """
    Numeric metadata features:
      - rating
      - log(1 + n_votes)
    """
    raw_rating = review_dict.get('rating', 0.0)
    try:
        rating = float(raw_rating)
    except (TypeError, ValueError):
        rating = 0.0

    raw_votes = review_dict.get('n_votes', 0.0)
    try:
        n_votes = float(raw_votes)
    except (TypeError, ValueError):
        n_votes = 0.0

    if n_votes < 0.0:
        n_votes = 0.0

    return [rating, math.log1p(n_votes)]

def train_category_model(catTrain, nlp_tools):
    """
    Train a single, powerful LinearSVC model on a hybrid feature set.
    """
    print("  Running V12 NLTK preprocessing (no stemming) on training data...")
    preprocessed_reviews = [preprocess_text_pipeline(d['review_text'], nlp_tools)
                            for d in catTrain]
    y_train = [d['genreID'] for d in catTrain]
    numeric_features = np.array([extract_numeric_features(d) for d in catTrain])

    print(f"  Using all {len(catTrain)} samples for category training...")

    # Scale numeric features
    numeric_scaler = StandardScaler()
    numeric_scaled = numeric_scaler.fit_transform(numeric_features)
    numeric_sparse = csr_matrix(numeric_scaled)

    print("  Fitting tuned TfidfVectorizer (60k features, 1-2 ngrams, min_df=3, max_df=0.8)...")
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.8,
        min_df=3,
        max_features=60000, # <-- YOUR RECOMMENDED PARAMETER
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X_train_tfidf = tfidf_vectorizer.fit_transform(preprocessed_reviews)

    # Build combined feature matrix
    print("  Building hybrid feature matrix...")
    X_train_hybrid = hstack([X_train_tfidf, numeric_sparse])

    print("  Training category model (LinearSVC, C=1.0)...")
    model = LinearSVC(C=1.0, max_iter=2000, dual=True) # dual=True is often faster
    model.fit(X_train_hybrid, y_train)

    print("  Category model training complete.")
    return model, tfidf_vectorizer, numeric_scaler

def predict_categories(pairs, catTest, model, tfidf_vectorizer, numeric_scaler, nlp_tools):
    """
    Generates category predictions using the single LinearSVC model.
    """
    print("  Running V12 NLTK preprocessing on test data...")
    testReviewDict = {d['review_id']: d for d in catTest}

    texts = []
    numeric_list = []
    for u, rID in pairs:
        d = testReviewDict.get(rID, {})
        texts.append(preprocess_text_pipeline(d.get('review_text', ""), nlp_tools))
        numeric_list.append(extract_numeric_features(d))

    numeric_array = np.array(numeric_list)
    numeric_scaled = numeric_scaler.transform(numeric_array)
    numeric_sparse = csr_matrix(numeric_scaled)

    print("  Transforming test data with TF-IDF...")
    X_test_tfidf = tfidf_vectorizer.transform(texts)

    X_test_hybrid = hstack([X_test_tfidf, numeric_sparse])

    print("  Predicting categories...")
    predictions = model.predict(X_test_hybrid)

    final_predictions = []
    for i in range(len(pairs)):
        u, rID = pairs[i]
        pred = predictions[i]
        final_predictions.append([u, rID, pred])

    return final_predictions

##################################################
#  ====== MAIN SCRIPT ========================== #
##################################################

if __name__ == "__main__":

    # --- 0. NLTK Setup ---
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

    # --- 2. Process Interaction Data (for all models) ---
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
    alpha, betaU, betaI = train_rating_model(
        allRatings, ratingsPerUser, ratingsPerItem, trainRatingsList
    )

    rating_predictions = []
    for u, b in pairsRating:
        pred = alpha + betaU.get(u, 0.0) + betaI.get(b, 0.0)
        # Clip to [1, 5]
        pred = max(1.0, min(5.0, pred))
        rating_predictions.append([u, b, pred])

    writeCSV(
        "predictions_Rating.csv",
        rating_predictions,
        header=["userID", "bookID", "prediction"]
    )
    print("Task 3 Complete: 'predictions_Rating.csv' written.\n")

    # --- 4. TASK 1: READ PREDICTION ---
    print("Starting Task 1: Read Prediction")
    read_model, read_scaler, user_mean, item_mean, popular_books = train_read_model(
        allRatings,
        ratingsPerUser,
        ratingsPerItem,
        betaU,
        betaI,
        alpha
    )
    read_predictions = predict_reads(
        pairsRead,
        read_model,
        read_scaler,
        betaU,
        betaI,
        alpha,
        ratingsPerUser,
        ratingsPerItem,
        user_mean,
        item_mean,
        popular_books
    )

    writeCSV(
        "predictions_Read.csv",
        read_predictions,
        header=["userID", "bookID", "prediction"]
    )
    print("Task 1 Complete: 'predictions_Read.csv' written.\n")

    # --- 5. TASK 2: CATEGORY PREDICTION ---
    print("Starting Task 2: Category Prediction")

    nlp_tools = {
        'stopwords': set(stopwords.words('english')),
        'lemma': WordNetLemmatizer(),
        'stemmer': PorterStemmer() # Not used in V12, but tool dict is harmless
    }

    # This is the new V12 model
    category_model, tfidf_vectorizer, numeric_scaler = train_category_model(catTrain, nlp_tools)
    category_predictions = predict_categories(
        pairsCategory, catTest, category_model, tfidf_vectorizer, numeric_scaler, nlp_tools
    )

    writeCSV(
        "predictions_Category.csv",
        category_predictions,
        header=["userID", "reviewID", "prediction"]
    )
    print("Task 2 Complete: 'predictions_Category.csv' written.\n")

    print("All tasks complete. Your prediction files are ready for submission.")