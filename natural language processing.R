# Natural Language Processing/ Sparce Matrix

# Import Dataset
data = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)

# Cleaning the text
install.packages('tm')
library(tm)
corpus = VCorpus(VectorSource(data$Review))

# Change uppercase to lower
corpus = tm_map(corpus, content_transformer(tolower))

# Remove numbers
corpus = tm_map(corpus, removeNumbers)

# Remove punctuations
corpus = tm_map(corpus, removePunctuation)

# Remove Stop words
install.packages('SnowballC')
library(SnowballC)
corpus = tm_map(corpus, removeWords, stopwords())

# Remove White Space
corpus = tm_map(corpus, stripWhitespace)

# Stem Words
corpus = tm_map(corpus, stemDocument)

# Bag of words model
dtm = DocumentTermMatrix(corpus)

# Demensionality reduction (Sparsity reduction)
dtm = removeSparseTerms(dtm, 0.999)

#Transform this into a dataframe so I can fead it to a class model
dataset = as.data.frame(as.matrix(dtm))

#add col liked
dataset$liked = data$Liked

#Split data to Test and Training
library(caTools)
set.seed(42)
split = sample.split(dataset$liked, SplitRatio = 0.75)
training_set = subset(dataset, split = TRUE)
test_set = subset(dataset, split = FALSE)

# Logistic**
#Fitting logistic onto the training and test sets
classifier = glm(formula = liked ~ ., family = binomial, data = training_set)

#predicting the probability that it has a good rating
prob_predict = predict(classifier, 
                       typle = 'response', 
                       dataset = test_set[-692])

goodRating_predict = ifelse(prob_predict > 0.5, 1, 0)

#Making the confusion matrix
con_matrix = table(dataset[,692], goodRating_predict > 0.5)
#((20+9)/1000*100= 2.9% error

# kNN*

# Fit k-NN to the training and test sets
library(class)
y_predict = knn(train = training_set[, -692], 
                test = test_set[, -692],
                cl = training_set[, 692],
                k = 3)

# Make confusion matrix
con_matrix = table(test_set[,692], y_predict)
#((53+238)/1000*100= 29.1% error @ K=5
#((49+127)/1000*100= 17.6% error @ K=3

#SVM***

# Fitting data to SVM
install.packages('e1071')
library(e1071)
classifier = svm(formula = liked ~., data = training_set,
                 type = 'C-classification', kernel = 'linear')

# Pridicting the test set results
y_predict = predict(classifier, dataset = test_set[, -692])

#confusion Matrix
con_matrix = table(test_set[, 692], y_predict)
#((1+7)/1000*100) = .8% error