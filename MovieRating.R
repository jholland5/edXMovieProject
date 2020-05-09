# Movie Rating Project

################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
library(ggplot2)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

######################################################################
######################################################################
# Investigate structure of edX and validation
str(edx)                 # 9000055 rows with 6 vars
str(validation)          # 999999 rows with the same 6 variables.
                         # The edx (training data) is about 89% of the data
#########################
# How many of each rating are there?
zees <- edx$rating
y <- table(zees)
barplot(y, main = "Rating Counts in the Data Set",
        col = "hotpink") # Interesting to note "point-5's"            
#########################
######################### Distinct number of movies and users.
edx %>% summarize(numovies = n_distinct(movieId),
                  numusers = n_distinct(userId))

###########################################################3
###########################################################
# Movies with the most ratings:

edx %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%             # This shows Pulp Fiction
  arrange(desc(count)) %>% head()

###################################

# We partition the edx data into training and test sets.

edx_index <- createDataPartition(edx$rating,times = 1,p=.5,list = FALSE)
edx_test <- edx %>% slice(edx_index)
edx_train <- edx %>% slice(-edx_index)

# We remove entries in the test set that do not have userId's or movieId's
# in the training set.

edx_test <- edx_test %>% semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train,by = "userId")

# We start with a baseline of just guessing the same number for all movies.
# We know that the number the minimizes RMSE is the mean.

mu <- mean(edx_train$rating)

# Compute RMSE using mu as the guess for every movie.

naive_RMSE <- sqrt(mean((edx_test$rating - mu)^2))   
naive_RMSE                                           #1.06

# We attempt to do better by incorporating a bias term for each movie
# We call this term b_i which stands for bias for movie i.

movie_avgs <- edx_train %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))  
# This is the average rating minus mu, it is the 
# least squares estimate of b_i.
# A quick plot reveals the distribution of the b_i's
movie_avgs %>% qplot(b_i, geom ="histogram", 
                     bins = 10, data = ., color = I("black"))
str(movie_avgs)

predicted_ratings <- mu + edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

### RMSE for model with movie effects:
model1_RMSE <- sqrt(mean((edx_test$rating - predicted_ratings)^2))
model1_RMSE
# We get an improvement in RMSE down to .944.

# For our final model, we keep the movie effects, incorporate 
# user effects and add a penalty parameter lambda per the methods
# discussed on p.647-649 of [1].

lambdas <- seq(0,7,.25)  # We find the best lambda between 0 an 7.

RMSEs <- sapply(lambdas, function(l){
  mu <- mean(edx_train$rating)
# The b_i's are needed for movie effects.  
  b_i <- edx_train %>% group_by(movieId) %>%
    summarize(b_i = sum(rating-mu)/(n()+l))
# The b_u's are needed for user effects. 
  b_u <- edx_train %>% left_join(b_i, by="movieId") %>%
    group_by(userId) %>% 
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- edx_test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(sqrt(mean((edx_test$rating - predicted_ratings)^2)))
})
qplot(lambdas,RMSEs) # Quick plot of lambdas vs RMSE's


lambda <- lambdas[which.min(RMSEs)]
lambda              # A lambda of 4.75 minimizes RMSE values.
                    # The lowest RMSE is about .868
  
# The final model will use the lambda found in the previous step 
# to compute RMSE on the validation set. 
  

mu <- mean(edx$rating)  # We use the full edx set for mu.
# The b_i's are needed for movie effects.  
b_i <- edx %>% group_by(movieId) %>% # full edx for b_i
  summarize(b_i = sum(rating-mu)/(n()+4.75))
# The b_u's are needed for user effects. Full edx for b_u
b_u <- edx %>% left_join(b_i, by="movieId") %>%
  group_by(userId) %>% 
  summarize(b_u = sum(rating - b_i - mu)/(n()+4.75))

predicted_ratings <- validation %>%   #Compute predictions and add
  left_join(b_i, by = "movieId") %>%  # to validation set.
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

FINAL_RMSE <- (sqrt(mean((validation$rating - predicted_ratings)^2)))
FINAL_RMSE                           # Hopefully this is worth full credit!





