import os
import sys
from operator import itemgetter
from surprise import SVD,SVDpp,KNNWithMeans,Dataset,Reader,accuracy,print_perf,evaluate,dump

def generate_testset(test_set):
	"""
	Adjusting the loaded test set in order to be used in the test function
	----------------------------------------------------------------------------
	Preconditions:
		test_set - a set of ratings in the form of user,item,rating,timestamp
	
	Postconditions:
		new_set - a set of ratings in the form of user,item,rating
	"""
	new_set=[]
	for rating in test_set:
		new_set.append((rating[0],rating[1],rating[2]))
	return new_set

def basic(train_file,test_file,recommender,data_reader):
	folds_file=[(train_file,test_file)]

	data = Dataset.load_from_folds(folds_file,reader=data_reader)
	
	print_perf(evaluate(recommender,data,measures=['RMSE']))
	
	return

def test_model(ds_dir,data_reader,recommender):
	"""
	Function that is used to test the recommender system model
	---------------------------------------------------------------------------------------------
	Preconditions:
		ds_dir - directory of the test file
		data_reader - an instance of surprise.Reader, used to read the test file
		recommender - an instance of the chosen recommender model (SVD, SVDpp or KNNWithMeans)
	Postconditions:
		recommender - the same recommender instance in the input, now tested
		rmse - root mean square score for the model
		precision - how many relevant items were recommended out of all recommended items
		recall - how many relevant items were recommended out of the relevant items
	"""
	test_file = ds_dir + "test.txt"
	# Loading data from file
	test_ds = Dataset.load_from_file(test_file,data_reader)
	# Adjusting the structure in order to test the set
	full_test_ds = generate_testset(test_ds.raw_ratings)
	# Test the set using the given recommender method
	predictions = recommender.test(full_test_ds)
	# Calculating the rmse using surprise accuracy.rmse function
	rmse = accuracy.rmse(predictions,True)
	# Calculate the precision and recall for top n recommendations and recommendation threshold of t
	# n = 5, t = 4
	precision, recall = precision_recall(predictions,5,4)
	print("Precision={0:.3f}, Recall={1:.3f}".format(precision, recall))
	return recommender,rmse,precision,recall


def train_model(ds_dir,data_reader,test,recommender,dump_file):
	"""
	Function that loads a trainset and trains the model
	----------------------------------------------------------------------------------------------
	Preconditions:
		ds_dir - directory of the train file
		data_reader - an instance of surprise.Reader, used to read the trainset file
		test - boolean variable, representing if to test the model, or not
		recommender - an instance of the chosen recommender model (SVD, SVDpp or KNNWithMeans)
		dump_file - file name and directory to save the recommender model, after it was trained
	
	Postconditions:
		saves the model into file in dump_file
	"""
        train_file = ds_dir + "train.txt"
	# Load the trainset from file
	train_ds = Dataset.load_from_file(train_file,data_reader)
	# All the data from the trainset will be used to train the model (testset is in a seperate file)
        full_train_ds = train_ds.build_full_trainset()
	# Train the given model
	recommender.train(full_train_ds)
	# If test==True, test the model and print its rmse, precision and recall results to screen
	if test:
		recommender,rmse,precision,recall = test_model(ds_dir,data_reader,recommender)
	# Dump the model for future use
	dump.dump(dump_file,algo=recommender)

def use_model(recommender,user,movie):
	"""
	Function that is called for when a prediction is required for a given user and movie
	-------------------------------------------------------------------------------------------------------------------------------------------------------
	Preconditions:
		recommender - an instance of the chosen recommender model (SVD, SVDpp or KNNWithMeans)
		user - user id as it exists in the dataset
		movie - item id as it exists in the dataset
	Postconditions::
		prediction of the score of a given user and movie, in the form of (user id, movie id, true rating,predicted ratin, was it impossible to predict
	"""
	return recommender.predict(user,movie)

def to_movie_names(movie_set):
	"""
	Translates a given set of movie ids to movie names, using the movie description file
	-------------------------------------------------------------------------------------------------------
	Preconditions:
		movie_set - a set of movie data, where the movie id is found in the first position

	Postconditions:
		output - a set of the same structure as the input, where instead of a movie id there is a name
	"""
	movies = {}
	# Loading the movie descriptions from a file
	movie_file = os.path.dirname(os.path.abspath(__file__))+"/files_processed/movies.txt"
	f = open(movie_file,'r')
	# Adding each movie name from the file to the relevant movie id key in the dictionary
	for m in f:
		movie_data = m.split('\t')
		movies[movie_data[0]] = movie_data[1]
	output = []
	# Finding the name of each movie in the given set and adding it and the rest of the input to the output list
	for item in movie_set:
		output.append((movies[item[0]],item[1:]))
	return output

def user_ratings(trainset,user,with_rating):
	"""
	Retreiving all movies user has rated based on the training set, plus the real rating (optional)
	----------------------------------------------------------------------------------------------
	Preconditions:
		trainset - the trainset in the structure of [movie id, rating]
		user - user id
		with rating - boolean, if True then the real ratings are added to the rated movie list
	Postconditions:
		user_movies - a set of rated movies by the user, plus the ratings if asked
	"""
	user_movies=[]
	# Translating the given user id to the system's internal user id
	user_id = trainset.to_inner_uid(user)
	# Iterating over the user's rated items set in the training set
	for user_movie in trainset.ur[user_id]:
		# Adding the movies and ratings (if with_rating is true) to the result set
		if (with_rating):
			user_movies.append((trainset.to_raw_iid(user_movie[0]),user_movie[1]))
		else:
	                user_movies.append(user_movie[0])
	return user_movies

def get_ratings(recommender,user,names):
	"""
	Method meant to get all past ratings of a user from the model
	---------------------------------------------------------------------------------------------------
	Preconditions:
		recommender - an instance of the chosen recommender model (SVD, SVDpp or KNNWithMeans)
		user - a user id
		names - boolean, if True returns the list with movie names, if False with movie ids instead
	Postconditions:
		user ratings in form of (movie, rating)
	"""
	# Get the user ratings from the model's training set
	ratings = user_ratings(recommender.trainset,user,True)
	if (names):
		# Translate the movie ids to names
		return to_movie_names(ratings)
	else:
		return ratings

def top_recom(recommender,user,n,names):
	"""
	Retreive the top n reccomendations for a given user
	-----------------------------------------------------------------------------------------------
	Preconditions:
		recommender - an instance of the chosen recommender model (SVD, SVDpp or KNNWithMeans)
		user - a given user id
		n - number of items to recommend to the user
		names - boolean, wether to return results with movie names or movie ids
	Postconditions:
		a list of recommendations in the form of [item, predicted rating]
	"""
	user_testset = []
	user_movies = []
	trainset = recommender.trainset
	# Get a list of all movies exists in the trainset
	movies = trainset.all_items()
	# Get the user's past ratings from the training set
	user_movies = user_ratings(trainset,user,False)
	for m in movies:
		# Append all movies from the training set that were not rated by the user to the user_testset list
		if m not in user_movies:
			user_testset.append((user,trainset.to_raw_iid(m),trainset.global_mean))
	# Use the user_testset to test the reccomender model, returns a list of predictions.
	# Predictions are of structure user,movie, real rating, predicted rating, was it impossible to predict
	predictions = recommender.test(user_testset)
	# Sort the predictions in descending order by the predicted rating score
	predictions.sort(key=itemgetter(3),reverse=True)
	result = []
	# Adding the item id and predicted rating of the first n movies in the sorted set to result
	for i in range(n):
		result.append([predictions[i][1],predictions[i][3]])
	# If names is True, translate the movie ids into names
	if (names):
		return to_movie_names(result)
	else:
		return result
	
def precision_recall(predictions,n,t):
	"""
	Returns the precision and recall scores for the list of predictions
	---------------------------------------------------------------------------------------------------------------------
	Postconditions:
		predictions - list of predicted ratings, represented as (user,movie,real rating, predicted rating,imposssible
		n - the number of items recommended to the user
		t - threshold for what is considered a recommendation
	Postconditions:
		precision - relevant recommended items / recommended items
		recall - relevant recommended items / relevant items
	"""
	user_ratings = {}
	# For each predictedrating values, add a tuple of (real rating, predicted_rating) to a user's list in the dictionary
	for rating in predictions:
		user = rating[0]
		rating_tuple = (rating[2],rating[3])
		# Create the key for the user in the dictionary if new, or append ratings to it if not
		if user_ratings.has_key(user):
			user_ratings[user].append(rating_tuple)
		else:
			user_ratings[user] = [rating_tuple]
	# Initalizing the counters for precision and recall
	relevant = 0
	recommended = 0
	both = 0
	# Iterating over the rating tuples in the users dictionary
	for ratings in user_ratings.values():
		# sort the ratings based on predictions in descending order
		ratings.sort(key=itemgetter(1),reverse=True)
		i = 0
		# For each of the user ratings, count 
		for r in ratings:
			# If the real rating is bigger then the threshold t, its a relevant item
			if r[0]	> t:
				relevant+=1.0
			# If the prediction is above the threshold t and item is in the top n, it will be recommended
			if r[1] > t and i<n:
				recommended+=1.0
			# If the prediction and the real rating are above the threshold t and the item is in the top n, it's both relevant and recommended
			if r[0] > t and r[1] >t and i<n:
				both+=1.0
			i+=1
	# Calculate precision as items that are relevant and recommended divided by the recommended items
	precision = both / recommended
	# Calculate recall as items that are relevant and recommended divided by the relevant items
	recall = both / relevant
	return precision, recall
			

def main():
	# Get the required action from command line - test or train to build a model,predict or top to use it or data to retrieve user's data
	action = sys.argv[1]
	# Get the required model name from command line - svd, svd++ or k-nearest neighbor with means
	model = sys.argv[2]
	# Get the directory of the program
	prog_dir = os.path.dirname(os.path.abspath(__file__))
	ds_dir = prog_dir+"/files_processed/"
	dump_file = prog_dir+"/models/"+model+".dmp"
	if action=='train' or action=='test':
		# Configuring a data reader method - a line format is user,item,rating and the seperator is tab
		data_reader = Reader(line_format='user item rating', sep='\t')
		# Initalizing the recommendation model - svd, svd++ or k-nearest neighbor with means
		if model=='svd':
			recommender = SVD()
		elif model=='svd++':
			recommender = SVDpp()
		elif model=='knn_means':
			recommender = KNNWithMeans()
		# Setting up the test attribute best on the action
		if action == 'test':
			test=True
			print("Training and testing model")
		else:
			test=False
			print("Training model with no test")
		# Train the model and save it to file
		train_model(ds_dir,data_reader,test,recommender,dump_file)
	elif action=='predict' or action=='top':
		# Get the user id from command line
		user = sys.argv[3]
		# Load the recommender model from disk
		_, recommender = dump.load(dump_file)
		if (action=='predict'):
			print("Using model to predict")
			# Get movie to predict from command line
			movie = sys.argv[4]
			# Get the user predicted rating for the movie and print it
			rating = use_model(recommender,user,movie)
			print(rating)
		else:
			print("Using top n")
			# Get the required top n movies to predict from command line
			n = int(sys.argv[4])
			# Whether the result set should have movie names or ids
			if sys.argv[5] == "names":
				names = True
			else:
				names = False
			# Retreive and print the top n recommendations for the user, with or without names
			recommendations = top_recom(recommender,user,n,names)
			print(recommendations)
	elif action=='data':
		# Get user id from command line
		user = sys.argv[3]
		# Load the recommender model from disk
		_, recommender = dump.load(dump_file)
		print("Printing rating data for user")
		# whether the rated movies should be presented with names or ids
		if sys.argv[4] == "names":
			names = True
		else:
			names = False
		# Print user's past ratings
		print(get_ratings(recommender,user,names))	

main()
