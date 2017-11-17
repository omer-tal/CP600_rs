import os
import sys
from operator import itemgetter
from surprise import SVD,SVDpp,KNNWithMeans,Dataset,Reader,accuracy,print_perf,evaluate,dump

def generate_testset(test_set):
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
	test_file = ds_dir + "test.txt"
	test_ds = Dataset.load_from_file(test_file,data_reader)
	full_test_ds = generate_testset(test_ds.raw_ratings)
	predictions = recommender.test(full_test_ds)
	rmse = accuracy.rmse(predictions,True)

	return recommender,rmse


def train_model(ds_dir,data_reader,test,recommender,dump_file):
        train_file = ds_dir + "train.txt"
	train_ds = Dataset.load_from_file(train_file,data_reader)
        full_train_ds = train_ds.build_full_trainset()
	recommender.train(full_train_ds)
	if test:
		recommender,rmse = test_model(ds_dir,data_reader,recommender)

	dump.dump(dump_file,algo=recommender)

def use_model(recommender,user,movie):
	return recommender.predict(user,movie)

def to_movie_names(movie_set):
	movies = {}
	movie_file = os.path.dirname(os.path.abspath(__file__))+"/files/movies.txt"
	f = open(movie_file,'r')
	for m in f:
		movie_data = m.split('\t')
		movies[movie_data[0]] = movie_data[1]
	output = []
	for item in movie_set:
		output.append((movies[item[0]],item[1:]))
	return output

def user_ratings(trainset,user,with_rating):
	user_movies=[]
	user_id = trainset.to_inner_uid(user)
	for user_movie in trainset.ur[user_id]:
		if (with_rating):
			user_movies.append((trainset.to_raw_iid(user_movie[0]),user_movie[1]))
		else:
	                user_movies.append(user_movie[0])
	return user_movies

def get_ratings(recommender,user,names):
	ratings = user_ratings(recommender.trainset,user,True)
	if (names):
		return to_movie_names(ratings)
	else:
		return ratings

def top_recom(recommender,user,n,names):
	user_testset = []
	user_movies = []
	trainset = recommender.trainset
	movies = trainset.all_items()
	user_movies = user_ratings(trainset,user,False)
	for m in movies:
		if m not in user_movies:
			user_testset.append((user,trainset.to_raw_iid(m),trainset.global_mean))
	predictions = recommender.test(user_testset)
	predicted_movies = []
	for p in predictions:
		predicted_movies.append((p[1],p[3]))
	predicted_movies.sort(key=itemgetter(1),reverse=True)
	result = []
	for i in range(n):
		result.append([predicted_movies[i][0],predicted_movies[i][1]])
	if (names):
		return to_movie_names(result)
	else:
		return result
	
	

def main():
	action = sys.argv[1]
	model = sys.argv[2]
	prog_dir = os.path.dirname(os.path.abspath(__file__))
	ds_dir = prog_dir+"/files_processed/"
	dump_file = prog_dir+"/models/"+model+".dmp"
	if action=='train' or action=='test':
		data_reader = Reader(line_format='user item rating', sep='\t')
		if model=='svd':
			recommender = SVD()
		elif model=='svd++':
			recommender = SVDpp()
		elif model=='knn_means':
			recommender = KNNWithMeans()
		if action == 'test':
			test=True
			print("Training and testing model")
		else:
			test=False
			print("Training model with no test")
		train_model(ds_dir,data_reader,test,recommender,dump_file)
	elif action=='predict' or action=='top':
		user = sys.argv[3]
		_, recommender = dump.load(dump_file)
		if (action=='predict'):
			print("Using model to predict")
			movie = sys.argv[4]
			rating = use_model(recommender,user,movie)
			print(rating)
		else:
			print("Using top n")
			n = int(sys.argv[4])
			if sys.argv[5] == "names":
				names = True
			else:
				names = False
			recommendations = top_recom(recommender,user,n,names)
			print(recommendations)
	elif action=='data':
		user = sys.argv[3]
		_, recommender = dump.load(dump_file)
		print("Printing rating data for user")
		if sys.argv[4] == "names":
			names = True
		else:
			names = False
		print(get_ratings(recommender,user,names))	

main()
