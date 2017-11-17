"# CP600_rs" 
Run using python prog.py <action> <model> <parameter1> <parameter2>
Actions:
	train: python prog.py train <model>
	test: python prog.py test <model>
	predict: python prog.py predict <model> <user_id> <movie_id> 
	top: python prog.py top <model> <user_id> <top_n> <print_names>
	data: python prog.py data <model> <user_id> <print_names>
Models:
	svd
	svd++
	knn_means
Print names:
	names to print names
	empty otherwise