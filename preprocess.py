import sys

input_file = open('files/'+sys.argv[1],'r')
output_file = open('files_processed/'+sys.argv[1],'w')

for line in input_file:
	record = line.split('\t')
	output_file.write(record[0]+'\t'+record[1]+'\t'+record[3])

input_file.close()
output_file.close()
