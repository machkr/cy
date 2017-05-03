from datetime import datetime, timedelta
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn import linear_model, svm, tree
from subprocess import call
from tldextract import extract
import csv
import numpy as np
import matplotlib.pyplot as plt
import re
import logging

# Hide warning
logging.getLogger("tldextract").setLevel(logging.CRITICAL)

def import_data(filename):

	# Extracted feature dictionaries
	dicts = []

	# List of labels
	labels = []

	# Registration Info {domain: timestamp}
	reg = {}

	# Open registration info
	with open('registration_info.csv', 'r') as file:

		# Initialize reader object
		reader = csv.reader(file)

		# Skip row of headers
		next(reader, None)

		#Process each line of data from CSV
		for line in list(reader):

			reg[line[0]] = datetime.strptime(line[1], '%Y-%m-%dT%H:%M:%S')

	# Open input file
	with open(filename, 'r') as file:

		# Initialize reader object
		reader = csv.reader(file)

		# Skip row of headers
		next(reader, None)

		#Process each line of data from CSV
		for line in list(reader):

			# Extract URL
			url = line[0]

			# Extract timestamp
			timestamp = datetime.strptime(line[1], '%Y-%m-%dT%H:%M:%S')

			# Extract label
			if line[3] == 'malicious':

				# Set to 0
				label = 0

			# If label is benign
			elif line[3] == 'benign':

				# Set to 1
				label = 1

			# Extract URL parts
			ext = extract(url)

			# Determine if domain is an IP address
			if re.match('^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', ext.domain):

				# Set feature
				ip = 1

			# Domain is not an IP address
			else:

				# Set feature
				ip = 0

			# If registered domain is matched in info file
			if ext.registered_domain in reg.keys():

				matched_key = ext.registered_domain

			# If suffix is matched in info file
			elif ext.suffix in reg.keys():

				matched_key = ext.suffix

			# If domain is matched in info file
			elif ext.domain in reg.keys():

				matched_key = ext.domain

			# If a match was found
			if matched_key:

				# Calculate time delta in days
				delta = (timestamp - reg[matched_key]).days

			# Perform regex split
			regex = re.split('\W+', url)

			# Extract and store features in dictionary
			entry = {
						'protocol': regex[0],
						'last_token': regex[-1],
						'num_tokens': len(regex),
						'avg_str_len': sum(map(len, regex)) / len(regex),
						'subdom_cnt': len(ext.subdomain.split('.')),
						'domain_len': len(ext.domain),
						'tld': ext.suffix,
						'ip': ip,
						'delta': delta,
						'dots': url.count('.'),
						'slashes': url.count('/')
					}

			# Append dictionary to data list
			dicts.append(entry)
			labels.append(label)

	# Initialize DictVectorizer object
	vec = DictVectorizer(sort=False)

	# Transform list of dictionaries to array of arrays
	features = vec.fit_transform(dicts).toarray()

	# Feature names
	names = vec.get_feature_names()

	# Return list of features and labels
	return features, labels, names

def cross_validation(clf, data, target):
	
	# Calculate cross validated score
	scores = cross_val_score(clf, data, target, cv=5)

	# Reformat scores
	scores = ['{:.0%}'.format(score) for score in scores]

	# Output scores
	print 'Cross Validation Scores: ' + ' '.join(scores)

def fit_classifier(clf, data, target):

	# Fit classifier to model
	fitted_clf = clf.fit(data, target)

	# Return fitted classifier
	return fitted_clf

def test_model(clf, data, target):

	# Test model
	score = clf.score(data, target)

	# Output score
	print 'Test Score: ' + '{:.0%}'.format(score)

def get_misclassified(clf, data, target):

	# Find misclassified instances
	misclassified = np.where(target != clf.predict(data))

	# Output
	print 'Misclassified Instance Indices (' + str(len(misclassified[0])) + ')'
	print 'First Five:' + str(misclassified[0][:15])

def predict(clf, data, filename):

	# Make predictions
	predictions = list(clf.predict(data))

	# Open text file
	with open(filename, 'w') as file:

		# For each prediction made
		for prediction in predictions:
			
			# Write to text file
			file.write('%s\n' % prediction)

def plot_svm(x, y):

	SVM = svm.SVC(kernel='linear', C=1).fit(x, y)

	# Plot decision boundary
	x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
	y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
	zz = SVM.predict(np.c_[xx.ravel(), yy.ravel()])

	# Place result in color plot
	zz = zz.reshape(xx.shape)
	plt.figure(1, figsize=(4, 3))
	plt.contourf(xx, yy, zz, cmap=plt.cm.coolwarm, alpha=0.8)

	# Plot training points
	plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm)
	plt.xlabel('')
	plt.ylabel('')
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.xticks(())
	plt.yticks(())
	plt.title('Support Vector Machines')

	# Save plot
	plt.savefig('svm.png')
	print 'Saved plot to \'svm.png\'.'

def plot_log(x, y):

	LOG = linear_model.LogisticRegression(C=1e5).fit(x, y)

	# Plot the decision boundary
	x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
	y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
	xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
	zz = LOG.predict(np.c_[xx.ravel(), yy.ravel()])

	# Place result in color plot
	zz = zz.reshape(xx.shape)
	plt.figure(2, figsize=(4, 3))
	plt.pcolormesh(xx, yy, zz, cmap=plt.cm.Paired)

	# Plot training points
	plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
	plt.xlabel('')
	plt.ylabel('')
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.xticks(())
	plt.yticks(())
	plt.title('Logistic Regression')

	# Show plot
	plt.savefig('log.png')
	print 'Saved plot to \'log.png\'.'

def plot_dec(x, y, feature_names, target_names):

	DEC = tree.DecisionTreeClassifier()
	DEC.fit(x, y)
	tree.export_graphviz(DEC, out_file='tree.dot', 
		feature_names = feature_names, class_names = target_names)
	call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'])
	call(['rm', 'tree.dot'])
	print 'Saved plot to \'tree.png\'.'

def run(clf, data, target, filename):

	# Split data into random training and test subsets
	x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.4, random_state=0)

	# Cross validation
	cross_validation(clf, x_train, y_train)

	# Fit classifier
	FITTED_CLF = fit_classifier(clf, x_train, y_train)

	# Test model
	test_model(FITTED_CLF, x_test, y_test)

	# Get misclassified instances
	get_misclassified(FITTED_CLF, x_test, y_test)

	# Perform prediction, output results to filename
	predict(FITTED_CLF, data, filename)

def main():

	# Request filename
	filename = input('Filename:')

	# Check if filename was entered
	if not filename:

		# Default filename
		filename = 'input.csv'

	# Import training data from CSV file
	data, target, feature_names = import_data(filename)
	
	# SVM
	print '\n-- Support Vector Machines --'
	SVM = svm.SVC(kernel='linear', C=1)
	run(SVM, scale(data), target, 'svm-output.txt')
	plot_svm(data[:, :2], target)

	# Logistic Regression
	print '\n-- Logistic Regression --'
	LOG = linear_model.LogisticRegression(C=1e5)
	run(LOG, scale(data), target, 'logistic-output.txt')
	plot_log(data[:, :2], target)

	# Decision Tree
	print '\n-- Decision Tree --'
	DEC = tree.DecisionTreeClassifier()
	run(DEC, scale(data), target, 'dtree-output.txt')
	plot_dec(scale(data), target, feature_names, ['benign', 'malicious'])

if __name__ == '__main__':
	main()
