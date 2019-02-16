# CS74 Homework 1 - Making a Naive Bayes Classifier
# By Aidan Low

# A Bayesian classifier that associates 6 variables with a 1 or 0 output.
# Coded to be easily generalized.
from math import log
import time
import sys
sys.path.insert(0, '../datafiles')
import shared_library

# Time the run
start_time = time.time()

# defining constants
calibration_frac = 0.8      # Sets the fraction of the training set that should be used for internal tests.
iv_count = 6                # Sets the number of independent variables.
dv_count = 1                # Sets the number of dependent variables.
prefix = "../datafiles/hw1_"
training_set_loc = "training_set.csv"    # Name of the training data file.
test_set_loc = "test_set.csv"            # Name of the test data file.
results_loc = "output_set.csv"             # Name of the results file.


# The classifier itself, with appropriate internal methods.
class Classifier:
    def __init__(self):
        self.prior_counts = {}
        self.prior_prob_logs = {}
        self.feature_counts_given_prior = {}
        self.feature_counts = {}
        self.total_trained_items = 0

    # Add appropriate counts for prior, count, and count given prior
    def count_item(self, features, prior):
        # Increment probability of prior
        try:        # try to increment the count
            self.prior_counts[prior] = self.prior_counts[prior] + 1
        except KeyError:     # but the value for the feature may not exist yet, so, if it doesn't
            self.prior_counts[prior] = 1
        # Increment probability of feature, given prior
        for feature_number in range(len(features)):
            feature_value = features[feature_number]
            try:
                self.feature_counts_given_prior[prior][feature_number][feature_value] = \
                    self.feature_counts_given_prior[prior][feature_number][feature_value] + 1
            except KeyError:
                try:
                    self.feature_counts_given_prior[prior][feature_number][feature_value] = 1
                except KeyError:
                    try:
                        self.feature_counts_given_prior[prior][feature_number] = {}
                        self.feature_counts_given_prior[prior][feature_number][feature_value] = 1
                    except KeyError:
                        self.feature_counts_given_prior[prior] = {}
                        self.feature_counts_given_prior[prior][feature_number] = {}
                        self.feature_counts_given_prior[prior][feature_number][feature_value] = 1
        # Increment probability of feature w/o prior
        for feature_number in range(len(features)):
            feature_value = features[feature_number]
            try:
                self.feature_counts[feature_number][feature_value] = \
                    self.feature_counts[feature_number][feature_value] + 1
            except KeyError:
                try:
                    self.feature_counts[feature_number][feature_value] = 1
                except KeyError:
                    self.feature_counts[feature_number] = {}
                    self.feature_counts[feature_number][feature_value] = 1
        # Increment total
        self.total_trained_items = self.total_trained_items + 1

    def pre_process(self):
        # Calculate prior probability, having collected all counts
        for prior in self.prior_counts.keys():
            prob_prior = self.prior_counts[prior] / sum(self.prior_counts.values())
            self.prior_prob_logs[prior] = log(prob_prior)

    # Project the dependent variable given a line of independent variables
    def predict(self, features):
        calculated_final_probs = {}
        # For each possible future
        for prior in self.prior_counts.keys():
            # Get prior probability
            prob_prior = self.prior_prob_logs[prior]
            # Initialized total prior probabilities
            prob_feature_no_prior_total = 0.0
            prob_feature_given_prior_total = 0.0
            for feature_number in range(iv_count):
                # Get a feature
                feature = features[feature_number]
                # Get the probability that the feature is valid w/o prior
                try:
                    prob_feature_no_prior = (self.feature_counts[feature_number][feature] + 1) \
                        / (sum(self.feature_counts[feature_number].values()) + self.total_trained_items)
                except KeyError:
                    prob_feature_no_prior = (0 + 1) \
                        / (sum(self.feature_counts[feature_number].values()) + self.total_trained_items)
                prob_feature_no_prior_total += log(prob_feature_no_prior)
                # Get the probability that the feature is valid with prior
                try:
                    prob_feature_given_prior = (self.feature_counts_given_prior[prior][feature_number][feature] + 1) \
                        / (sum(self.feature_counts_given_prior[prior][feature_number].values()) + self.total_trained_items)
                except KeyError:
                    prob_feature_given_prior = (0 + 1) \
                        / (sum(self.feature_counts_given_prior[prior][feature_number].values()) + self.total_trained_items)
                prob_feature_given_prior_total += log(prob_feature_given_prior)
            # Bayes Equation to calculate the possibility of future, given features
            calculated_final_prob = prob_feature_given_prior_total + prob_prior - prob_feature_no_prior_total
            calculated_final_probs[prior] = calculated_final_prob
        # Find which prior is valid
        priors_list = list(calculated_final_probs.keys())
        max_prior = priors_list[0]
        for prior in priors_list[1:]:
            if calculated_final_probs[prior] > calculated_final_probs[max_prior]:
                max_prior = prior
        return max_prior


# Given a filename, create a classifier and train it with the data inside.
def train(data):
    # Read in all items from a file and convert them into an array of items
    training_number = int(calibration_frac * len(data))
    testing_number = len(data) - training_number
    # Initialize the classifier
    classifier = Classifier()
    # Read in data
    for i in range(training_number):  # read each line of the testing fraction
        classifier.count_item(data[i]["features"], data[i]["label"])
    # Having trained the classifier, pre-process probabilities to save runtime
    classifier.pre_process()
    # Test the remaining lines
    print("Testing starts at line " + str(training_number) + ".")
    correct_count = 0
    for i in range(testing_number):
        projected_value = classifier.predict(data[i + training_number]["features"])
        if projected_value == data[i + training_number]["label"]:
            correct_count += 1
    # Report success (or not).
    if testing_number > 0:
        print("Correct " + str(correct_count) + " out of " + str(testing_number) + " times, for a " + str(
            100 * float(correct_count) / testing_number) + "% success rate.")
    else:
        print("The entire dataset was used on training. No data was tested.")

    return classifier


# Run through the test set of data, get projections
def predict(data, classifier):
    print("Starting predictions.")
    for item in data:
        features = item['features']
        item['label'] = classifier.predict(features)
    print("Done. Check " + results_loc + " for results.")


# Run the code
training_data = shared_library.get_data(prefix + training_set_loc, iv_count)
trained_classifier = train(training_data)
testing_data = shared_library.get_data(prefix + test_set_loc, iv_count)
predict(testing_data, trained_classifier)
shared_library.write_results(testing_data, prefix + results_loc)
print("--- %s seconds ---" % (time.time() - start_time))

