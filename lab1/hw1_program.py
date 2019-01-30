# CS74 Homework 1 - Making a Naive Bayes Classifier
# By Aidan Low

# A Bayesian classifier that associates 6 variables with a 1 or 0 output.
# Coded to be easily generalized.

from math import log
import time

# Time the run
start_time = time.time()

# defining constants
calibration_frac = 0.8      # Sets the fraction of the training set that should be used for internal tests.
iv_count = 6                # Sets the number of independent variables.
dv_count = 1                # Sets the number of dependent variables.
prefix = "../datafiles/"
training_set_loc = "hw1_trainingset.csv"    # Name of the training data file.
test_set_loc = "hw1_testset.csv"            # Name of the test data file.
results_loc = "hw1_results.csv"             # Name of the results file.


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
def train(training_filename):
    # Count lines.
    set_file = open(training_filename, "r")  # open the file
    count_lines = len(set_file.readlines()) - 1
    set_file.close()

    # Read in all items from a file and convert them into an array of items
    set_file = open(training_filename, "r")  # open the file
    set_file.readline()  # read title line so it isn't used as data
    testing_number = int(calibration_frac * count_lines)
    classifier = Classifier()
    for i in range(testing_number):  # read each line of the testing fraction
        words = set_file.readline().split(",")
        features = words[:-1]
        prior = words[-1]
        classifier.count_item(features, prior)

    # Having trained the classifier, pre-process probabilities to save runtime
    classifier.pre_process()

    # Test the remaining lines
    print("Testing starts at line " + str(testing_number) + ".")
    correct_count = 0
    line_count = 0
    for line in set_file:
        words = line.split(",")
        features = words[:-1]
        prior = words[-1]
        projected_value = classifier.predict(features)
        if projected_value == prior:
            correct_count += 1
        line_count += 1

    # No longer using training set
    set_file.close()

    # Report success (or not).
    if line_count > 0:
        print("Correct " + str(correct_count) + " out of " + str(line_count) + " times, for a " + str(
            100 * float(correct_count) / line_count) + "% success rate.")
    else:
        print("The entire dataset was used on training. No data was tested.")

    return classifier


# Run through the test set of data, get projections
def predict(testing_filename, output_filename, classifier):
    print("Starting predictions.")
    test_file = open(testing_filename, "r")
    output_file = open(output_filename, "w")
    output_file.write(test_file.readline()[:-1] + ",Label\n")
    for line in test_file:
        words = line.split(",")
        projected_value = classifier.predict(words)
        output_file.write(words[0] + "," + words[1] + "," + words[2] + "," + words[3] + "," + words[4] + "," + words[5][
                                                                                                               :-1] + "," + projected_value)
    test_file.close()
    output_file.close()

    print("Done. Check " + results_loc + " for results.")

# # Get the Gini Indices of the values of a given feature.
# def gini(feature_number, prior_number, training_filename):
#     set_file = open(training_filename, "r")
#     posteriors = {}
#     item_count = 0
#     for line in set_file:
#         words = line.split(",")
#         feature_value = words[feature_number]
#         prior = words[-prior_number]
#         try:
#             posteriors[feature_value][prior] += 1
#         except KeyError:
#             try:
#                 posteriors[feature_value][prior] = 1
#             except KeyError:
#                 posteriors[feature_value] = {}
#                 posteriors[feature_value][prior] = 1
#         item_count += 1
#     ginis_calculated = []
#     for feature_value in posteriors.keys():
#         feature_value_gini = 1
#         for prior in posteriors[feature_value].keys():
#             feature_value_gini -= (posteriors[feature_value][prior] / item_count) ** 2
#         ginis_calculated.append(feature_value_gini)
#     aggregegate_gini = sum(ginis_calculated) / len(ginis_calculated)
#     return aggregegate_gini
# print(gini(5, 1, training_set_loc))


# Run the code
trained_classifier = train(prefix + training_set_loc)
predict(prefix + test_set_loc, prefix + results_loc, trained_classifier)
print("--- %s seconds ---" % (time.time() - start_time))

