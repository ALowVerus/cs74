# CS74 Homework 1 - Making a Naive Bayes Classifier
# By Aidan Low

# A Bayesian classifier that associates 6 variables with a 1 or 0 output.
# Coded to be easily generalized.
from math import log
import time
import shared_library

# Time the run
start_time = time.time()

# defining constants
iv_count = 6                # Sets the number of independent variables.
level_of_validation = 10     # Levels of cross-validation
prefix = "../datafiles/hw4_"
training_set_loc = prefix + "training_set.csv"    # Name of the training data file.
testing_set_loc = prefix + "test_set.csv"            # Name of the test data file.
results_loc = prefix + "output_set_naive_bayes.csv"             # Name of the results file.


# The classifier itself, with appropriate internal methods.
class NaiveBayesClassifier:
    def __init__(self, data, validating=True, best_item_url=False, best_item_loc=False):
        self.dv_counts = {}
        self.dv_prob_logs = {}
        self.feature_counts_for_dv = {}
        self.feature_counts = {}
        self.total_trained_items = 0
        if validating:
            print("Accurate {:.2%} of the time.".format(self.n_fold_validate(data, level_of_validation)))
            # Train with entire site
        self.train_with_data(data)

    # Add appropriate counts for dv, count, and count given dv
    def count_item(self, features, dv):
        # Increment probability of dv
        try:        # try to increment the count
            self.dv_counts[dv] = self.dv_counts[dv] + 1
        except KeyError:     # but the value for the feature may not exist yet, so, if it doesn't
            self.dv_counts[dv] = 1
        # Increment probability of feature, given dv
        for feature_number in range(len(features)):
            feature_value = features[feature_number]
            try:
                self.feature_counts_for_dv[dv][feature_number][feature_value] = \
                    self.feature_counts_for_dv[dv][feature_number][feature_value] + 1
            except KeyError:
                try:
                    self.feature_counts_for_dv[dv][feature_number][feature_value] = 1
                except KeyError:
                    try:
                        self.feature_counts_for_dv[dv][feature_number] = {}
                        self.feature_counts_for_dv[dv][feature_number][feature_value] = 1
                    except KeyError:
                        self.feature_counts_for_dv[dv] = {}
                        self.feature_counts_for_dv[dv][feature_number] = {}
                        self.feature_counts_for_dv[dv][feature_number][feature_value] = 1
        # Increment probability of feature w/o dv
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
        # Calculate dv probability, having collected all counts
        for dv in self.dv_counts.keys():
            prob_dv = self.dv_counts[dv] / sum(self.dv_counts.values())
            self.dv_prob_logs[dv] = log(prob_dv)

    def train_with_data(self, data):
        self.dv_counts = {}
        self.dv_prob_logs = {}
        self.feature_counts_for_dv = {}
        self.feature_counts = {}
        self.total_trained_items = 0
        # Read in data
        for datum in data:  # read each line of the testing fraction
            self.count_item(datum["features"], datum["label"])
        # Having trained the classifier, pre-process probabilities to save runtime
        self.pre_process()

    def test_with_data(self, data):
        correct_count = 0
        for datum in data:
            if datum["label"] == self.predict(datum["features"]):
                correct_count += 1
        return correct_count / len(data)

    # Project the dependent variable given a line of independent variables
    def predict(self, features):
        calculated_final_probs = {}
        # For each possible future
        for dv in self.dv_counts.keys():
            # Get dv probability
            prob_dv = self.dv_prob_logs[dv]
            # Initialized total dv probabilities
            prob_feature_no_dv_total = 0.0
            prob_feature_given_dv_total = 0.0
            for feature_number in range(iv_count):
                # Get a feature
                feature = features[feature_number]
                # Get the probability that the feature is valid w/o dv
                try:
                    prob_feature_no_dv = (self.feature_counts[feature_number][feature] + 1) \
                        / (sum(self.feature_counts[feature_number].values()) + self.total_trained_items)
                except KeyError:
                    prob_feature_no_dv = (0 + 1) \
                        / (sum(self.feature_counts[feature_number].values()) + self.total_trained_items)
                prob_feature_no_dv_total += log(prob_feature_no_dv)
                # Get the probability that the feature is valid with dv
                try:
                    prob_feature_given_dv = (self.feature_counts_for_dv[dv][feature_number][feature] + 1) \
                        / (sum(self.feature_counts_for_dv[dv][feature_number].values()) + self.total_trained_items)
                except KeyError:
                    prob_feature_given_dv = (0 + 1) \
                        / (sum(self.feature_counts_for_dv[dv][feature_number].values()) + self.total_trained_items)
                prob_feature_given_dv_total += log(prob_feature_given_dv)
            # Bayes Equation to calculate the possibility of future, given features
            calculated_final_prob = prob_feature_given_dv_total + prob_dv - prob_feature_no_dv_total
            calculated_final_probs[dv] = calculated_final_prob
        # Find which dv is valid
        dvs_list = list(calculated_final_probs.keys())
        max_dv = dvs_list[0]
        for dv in dvs_list[1:]:
            if calculated_final_probs[dv] > calculated_final_probs[max_dv]:
                max_dv = dv
        return max_dv

    def n_fold_validate(self, data_set, sample_count):
        subset_list = []
        list_size = len(data_set)
        for i in range(sample_count):
            subset_list.append(data_set[int(i * list_size / sample_count):int((i + 1) * list_size / sample_count)])
        accuracy = 0.0
        for i in range(sample_count):
            data_set_without_chosen_sample = []
            for j in range(sample_count):
                if i != j:
                    data_set_without_chosen_sample += subset_list[j]
            chosen_sample = subset_list[i]
            self.train_with_data(data_set_without_chosen_sample)
            accuracy += self.test_with_data(chosen_sample)
        accuracy /= sample_count
        return accuracy


# # Run the code.
# shared_library.main(
#     Model=NaiveBayesClassifier,
#     training_set_loc=training_set_loc,
#     testing_set_loc=testing_set_loc,
#     results_loc=results_loc,
#     iv_count=iv_count
# )
