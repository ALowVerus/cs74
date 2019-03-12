import time
import shared_library
from hw1_naive_bayes import NaiveBayesClassifier

# Set constants
iv_count = 6                             # Number of features in data set
validation_count = 10
prefix = "../datafiles/hw4_"                    # Used if data files are not in same directory as code
training_set_loc = prefix + 'training_set.csv'
testing_set_loc = prefix + 'test_set.csv'
results_loc = prefix + 'output_set_binary_multiclasser.csv'


class MultiClassNaiveBayesClassifier:
    def __init__(self, best_item_loc=False):
        self.accuracies = {}
        self.classifiers = {}
        self.dvs = {}

    def store_actual_labels(self, data):
        for datum in data:
            if 'actual_label' not in list(datum.keys()):
                datum['actual_label'] = datum['label']

    def train_with_data(self, data):
        self.accuracies = {}
        self.classifiers = {}
        self.store_actual_labels(data)
        # Enumerate labels, split data by labels
        dv_split_data = {}
        for datum in data:
            dv = datum['actual_label']
            try:
                dv_split_data[dv].append(datum)
            except KeyError:
                dv_split_data[dv] = []
                dv_split_data[dv].append(datum)
        # Make a copy of the data to freely manipulate
        self.dvs = list(dv_split_data.keys())
        # Generate splitter list and classifiers
        for dv_1 in self.dvs:
            for dv_2 in self.dvs:
                if dv_1 < dv_2:
                    # Make a subset of data
                    working_data = []
                    working_data.extend(dv_split_data[dv_1])
                    working_data.extend(dv_split_data[dv_2])
                    # Get accuracies of each splitter
                    classifier = NaiveBayesClassifier()
                    self.make_data_true_on_sole_dv(working_data, dv_1)
                    accuracy = classifier.n_fold_validate(working_data, 0)
                    try:
                        self.accuracies[dv_1][dv_2] = round(accuracy, 4)
                    except KeyError:
                        self.accuracies[dv_1] = {}
                        self.accuracies[dv_1][dv_2] = round(accuracy, 4)
                    # Retrain with all data
                    classifier.train_with_data(working_data)
                    try:
                        self.classifiers[dv_1][dv_2] = classifier
                    except KeyError:
                        self.classifiers[dv_1] = {}
                        self.classifiers[dv_1][dv_2] = classifier

    def test_with_data(self, data):
        correct_count = 0
        for datum in data:
            if datum["actual_label"] == self.predict(datum["features"]):
                correct_count += 1
        return correct_count / len(data)

    def predict(self, features):
        confidences = {}
        for dv in self.dvs:
            confidences[dv] = 0.0
        for dv_1 in self.dvs:
            for dv_2 in self.dvs:
                if dv_1 < dv_2:
                    prediction = self.classifiers[dv_1][dv_2].predict(features)
                    reliability = self.accuracies[dv_1][dv_2]
                    if prediction == 1:
                        winner = dv_1
                    else:
                        winner = dv_2
                    confidences[winner] += reliability
        # Find best confidence
        best = self.dvs[0]
        for dv in self.dvs:
            if confidences[dv] > confidences[best]:
                best = dv
        return best

    def make_data_true_on_sole_dv(self, data, target_dv):
        for datum in data:
            datum['label'] = int(datum['actual_label'] == target_dv)

    def n_fold_validate(self, data, sample_count):
        self.store_actual_labels(data)
        subset_list = []
        list_size = len(data)
        for i in range(sample_count):
            subset_list.append(data[int(i * list_size / sample_count):int((i + 1) * list_size / sample_count)])
        total_accuracy = 0.0
        for i in range(sample_count):
            data_set_without_chosen_sample = []
            for j in range(sample_count):
                if i != j:
                    data_set_without_chosen_sample += subset_list[j]
            chosen_sample = subset_list[i]
            self.train_with_data(data_set_without_chosen_sample)
            accuracy = self.test_with_data(chosen_sample)
            print("Accuracy for round " + str(i) + " is " + str(round(accuracy, 3)))
            total_accuracy += accuracy
        total_accuracy /= sample_count
        return total_accuracy


# Run the code.
shared_library.main(
    Model=MultiClassNaiveBayesClassifier,
    training_set_loc=training_set_loc,
    testing_set_loc=testing_set_loc,
    results_loc=results_loc,
    iv_count=iv_count,
    validation_count=validation_count
)
