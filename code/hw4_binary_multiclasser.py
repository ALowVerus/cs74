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
        self.best_splitter_list = []
        self.best_classifier_list = []

    def store_actual_labels(self, data):
        for datum in data:
            if 'actual_label' not in list(datum.keys()):
                datum['actual_label'] = datum['label']

    def train_with_data(self, data):
        self.store_actual_labels(data)
        self.best_splitter_list = []
        self.best_classifier_list = []
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
        working_dvs = set(dv_split_data.keys())
        # Split data by labels
        best_splitter_list = []
        best_classifier_list = []
        # Generate splitter list and classifiers
        for i in range(len(list(dv_split_data.keys()))):
            # Make a subset of data
            working_data = []
            for dv in working_dvs:
                working_data.extend(dv_split_data[dv])
            # Get accuracies of each splitter
            accuracies = {}
            classifier = NaiveBayesClassifier()
            for dv in working_dvs:
                self.make_data_true_on_sole_dv(working_data, dv)
                accuracy = classifier.n_fold_validate(working_data, 0)
                accuracies[dv] = round(accuracy, 4)
            # Find the best splitter
            best_dv_splitter = list(working_dvs)[0]
            for key, value in accuracies.items():
                if value > accuracies[best_dv_splitter]:
                    best_dv_splitter = key
            print("Best splitter was " + str(best_dv_splitter) + ". " + str(accuracies))
            # Retrain with all data
            self.make_data_true_on_sole_dv(working_data, best_dv_splitter)
            classifier.train_with_data(working_data)
            best_classifier_list.append(classifier)
            best_splitter_list.append(best_dv_splitter)
            working_dvs.remove(best_dv_splitter)
        self.best_splitter_list = best_splitter_list
        self.best_classifier_list = best_classifier_list

    def test_with_data(self, data):
        correct_count = 0
        for datum in data:
            if datum["actual_label"] == self.predict(datum["features"]):
                correct_count += 1
        return correct_count / len(data)

    def predict(self, features):
        done = False
        i = 0
        while not done:
            prediction = self.best_classifier_list[i].predict(features)
            if prediction == 1:
                done = True
            else:
                i += 1
        return self.best_splitter_list[i]

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
