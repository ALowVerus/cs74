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


class OneVsAllClassifier:
    def __init__(self, best_item_loc=False):
        self.dvs = []
        self.classifiers = {}
        self.accuracies = {}

    def store_actual_labels(self, data):
        for datum in data:
            if 'actual_label' not in list(datum.keys()):
                datum['actual_label'] = datum['label']

    def train_with_data(self, data):
        self.store_actual_labels(data)
        self.dvs = set()
        self.classifiers = {}
        self.accuracies = {}
        # Enumerate labels, split data by labels
        for datum in data:
            self.dvs.add(datum['actual_label'])
        # Make a copy of the data to freely manipulate
        self.dvs = list(self.dvs)
        # Generate classifiers
        for dv in self.dvs:
            # Get accuracies of each splitter
            classifier = NaiveBayesClassifier()
            self.make_data_true_on_sole_dv(data, dv)
            accuracy = classifier.n_fold_validate(data, 0)
            self.accuracies[dv] = round(accuracy, 4)
            classifier.train_with_data(data)
            self.classifiers[dv] = classifier
        self.quick_sort(self.dvs)

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
            prediction = self.classifiers[self.dvs[i]].predict(features)
            if prediction == 1:
                done = True
            elif i >= len(self.dvs) - 1:
                done = True
            else:
                i += 1
        return self.dvs[i]

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

    def quick_sort(self, my_list):
        print("Sorting the hierarchy of SVM parameters.")
        self.quick_sort_helper(my_list, 0, len(my_list) - 1)
        for i in range(int(len(my_list) / 2)):
            temp = my_list[i]
            my_list[i] = my_list[-(1 + i)]
            my_list[-(1 + i)] = temp
        print("Sorting complete.")

    def quick_sort_helper(self, my_list, first, last):
        if first < last:
            split_point = self.partition(my_list, first, last)
            self.quick_sort_helper(my_list, first, split_point - 1)
            self.quick_sort_helper(my_list, split_point + 1, last)

    def partition(self, my_list, first, last):
        pivot_value = self.accuracies[my_list[first]]
        lower_index = first + 1
        higher_index = last
        done = False
        while not done:
            while lower_index <= higher_index and self.accuracies[my_list[lower_index]] <= pivot_value:
                lower_index += 1
            while lower_index <= higher_index and self.accuracies[my_list[higher_index]] >= pivot_value:
                higher_index -= 1
            if lower_index > higher_index:
                done = True
            else:
                temp = my_list[lower_index]
                my_list[lower_index] = my_list[higher_index]
                my_list[higher_index] = temp
        temp = my_list[first]
        my_list[first] = my_list[higher_index]
        my_list[higher_index] = temp
        return higher_index


# Run the code.
shared_library.main(
    Model=OneVsAllClassifier,
    training_set_loc=training_set_loc,
    testing_set_loc=testing_set_loc,
    results_loc=results_loc,
    iv_count=iv_count,
    validation_count=validation_count
)
