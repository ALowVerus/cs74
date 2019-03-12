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


class RandomForestClassifier:
    def __init__(self, best_item_loc=False):
        self.classifiers = []

    def train_with_data(self, data):
        partition_count = 10
        self.classifiers = []
        # Generate classifiers
        for i in range(partition_count):
            # Get accuracies of each splitter
            classifier = NaiveBayesClassifier()
            data_section = data[int(i * len(data) / partition_count):int((i + 1) * len(data) / partition_count + 1)]
            classifier.train_with_data(data_section)
            self.classifiers.append(classifier)

    def test_with_data(self, data):
        correct_count = 0
        for datum in data:
            if datum["label"] == self.predict(datum["features"]):
                correct_count += 1
        return correct_count / len(data)

    def predict(self, features):
        prediction_counts = {}
        for classifier in self.classifiers:
            prediction = classifier.predict(features)
            try:
                prediction_counts[prediction] += 1
            except KeyError:
                prediction_counts[prediction] = 1
        predictions = list(prediction_counts.keys())
        best_prediction = predictions[0]
        for next_prediction in predictions:
            if prediction_counts[next_prediction] > prediction_counts[best_prediction]:
                best_prediction = next_prediction
        return best_prediction

    def n_fold_validate(self, data, sample_count):
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
    Model=RandomForestClassifier,
    training_set_loc=training_set_loc,
    testing_set_loc=testing_set_loc,
    results_loc=results_loc,
    iv_count=iv_count,
    validation_count=validation_count
)
