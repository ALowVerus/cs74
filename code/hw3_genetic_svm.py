# Assignment: CS74 Homework Assignment 3: Hyper-parameter Optimization through Genetic Training
# Name: Aidan Low
"""
This program does the following:
1. takes in data from a testing file,
2. randomly generates initial SVM hyper-parameters,
3. ranks those hyper-parameters by accuracy in a listed hierarchy, then
4. generates children as
    a. mutants (versions of the top hyper-parameter sets with a single parameter changed) and
    b. crossovers (children of two highly-ranked hyper-parameter sets, with each hyper-parameter of the child randomly
        sourced from either parent),
5. adds those new children to the hierarchy of hyper-parameter sets,
6. does 3-5 until only a small amount of lift comes from adding new children,
7. returns the highest-performing hyper-parameter set, and finally
8. uses an SVM trained by the highest-performing hyper-parameter set to predict the values of a testing data set.

The best run I've gotten from this returned the following parameters:
{
    'shrinking': True, 'accuracy': 0.7614285714285713, 'max_iter': 100000,
    'tol': 0.001, 'gamma': 'scale', 'coef0': 0.0, 'C': 2.0, 'kernel': 'rbf'
}
I'm not sure whether you'll be able to reproduce this 76% accurate result, given that I use a partially
random genetic algorithm to optimize my hyper-parameters. If you don't, run the program again.

I tried normalizing my data using from sklearn.preprocessing.scale() but that ended up reducing my accuracy down to 56%.
Maybe normalizing the data squishes it together, so it is harder to draw lines between one group of data and another?
"""

from sklearn.svm import SVC
import numpy as np
from random import randint
import warnings
import time
import json
import shared_library

# Disable calls for pre-processing data
warnings.filterwarnings('ignore', 'Solver terminated early.*')

# Set constants
iv_count = 6                             # Number of features in data set
prefix = "../datafiles/hw4_"                    # Used if data files are not in same directory as code
training_set_loc = prefix + 'training_set.csv'
testing_set_loc = prefix + 'test_set.csv'
results_loc = prefix + 'output_set_genetic_svm.csv'
best_item_loc = prefix + 'best_item.json'
number_of_cross_validating_samples = 2          # Number of sections of cross-validation data
number_of_genetic_samples = 5                  # Size of natural selection gene pool used in genetic development


class GeneticSVMClassifier:
    def __init__(self, data, validating=True, best_item_loc=False):
        # A list of hyper-parameter options to draw from during my genetic training phase
        self.HPO = {
            "C": [0.01, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0],
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "gamma": ["auto", "scale"],
            "degree": [1, 2, 3],
            "coef0": [0.0, 0.1, 0.2, 0.3, 1.0, 10.0],
            "shrinking": [True, False],
            "tol": [0.01, 0.001, 0.0001, 0.00001],
            "max_iter": [100, 1000, 10000, 100000]
        }
        # Minimum amount of extra precision required to continue testing
        self.lift_floor = 0.01
        # A list of keys and a corresponding list of the number of options for each key
        self.hp_keys = list(self.HPO.keys())
        self.hp_lengths = []
        for k in range(len(self.hp_keys)):
            self.hp_lengths.append(len(self.HPO[self.hp_keys[k]]))
        # A baseline best item, in case no external best item exists
        if best_item_loc is not False:
            # Check whether a dynamic seed location exists
            self.input_best_item = self.get_best_from_json(best_item_loc)
        else:
            self.input_best_item = self.get_random_parameters()
        # A global dictionary for containing probabilities already calculated
        self.calculated_probability_dict = {}
        # Split data into corresponding lists
        self.training_feature_list = self.list_entry(data, 'features')
        self.training_label_list = self.list_entry(data, 'label')
        # TRAIN
        self.optimal_parameters = self.get_optimal_parameters(self.training_feature_list, self.training_label_list)
        self.svm = self.make_machine_from_params(self.optimal_parameters)
        self.svm.fit(np.array(self.training_feature_list), np.array(self.training_label_list))
        # Save the optimal parameters in a safe location
        self.throw_best_in_json(best_item_loc)

    # Abstracted SVM generation to save space.
    def make_machine_from_params(self, op):
        return SVC(
            kernel=op["kernel"], max_iter=op["max_iter"], C=op["C"],
            gamma=op["gamma"], shrinking=op["shrinking"], tol=op["tol"],
            degree=op["degree"]
        )

    # Calculate the accuracy of a set of hyper-parameters.
    def calculate_accuracy_of_item(self, feature_list, label_list, item):
        print("Training with " + str(item))
        # Check to ensure that accuracies are not being needlessly recalculated.
        if item.keys().__contains__("accuracy"):
            print("Item already has an associated accuracy.")
        else:
            item_key = ""
            for key in self.hp_keys:
                item_key += str(item[key]) + ","
            try:
                accuracy = self.calculated_probability_dict[item_key]
                print("Already calculated. Accuracy is " + "{:.2%}".format(accuracy) + ".")
            except KeyError:
                # Generate a machine and run it only if this has not yet been done for the input set of hyper-parameters.
                machine = self.make_machine_from_params(item)
                accuracy = self.get_accuracy(machine, feature_list, label_list, number_of_cross_validating_samples)
                self.calculated_probability_dict[item_key] = accuracy
                print("Done! Accuracy is " + "{:.2%}".format(accuracy) + ".")
            item["accuracy"] = accuracy

    # Calculate net accuracy through 10-fold validation
    def get_accuracy(self, svm, feature_list, label_list, sample_count):
        accuracy = 0.0
        feature_list_list = []
        label_list_list = []
        list_size = len(feature_list)
        for i in range(sample_count):
            feature_list_list.append(feature_list[int(i*list_size/sample_count):int((i+1)*list_size/sample_count)])
            label_list_list.append(label_list[int(i*list_size/sample_count):int((i+1)*list_size/sample_count)])
        for i in range(sample_count):
            testing_features = feature_list_list[i]
            testing_labels = label_list_list[i]
            training_features = []
            training_labels = []
            for j in range(sample_count):
                if i != j:
                    training_features.extend(feature_list_list[j])
                    training_labels.extend(label_list_list[j])
            training_feature_array = np.array(training_features)
            training_label_array = np.array(training_labels)
            svm.fit(training_feature_array, training_label_array)
            # Get predictions
            correct_count = 0.0
            for j in range(len(testing_features)):
                predicted_label = svm.predict([testing_features[j]])
                actual_label = testing_labels[j]
                if predicted_label == actual_label:
                    correct_count += 1.0
            correct_decimal = correct_count / len(testing_features)
            accuracy += correct_decimal
        accuracy /= sample_count
        return accuracy

    def make_random_parameters(self):
        item = {}
        for j in range(len(self.hp_keys)):
            item[self.hp_keys[j]] = self.HPO[self.hp_keys[j]][randint(0, self.hp_lengths[j] - 1)]
        return item

    # Generate an initial set of hyper-parameters to grow from, before using genetic weeding methods.
    def generate_initial_hierarchy(self, feature_list, label_list):
        print("Generating initial hierarchy.")
        # Randomly select initial values for a number of sample items
        hierarchy = []
        for i in range(number_of_genetic_samples):
            hierarchy.append(self.make_random_parameters())
        # Put my best hyper-parameters from previous runs into the hierarchy to act as breeding stock
        hierarchy[0] = self.input_best_item
        # Calculate the value of each item
        for item in hierarchy:
            self.calculate_accuracy_of_item(feature_list, label_list, item)
        # Sort the items into a hierarchy to determine which should be used as breeding stock
        self.quick_sort(hierarchy)
        return hierarchy

    # Generate mutated and crossover items using the top hyper-parameters as parents, then re-rank.
    def update_hierarchy(self, feature_list, label_list, hierarchy):
        # Generate mutated items
        print("Generating mutated items.")
        mutated_item_list = []
        for i in range(int(len(hierarchy)/4)):
            next_item = hierarchy[-(1 + i)].copy()
            del next_item["accuracy"]
            mutation_index = randint(0, len(self.hp_keys) - 1)
            next_item[self.hp_keys[mutation_index]] = \
                self.HPO[self.hp_keys[mutation_index]][randint(0, self.hp_lengths[mutation_index] - 1)]
            mutated_item_list.append(next_item)
        # Generate crossover items
        print("Generating crossover items.")
        crossover_item_list = []
        for i in range(int(len(hierarchy)/4)):
            item_a = hierarchy[-(1 + randint(0, int(len(hierarchy)/4)))]
            item_b = hierarchy[-(1 + randint(0, int(len(hierarchy)/4)))]
            next_item = {}
            for j in range(len(self.hp_keys)):
                next_item[self.hp_keys[j]] = [item_a, item_b][randint(0, 1)][self.hp_keys[j]]
            crossover_item_list.append(next_item)
        # Calculate probabilities of items
        print("Calculating mutated item probabilities.")
        for item in mutated_item_list:
            self.calculate_accuracy_of_item(feature_list, label_list, item)
        print("Calculating crossover item probabilities.")
        for item in crossover_item_list:
            self.calculate_accuracy_of_item(feature_list, label_list, item)
        # Swap new mutated and crossover items with the items with the lowest scores
        print("Swapping new items with worst items from the last iteration.")
        for i in range(int(len(hierarchy) / 4)):
            hierarchy[i] = mutated_item_list[i]
        for i in range(int(len(hierarchy) / 4)):
            hierarchy[i + int(len(hierarchy) / 4)] = crossover_item_list[i]
        # Sort the updated list
        self.quick_sort(hierarchy)
        return hierarchy

    # Predict optimal hyper-parameters for an SVM using a genetic algorithm. Stop when the last run gave minimal lift.
    def get_optimal_parameters(self, feature_list, label_list):
        hierarchy = self.generate_initial_hierarchy(feature_list, label_list)
        print("\n Iteration 0: " + str(hierarchy) + "\n")
        done = False
        counter = 1
        while not done:
            # Calculate the gain in accuracy in the top contenders between the last and the current runs.
            last_hierarchy = hierarchy.copy()
            self.update_hierarchy(feature_list, label_list, hierarchy)
            print("\n Iteration " + str(counter) + ": " + str(hierarchy) + "\n")
            average_gained_accuracy = 0.0
            for i in range(int(len(hierarchy)/2)):
                index = int(len(hierarchy)/2) + i
                average_gained_accuracy += hierarchy[index]["accuracy"] - last_hierarchy[index]["accuracy"]
            average_gained_accuracy /= int(len(hierarchy)/2)
            print("Gained " + "{:.2%}".format(average_gained_accuracy) + " average accuracy during this last run.")
            # If the gain in accuracy is too low, end the program.
            if average_gained_accuracy <= self.lift_floor:
                done = True
            counter += 1
        winner = hierarchy[-1]
        print("The winner is: " + str(winner))
        print("Accuracy is " + "{:.2%}".format(winner['accuracy']) + ".")
        del winner["accuracy"]
        return winner

    # Quicksort is used to re-order mutant and crossover species into the hierarchy of hyper-parameters.
    def quick_sort(self, my_list):
        print("Sorting the hierarchy of SVM parameters.")
        self.quick_sort_helper(my_list, 0, len(my_list) - 1)
        print("Sorting complete.")

    def quick_sort_helper(self, my_list, first, last):
        if first < last:
            split_point = self.partition(my_list, first, last)
            self.quick_sort_helper(my_list, first, split_point - 1)
            self.quick_sort_helper(my_list, split_point + 1, last)

    def partition(self, my_list, first, last):
        pivot_value = my_list[first]["accuracy"]
        lower_index = first + 1
        higher_index = last
        done = False
        while not done:
            while lower_index <= higher_index and my_list[lower_index]["accuracy"] <= pivot_value:
                lower_index += 1
            while lower_index <= higher_index and my_list[higher_index]["accuracy"] >= pivot_value:
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

    # If the best item has from the last run has already been written to json, pull it rather than using the default.
    def get_best_from_json(self, target_location):
        with open(target_location, 'r') as fp:
            data = json.load(fp)
            fp.close()
        return data

    def throw_best_in_json(self, target_location):
        with open(target_location, 'w') as fp:
            json.dump(self.optimal_parameters, fp)
            fp.close()

    # Get a list of entries of a data type
    def list_entry(self, data, key):
        entry_list = []
        for datum in data:
            entry_list.append(datum[key])
        return entry_list

    def predict(self, features):
        return self.svm.predict([features])


# Run the code.
shared_library.main(
    Model=GeneticSVMClassifier,
    best_item_loc=best_item_loc,
    training_set_loc=training_set_loc,
    testing_set_loc=testing_set_loc,
    results_loc=results_loc,
    iv_count=iv_count
)
