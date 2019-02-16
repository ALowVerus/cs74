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
import sys
sys.path.insert(0, '../datafiles')
import shared_library

# Disable calls for pre-processing data
warnings.filterwarnings('ignore', 'Solver terminated early.*')

# Set constants
features_in_use = 6                             # Number of features in data set
prefix = "../datafiles/hw4_"                    # Used if data files are not in same directory as code
training_filename = 'training_set.csv'
testing_filename = 'test_set.csv'
output_filename = 'output_set.csv'
best_item_location = 'best_item.json'
number_of_cross_validating_samples = 3          # Number of sections of cross-validation data and, as a result, number of cross-validation runs
number_of_genetic_samples = 15                  # Size of natural selection gene pool used in genetic development
lift_floor = 0.01                               # Minimum amount of extra precision required to merit another run of natural selection
best_already_written_to_JSON = False            # Set True if "hw3_best_item.json" exists, else set False

# A list of hyper-parameter options to draw from during my genetic training phase
HPO = \
{
    "C": [0.01, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0],
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "gamma": ["auto", "scale"],
    "degree": [1, 2, 3],
    "coef0": [0.0, 0.1, 0.2, 0.3, 1.0, 10.0],
    "shrinking": [True, False],
    "tol": [0.01, 0.001, 0.0001, 0.00001],
    "max_iter": [100, 1000, 10000, 100000]
}

# A list of keys and a corresponding list of the number of options for each key
hp_keys = list(HPO.keys())
hp_lengths = []
for k in range(len(hp_keys)):
    hp_lengths.append(len(HPO[hp_keys[k]]))

# A global dictionary for containing probabilities already calculated, such that no repeat calculations are performed
calculated_probability_dict = {}

# The best item I got from my own runs of the data, for breeding in the gene pool. Will be naturally selected out
# if it is not in fact viable, but will greatly speed up development if it is.
input_best_item = \
    {
        "gamma": "scale", "C": 2.0, "shrinking": True, "kernel": "rbf",
        "tol": 0.001, "coef0": 0.2, "max_iter": 10000, "degree": 1
    }


# Abstracted SVM generation to save space.
def make_machine_from_params(op):
    return SVC(
        kernel=op["kernel"], max_iter=op["max_iter"], C=op["C"],
        gamma=op["gamma"], shrinking=op["shrinking"], tol=op["tol"],
        degree=op["degree"]
    )


# Calculate the accuracy of a set of hyper-parameters.
def calculate_accuracy_of_item(feature_list, label_list, item):
    print("Training with " + str(item))
    # Check to ensure that accuracies are not being needlessly recalculated.
    if item.keys().__contains__("accuracy"):
        print("Item already has an associated accuracy.")
    else:
        item_key = ""
        for key in hp_keys:
            item_key += str(item[key]) + ","
        try:
            accuracy = calculated_probability_dict[item_key]
            print("Already calculated. Accuracy is " + "{:.2%}".format(accuracy) + ".")
        except KeyError:
            # Generate a machine and run it only if this has not yet been done for the input set of hyper-parameters.
            machine = make_machine_from_params(item)
            accuracy = get_accuracy(machine, feature_list, label_list, number_of_cross_validating_samples)
            calculated_probability_dict[item_key] = accuracy
            print("Done! Accuracy is " + "{:.2%}".format(accuracy) + ".")
        item["accuracy"] = accuracy


# Calculate net accuracy through 10-fold validation
def get_accuracy(svm, feature_list, label_list, sample_count):
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


def get_n_fold_validated_accuracy(accuracy_function, data_set, sample_count):
    subset_list = []
    list_size = len(data_set)
    for i in range(sample_count):
        subset_list.append(data_set[int(i * list_size / sample_count):int((i + 1) * list_size / sample_count)])
    accuracy = 0.0
    for i in range(sample_count):
        accuracy += accuracy_function(subset_list[i])
    accuracy /= sample_count
    return accuracy


def make_random_parameters():
    item = {}
    for j in range(len(hp_keys)):
        item[hp_keys[j]] = HPO[hp_keys[j]][randint(0, hp_lengths[j] - 1)]
    return item


# Generate an initial set of hyper-parameters to grow from, before using genetic weeding methods.
def generate_initial_hierarchy(feature_list, label_list):
    print("Generating initial hierarchy.")
    # Randomly select initial values for a number of sample items
    hierarchy = []
    for i in range(number_of_genetic_samples):
        hierarchy.append(make_random_parameters())
    # Put my best hyper-parameters from previous runs into the hierarchy to act as breeding stock
    hierarchy[0] = input_best_item
    # Calculate the value of each item
    for item in hierarchy:
        calculate_accuracy_of_item(feature_list, label_list, item)
    # Sort the items into a hierarchy to determine which should be used as breeding stock
    quick_sort(hierarchy)
    return hierarchy


# Generate mutated and crossover items using the top hyper-parameters as parents, then re-rank.
def update_hierarchy(feature_list, label_list, hierarchy):
    # Generate mutated items
    print("Generating mutated items.")
    mutated_item_list = []
    for i in range(int(len(hierarchy)/4)):
        next_item = hierarchy[-(1 + i)].copy()
        del next_item["accuracy"]
        mutation_index = randint(0, len(hp_keys) - 1)
        next_item[hp_keys[mutation_index]] = HPO[hp_keys[mutation_index]][randint(0, hp_lengths[mutation_index] - 1)]
        mutated_item_list.append(next_item)
    # Generate crossover items
    print("Generating crossover items.")
    crossover_item_list = []
    for i in range(int(len(hierarchy)/4)):
        item_a = hierarchy[-(1 + randint(0, int(len(hierarchy)/4)))]
        item_b = hierarchy[-(1 + randint(0, int(len(hierarchy)/4)))]
        next_item = {}
        for j in range(len(hp_keys)):
            next_item[hp_keys[j]] = [item_a, item_b][randint(0, 1)][hp_keys[j]]
        crossover_item_list.append(next_item)
    # Calculate probabilities of items
    print("Calculating mutated item probabilities.")
    for item in mutated_item_list:
        calculate_accuracy_of_item(feature_list, label_list, item)
    print("Calculating crossover item probabilities.")
    for item in crossover_item_list:
        calculate_accuracy_of_item(feature_list, label_list, item)
    # Swap new mutated and crossover items with the items with the lowest scores
    print("Swapping new items with worst items from the last iteration.")
    for i in range(int(len(hierarchy) / 4)):
        hierarchy[i] = mutated_item_list[i]
    for i in range(int(len(hierarchy) / 4)):
        hierarchy[i + int(len(hierarchy) / 4)] = crossover_item_list[i]
    # Sort the updated list
    quick_sort(hierarchy)
    return hierarchy


# Predict optimal hyper-parameters for an SVM using a genetic algorithm. Stop when the last run gave little extra lift.
def get_optimal_parameters(feature_list, label_list):
    hierarchy = generate_initial_hierarchy(feature_list, label_list)
    print("\n Iteration 0: " + str(hierarchy) + "\n")
    done = False
    counter = 1
    while not done:
        # Calculate the gain in accuracy in the top contenders between the last and the current runs.
        last_hierarchy = hierarchy.copy()
        update_hierarchy(feature_list, label_list, hierarchy)
        print("\n Iteration " + str(counter) + ": " + str(hierarchy) + "\n")
        average_gained_accuracy = 0.0
        for i in range(int(len(hierarchy)/2)):
            average_gained_accuracy += hierarchy[int(len(hierarchy)/2) + i]["accuracy"] - last_hierarchy[int(len(hierarchy)/2) + i]["accuracy"]
        average_gained_accuracy /= int(len(hierarchy)/2)
        print("Gained " + "{:.2%}".format(average_gained_accuracy) + " average accuracy during this last run.")
        # If the gain in accuracy is too low, end the program.
        if average_gained_accuracy <= lift_floor:
            done = True
        counter += 1
    winner = hierarchy[-1]
    print("The winner is: " + str(winner))
    print("Accuracy is " + "{:.2%}".format(winner['accuracy']) + ".")
    del winner["accuracy"]
    return winner


# Quicksort is used to re-order mutant and crossover species into the hierarchy of hyper-parameters.
def quick_sort(my_list):
    print("Sorting the hierarchy of SVM parameters.")
    quick_sort_helper(my_list, 0, len(my_list) - 1)
    print("Sorting complete.")


def quick_sort_helper(my_list, first, last):
    if first < last:
        split_point = partition(my_list, first, last)
        quick_sort_helper(my_list, first, split_point - 1)
        quick_sort_helper(my_list, split_point + 1, last)


def partition(my_list, first, last):
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


# Given a final machine, predict the test data's labels
def predict_test_data(svm, data):
    print("Predicting data using optimal model.")
    for item in data:
        item['label'] = svm.predict(np.array([item['features']]))


# If the best item has from the last run has already been written to JSON, pull it rather than using the default seed.
def get_best_from_JSON(target_location):
    with open(target_location, 'r') as fp:
        data = json.load(fp)
        fp.close()
    return data


def throw_best_in_JSON(optimal_parameters, target_location):
    with open(target_location, 'w') as fp:
        json.dump(optimal_parameters, fp)
        fp.close()


# Run the code.
def main():
    global input_best_item
    # Check whether a dynamic seed location exists
    input_best_item = get_best_from_JSON(prefix + best_item_location)
    # Track start time
    start_time = time.time()
    # Get data from file, separate data into features and labels
    training_data = shared_library.get_data(prefix + training_filename, features_in_use)
    training_feature_list = shared_library.list_entry(training_data, 'features')
    training_label_list = shared_library.list_entry(training_data, 'label')
    testing_data = shared_library.get_data(prefix + testing_filename, features_in_use)
    # Get the optimal SVM hyper-parameters
    optimal_parameters = get_optimal_parameters(training_feature_list, training_label_list)
    # Make an SVM using the parameters
    machine = make_machine_from_params(optimal_parameters)
    # Train the SVM
    machine.fit(np.array(training_feature_list), np.array(training_label_list))
    # Predict using the SVM
    predict_test_data(machine, testing_data)
    # Write test data out to file
    shared_library.write_results(testing_data, prefix + output_filename)
    # Save the optimal parameters in a safe location
    throw_best_in_JSON(optimal_parameters, prefix + best_item_location)
    # Print runtime
    end_time = time.time()
    print("Run took " + str(end_time - start_time) + " seconds.")


main()
