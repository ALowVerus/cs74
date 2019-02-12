from sklearn.svm import SVC
import numpy as np

# Set constants
features_in_use = 6
prefix = "../datafiles/"
training_filename = 'hw3_training_data.csv'
testing_filename = 'hw3_test_data.csv'
output_filename = 'hw3_output.csv'
number_of_samples = 10


# Read in data
def get_data(filename):
    global top_file_line
    feature_list_list = []
    label_list = []
    file = open(filename)
    top_file_line = file.readline()
    for line in file:
        data_points = line.split(",")
        # Get the features as an np vector
        feature_list = [float(x) for x in data_points[0:features_in_use]]
        # Get the label as a -1 or +1 modifier
        label = int(data_points[-1])
        feature_list_list.append(feature_list)
        label_list.append(label)
    file.close()
    return {'features': feature_list_list, 'labels': label_list}


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
        for i in range(len(testing_features)):
            predicted_label = svm.predict([testing_features[i]])
            actual_label = testing_labels[i]
            if predicted_label == actual_label:
                correct_count += 1
        correct_decimal = correct_count / len(testing_features)
        print(str(float(int(correct_decimal * 1000)) / 10) + "%")
        accuracy += correct_decimal
    accuracy /= sample_count
    print("Net accuracy is " + str(accuracy))


def predict_test_data(svm):
    global top_file_line
    test_features = get_data(prefix + testing_filename)['features']
    output_file = open(prefix + output_filename, 'w')
    output_file.write(top_file_line[:-1] + ",Label\n")
    for test_feature in test_features:
        output_file.write(str(test_feature)[1:-1] + ", " + str(svm.predict([test_feature]))[1:-1] + "\n")


def main():
    machine = SVC(kernel='sigmoid', max_iter=100000, gamma='auto')
    data = get_data(prefix + training_filename)
    feature_list = data['features']
    label_list = data['labels']
    get_accuracy(machine, feature_list, label_list, number_of_samples)
    predict_test_data(machine)


main()
