import numpy as np

# Set constants
tau = 3.0
features_in_use = 6
prefix = "../datafiles/"
training_filename = 'hw1_trainingset.csv'
training_fraction = 0.8

# Gradient Descent parameters
slope_dif = 0.00001
rate = 0.000001
max_iter = 1000


# Read in data
def get_data(filename):
    data = []
    file = open(filename)
    file.readline()
    for line in file:
        data_points = line.split(",")
        # Get the features as an np vector
        features = np.array([float(x) for x in data_points[0:features_in_use]])
        # Get the label as a -1 or +1 modifier
        label = 2 * int(data_points[-1]) - 1
        data_point_dict = {"label": label, "features": features}
        data.append(data_point_dict)
    file.close()
    return data


def score(data, weight_vector):
    # Get total error
    total_error = 0
    for data_point in data:
        e = 1 - data_point["label"] * tau * (np.dot(weight_vector[:-1], data_point["features"]) + weight_vector[-1])
        if e < 0:  # ensure that good points don't lend extra weight
            e = 0
        total_error += e

    # Get margin
    margin = np.dot(weight_vector, weight_vector)

    # Return score
    return margin + tau * total_error


def slope(start_vector, data):
    global slope_dif
    slope_vector = []
    start_error = score(data, start_vector)
    for i in range(len(start_vector)):
        start_iv_value = start_vector[i]
        end_iv_value = start_iv_value + slope_dif
        end_iv_vector = start_vector.copy()
        end_iv_vector[i] = end_iv_value
        end_error = score(data, end_iv_vector)
        feature_slope = (end_error - start_error) / slope_dif
        slope_vector.append(feature_slope)
    slope_vector = np.array(slope_vector)
    return slope_vector


def get_weights(data):
    # Generate a base vector
    cur_v = []
    for i in range(features_in_use + 1):
        cur_v.append(0.0)
    cur_v = np.array(cur_v)

    # Gradient Descent!
    iter_count = 0
    while iter_count < max_iter:
        prev_v = cur_v
        cur_v = cur_v - rate * slope(prev_v, data)
        iter_count = iter_count + 1
        print("Iteration", iter_count, ", vector value is", cur_v)

    print("Final results: weights are ", cur_v[:-1], ", intercept at ", cur_v[-1], ".")
    return cur_v


def test(data, weight_vector):
    for data_point in data:
        result = np.dot(data_point["features"], weight_vector[:-1]) + weight_vector[-1]
        if result >= 0:
            data_point["prediction"] = 1
        else:
            data_point["prediction"] = -1
    return data


def main():
    # Get the training data
    data = get_data(prefix + training_filename)
    # Train vector weights
    training_data = data[:(int(len(data) * training_fraction))]
    weight_vector = get_weights(training_data)
    # Test vector weights
    testing_data = data[(int(len(data) * training_fraction)):]
    testing_data = test(testing_data, weight_vector)

    # Print results
    correct_count = 0
    for data_item in testing_data:
        if data_item["prediction"] == data_item["label"]:
            correct_count += 1
    print(int(100 * correct_count/len(testing_data)), "% correct. (", correct_count, "/", len(testing_data), ")")


main()
