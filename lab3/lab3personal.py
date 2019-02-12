import numpy as np

# Set constants
features_in_use = 6
prefix = "../datafiles/"
training_filename = 'hw3_training_data.csv'
training_fraction = 0.8


class SVM:
    def __init__(self, max_iter=1000, c=1.0, h=0.0000000001, rate=0.001, tau=5.0, epsilon=0.001):
        self.max_iter = max_iter
        self.weight_vector = ""
        self.c = c
        self.h = h
        self.rate = rate
        self.tau = tau
        self.epsilon = epsilon

    def train_weights(self, data):
        # Generate a base vector
        cur_v = np.zeros(len(data[0]["features"]))

        # Gradient Descent!
        iter_count = 0
        while True:
            prev_v = cur_v
            cur_v = cur_v - self.rate * self.slope(prev_v, data)
            iter_count = iter_count + 1
            print("Iteration", iter_count, ", vector value is", list(cur_v))

            # Check for completion
            vector_step = np.linalg.norm(cur_v - prev_v)
            if vector_step < self.epsilon:
                print("Reached adequate precision.")
                break

            # Check for overrun
            if iter_count >= self.max_iter:
                print("Reached max iterations.")
                break

        print("Final results: weights are ", cur_v, ", intercept at ", self.c, ".")
        self.weight_vector = cur_v

    def score(self, data, weight_vector):
        # Get total error
        total_error = 0
        for data_point in data:
            # t * (dot(w, point) + c) ≥ 1 − e
            e = 1 - data_point["label"] * self.tau * (np.dot(weight_vector, data_point["features"]) + self.c)
            if e < 0:  # ensure that good points don't lend extra weight
                e = 0
            total_error += e

        # Get margin
        margin = np.dot(weight_vector, weight_vector)

        # Return score
        return margin + self.tau * total_error

    def slope(self, start_vector, data):
        slope_vector = []
        start_error = self.score(data, start_vector)
        for i in range(len(start_vector)):
            start_iv_value = start_vector[i]
            end_iv_value = start_iv_value + self.h
            end_iv_vector = start_vector.copy()
            end_iv_vector[i] = end_iv_value
            end_error = self.score(data, end_iv_vector)
            feature_slope = (end_error - start_error) / self.h
            slope_vector.append(feature_slope)
        slope_vector = np.array(slope_vector)
        return slope_vector

    def test(self, data):
        # Set prediction flags
        for data_point in data:
            data_point["prediction"] = np.sign(
                np.dot(data_point["features"], self.weight_vector) + self.c
            ).astype(int)
        # Print results
        correct_count = 0
        for data_item in data:
            if data_item["prediction"] == data_item["label"]:
                correct_count += 1
        print(int(100 * correct_count / len(data)), "% correct. (", correct_count, "/", len(data), ")")


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


def main():
    # Make an SVM
    svm = SVM()
    # Get the training data
    data = get_data(prefix + training_filename)
    # Train vector weights
    training_data = data[:(int(len(data) * training_fraction))]
    svm.train_weights(training_data)
    # Test vector weights
    testing_data = data[(int(len(data) * training_fraction)):]
    svm.test(testing_data)


main()
