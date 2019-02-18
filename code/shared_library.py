import time

# Read in data
def get_data(filename, features_in_use):
    global top_file_line
    file = open(filename)
    top_file_line = file.readline()
    data = []
    for line in file:
        data_points = line.split(",")
        # Get the features as an np vector
        features = [float(x) for x in data_points[0:features_in_use]]
        # Get the label as a -1 or +1 modifier
        label = int(data_points[-1])
        data.append({"label": label, "features": features})
    file.close()
    return data


# Run through the test set of data, get projections
def predict(classifier, data):
    print("Starting predictions.")
    for item in data:
        features = item['features']
        item['label'] = classifier.predict(features)
    print("Done predicting.")


# Write results to output location
def write_results(data, output_location):
    output_file = open(output_location, "w")
    output_file.write("Feature_1,Feature_2,Feature_3,Feature_4,Feature_5,Feature_6,Label\n")
    for item in data:
        output_file.write(','.join(map(str, item['features'])) + "," + str(item['label']) + "\n")
    output_file.close()


# Run the code.
def main(Model, best_item_loc=False, training_set_loc=False, testing_set_loc=False, results_loc=False,
         iv_count=False, validation_count=3):
    start_time = time.time()
    # Grab data
    training_data = get_data(training_set_loc, iv_count)
    testing_data = get_data(testing_set_loc, iv_count)
    # Initialize and train the classifier
    classifier = Model(training_data, validating=True, best_item_loc=best_item_loc, validation_count=validation_count)
    # Predict end value
    predict(classifier, testing_data)
    # Print end values to document
    write_results(testing_data, results_loc)
    print("--- %s seconds ---" % (time.time() - start_time))