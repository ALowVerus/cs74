# Validate a prediction's accuracy using a given number of subsets of data
def n_fold_validate(accuracy_function, data_set, sample_count):
    subset_list = []
    list_size = len(data_set)
    for i in range(sample_count):
        subset_list.append(data_set[int(i * list_size / sample_count):int((i + 1) * list_size / sample_count)])
    accuracy = 0.0
    for i in range(sample_count):
        accuracy += accuracy_function(subset_list[i])
    accuracy /= sample_count
    return accuracy


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


# Get a list of entries of a data type
def list_entry(data, key):
    entry_list = []
    for datum in data:
        entry_list.append(datum[key])
    return entry_list


def write_results(data, output_location):
    output_file = open(output_location, "w")
    output_file.write("Feature_1,Feature_2,Feature_3,Feature_4,Feature_5,Feature_6,Label\n")
    for item in data:
        output_file.write(','.join(map(str, item['features'])) + "," + str(item['label']) + "\n")
    output_file.close()
