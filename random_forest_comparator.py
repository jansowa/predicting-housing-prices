from time import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from numpy import arange

n_estimators_default = 10
n_estimators_range_default = range(1, 30)
criterion_default = 'gini'
max_depth_default = None
min_samples_split_default = 2
min_samples_leaf_default = 1
min_samples_leaf_range_default = range(1, 202, 10)
min_weight_fraction_leaf_default = 0.0
min_weight_fraction_leaf_range_default = arange(0.0, 0.51, 0.1)
max_features_default = 'auto'
max_leaf_nodes_default = None
min_impurity_decrease_default = 0.0
min_impurity_split_default = None
bootstrap_default = True
oob_score_default = False
n_jobs_default = None
random_state_default = None
verbose_default = 0
warm_start_default = False
class_weight_default = None

def calculate_best_accuracy_for_random_forest_classifier(features_train, labels_train, features_test, labels_test,
                                                    n_estimators_range=n_estimators_range_default, criterion=criterion_default, max_depth = max_depth_default, min_samples_split = min_samples_split_default,
                                                    min_samples_leaf_range=min_samples_leaf_range_default, min_weight_fraction_leaf_range=min_weight_fraction_leaf_range_default, max_feature=max_features_default,
                                                    mex_leaf_nodex = max_leaf_nodes_default, min_impurity_decrease=min_impurity_decrease_default, min_impurity_split=min_impurity_split_default,
                                                         print_parameters=False):
    max_accuracy = 0
    n_estimators_with_max_accuracy = 0
    min_samples_leaf_with_max_accuracy = 0
    min_weight_fraction_leaf_with_max_accuracy = 0.0

    executing_tests_time = time()
    for n_estimators_number in n_estimators_range:
        print("Calculating for n_estimators: " + str(n_estimators_number))
        for min_sapmles_leaf_number in min_samples_leaf_range:
            for min_weight_fraction_leaf in min_weight_fraction_leaf_range:
                accuracy = calculate_accuracy_for_random_forest_classifier(features_train, labels_train, features_test, labels_test,
                                                                           n_estimators_number, criterion, max_depth, min_samples_split, min_sapmles_leaf_number,
                                                                           min_weight_fraction_leaf, max_feature, mex_leaf_nodex, min_impurity_decrease, min_impurity_split)
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    n_estimators_with_max_accuracy = n_estimators_number
                    min_samples_leaf_with_max_accuracy = min_sapmles_leaf_number
                    min_weight_fraction_leaf_with_max_accuracy = min_weight_fraction_leaf

    executing_tests_time = time() - executing_tests_time

    if print_parameters:
        print("Executing tests time: " + str(executing_tests_time))

    print("Best RandomForestClassifier accuracy: " + str(max_accuracy) + " for n_estimators = " + str(n_estimators_with_max_accuracy) + ", min_samples_leaf = " + str(min_samples_leaf_with_max_accuracy)
          + ", min_weight_fraction_leaf = " + str(min_weight_fraction_leaf_with_max_accuracy))



def calculate_accuracy_for_random_forest_classifier(features_train, labels_train, features_test, labels_test,
                                                    n_estimators=n_estimators_default, criterion=criterion_default, max_depth = max_depth_default, min_samples_split = min_samples_split_default,
                                                    min_samples_leaf=min_samples_leaf_default, min_weight_fraction_leaf=min_weight_fraction_leaf_default, max_feature=max_features_default,
                                                    mex_leaf_nodex = max_leaf_nodes_default, min_impurity_decrease=min_impurity_decrease_default, min_impurity_split=min_impurity_split_default,
                                                    print_time=False, print_accuracy=False):
    classifier = RandomForestClassifier(n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_feature,
                                        mex_leaf_nodex, min_impurity_decrease, min_impurity_split)

    training_time = time()
    classifier.fit(features_train, labels_train)
    if print_time:
        print("Time of training model: " + str(time() - training_time))

    predicted_labels = classifier.predict(features_test)

    accuracy = accuracy_score(labels_test, predicted_labels)

    if print_accuracy:
        print("Accuracy of predicted labels: " + str(accuracy))

    return accuracy