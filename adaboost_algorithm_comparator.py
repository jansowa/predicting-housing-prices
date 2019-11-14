from time import time

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from numpy import arange

base_estimator_default = None
decision_tree_max_depth_range_default = range(1, 10)
n_estimators_default = 50
learning_rate_default = 1.0
learning_rate_range_default = arange(0.5, 2.1, 0.1)
algorithm_default = "SAMME.R"
random_state_default = None
n_estimators_range_default = range(1, 10)


def calculate_best_accuracy_for_adaboost_classifier(features_train, labels_train, features_test, labels_test,
                                                    decision_tree_max_depth_range=decision_tree_max_depth_range_default,
                                                    n_estimators_range=n_estimators_range_default,
                                                    learning_rate_range=learning_rate_range_default, algorithm=algorithm_default,
                                                    random_state=random_state_default,
                                                    print_parameters=False):
    max_accuracy = 0
    n_estimators_with_max_accuracy = 0
    decision_tree_max_depth_with_max_accuracy = 0
    learning_rate_with_max_accuracy = 0

    executing_tests_time = time()
    for n_estimators_number in n_estimators_range:
        for decision_tree_max_depth in decision_tree_max_depth_range:
            for learning_rate in learning_rate_range:
                accuracy = calculate_accuracy_for_adaboost_classifier(features_train, labels_train, features_test,
                                                                      labels_test,
                                                                      DecisionTreeClassifier(
                                                                          max_depth=decision_tree_max_depth),
                                                                      n_estimators_number, learning_rate, algorithm,
                                                                      random_state)
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    n_estimators_with_max_accuracy = n_estimators_number
                    decision_tree_max_depth_with_max_accuracy = decision_tree_max_depth
                    learning_rate_with_max_accuracy = learning_rate

    executing_tests_time = time() - executing_tests_time

    if print_parameters:
        parameters_string = "Calculating for parameters: \n"
        parameters_string += "max depth of decision tree range: " + str(decision_tree_max_depth_range[0]) + " - " + str(
            decision_tree_max_depth_range[len(decision_tree_max_depth_range) - 1])
        parameters_string += ", n_estimator range: " + str(n_estimators_range[0]) + " - " + str(
            n_estimators_range[len(n_estimators_range) - 1])
        parameters_string += ", learning_rate_range: " + str(learning_rate_range[0]) + " - " + str(
            learning_rate_range[len(learning_rate_range) - 1])
        parameters_string += ", algorithm: " + str(algorithm)
        parameters_string += ", random_state: " + str(random_state)
        print(parameters_string)
        print("Executing test time: " + str(executing_tests_time))

    print("Best AdaBoostClassifier accuracy: " + str(max_accuracy) + " for n_estimators = " + str(
        n_estimators_with_max_accuracy) + ", desion tree max depth = " + str(decision_tree_max_depth_with_max_accuracy) +
          ", learning_rate = " + str(learning_rate_with_max_accuracy))


def calculate_accuracy_for_adaboost_classifier(features_train, labels_train, features_test, labels_test,
                                               base_estimator=base_estimator_default, n_estimators=n_estimators_default,
                                               learning_rate=learning_rate_default, algorithm=algorithm_default,
                                               random_state=random_state_default,
                                               print_time=False, print_parameters=False, print_accuracy=False):
    classifier = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators,
                                    learning_rate=learning_rate, algorithm=algorithm, random_state=random_state)

    training_time = time()
    classifier.fit(features_train, labels_train)
    if print_time:
        print("Time of training model: " + str(time() - training_time))

    predicted_labels = classifier.predict(features_test)

    if print_parameters:
        parameters_string = "AdaBoost classifier for parameters:\n"
        parameters_string += "base_estimator: " + str(base_estimator)
        parameters_string += ", n_estimators: " + str(n_estimators)
        parameters_string += ", learning_rate: " + str(learning_rate)
        parameters_string += ", algorithm: " + str(algorithm)
        parameters_string += ", random_state: " + str(random_state)
        print(parameters_string)

    accuracy = accuracy_score(labels_test, predicted_labels)

    if print_accuracy:
        print("Accuracy of predicted labels: " + str(accuracy))

    return accuracy
