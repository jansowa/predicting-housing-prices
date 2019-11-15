from time import time

from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.tree import DecisionTreeRegressor
from numpy import arange

base_estimator_default = None
decision_tree_max_depth_range_default = range(1, 10)
n_estimators_default = 50
learning_rate_default = 1.0
learning_rate_range_default = arange(0.5, 2.1, 0.1)
random_state_default = 1
n_estimators_range_default = range(1, 10)


def calculate_min_error_for_adaboost_regressor(features_train, labels_train, features_test, labels_test,
                                               decision_tree_max_depth_range=decision_tree_max_depth_range_default,
                                               n_estimators_range=n_estimators_range_default,
                                               learning_rate_range=learning_rate_range_default,
                                               random_state=random_state_default,
                                               print_parameters=False):
    min_error = -1
    n_estimators_with_min_error = 0
    decision_tree_max_depth_with_min_error = 0
    learning_rate_with_min_error = 0

    executing_tests_time = time()
    for n_estimators_number in n_estimators_range:
        print("Calculating error for adaboost regressor with n_estimators = " + str(n_estimators_number))
        for decision_tree_max_depth in decision_tree_max_depth_range:
            for learning_rate in learning_rate_range:
                error = calculate_error_for_adaboost_regressor(features_train, labels_train, features_test,
                                                               labels_test,
                                                               DecisionTreeRegressor(
                                                                          max_depth=decision_tree_max_depth),
                                                               n_estimators_number, learning_rate,
                                                               random_state)
                if (min_error == -1) or (error < min_error):
                    min_error = error
                    n_estimators_with_min_error = n_estimators_number
                    decision_tree_max_depth_with_min_error = decision_tree_max_depth
                    learning_rate_with_min_error = learning_rate

    executing_tests_time = time() - executing_tests_time

    if print_parameters:
        parameters_string = "Calculating for parameters: \n"
        parameters_string += "max depth of decision tree range: " + str(decision_tree_max_depth_range[0]) + " - " + str(
            decision_tree_max_depth_range[len(decision_tree_max_depth_range) - 1])
        parameters_string += ", n_estimator range: " + str(n_estimators_range[0]) + " - " + str(
            n_estimators_range[len(n_estimators_range) - 1])
        parameters_string += ", learning_rate_range: " + str(learning_rate_range[0]) + " - " + str(
            learning_rate_range[len(learning_rate_range) - 1])
        parameters_string += ", random_state: " + str(random_state)
        print(parameters_string)
        print("Executing test time: " + str(executing_tests_time))

    print("Minimum AdaBoostRegressor error: " + str(min_error) + " for n_estimators = " + str(
        n_estimators_with_min_error) + ", desion tree max depth = " + str(decision_tree_max_depth_with_min_error) +
          ", learning_rate = " + str(learning_rate_with_min_error))

    return min_error


def calculate_error_for_adaboost_regressor(features_train, labels_train, features_test, labels_test,
                                           base_estimator=base_estimator_default, n_estimators=n_estimators_default,
                                           learning_rate=learning_rate_default,
                                           random_state=random_state_default,
                                           print_time=False, print_parameters=False, print_error=False):
    regressor = AdaBoostRegressor(base_estimator=base_estimator, n_estimators=n_estimators,
                                    learning_rate=learning_rate, random_state=random_state)

    training_time = time()
    regressor.fit(features_train, labels_train)
    if print_time:
        print("Time of training model: " + str(time() - training_time))

    predicted_labels = regressor.predict(features_test)

    if print_parameters:
        parameters_string = "AdaBoost regressor for parameters:\n"
        parameters_string += "base_estimator: " + str(base_estimator)
        parameters_string += ", n_estimators: " + str(n_estimators)
        parameters_string += ", learning_rate: " + str(learning_rate)
        parameters_string += ", random_state: " + str(random_state)
        print(parameters_string)

    error = mean_squared_log_error(labels_test, predicted_labels)

    if print_error:
        print("Error of predicted labels: " + str(error))

    return error
