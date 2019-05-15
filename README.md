# Recommender-System

The project is about creating a basic recommending system using jester dataset. The system is evaluated based on RMSE.

There a few functions used in the process. In the following lines I will explain how to use those functions and what is the result.


“”” best_parameter_search.py ”””

This script is used to determine the best parameter combination for given algorithm. There are 2 different algorithms available for testing, SVD and BaselineOnly.

Parameters used in functions are for the sake of UI simplicity changed directly in the code:

“param_grid = {'n_epochs': [12,13], 'lr_all': [0.0013, 0.0015], 'reg_all': [0.05,0.06]}” (“” should not be included in code)

The script is used with arguments (3 options):
python3 best_parameter_search.py SVD
python3 best_parameter_search.py BaselineOnly als
python3 best_parameter_search.py BaselineOnly sgd

The output of the script is best RMSE and parameters used to determine it.



“”” alg_initial_test.py ”””

The script is used to determine basic RMSE of specific algorithm.

Available algorithms are SVD, SVDpp, SlopeOne, NMF, NormalPredictor, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering.

The script is used with arguments:
python3 alg_initial_test.py SVD KNNBasic

The output of the script is RMSE, fit_time, test_time for tested algorithms.



“”” validate_model.py “””

The script evaluates the selected model using test data gathered from “train_test_split” function. Best models from “best_parameter_search.py” script are imported and used on test data.

The script is used with one argument:
python3 validate_model.py SVD

The output of the script is RMSE of a tested model.

© 2019 jakobpremrn
