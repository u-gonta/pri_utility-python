# パラメータ
PARAM_CLASSIFIER = {
    "learning_rate" : [0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0]
    , "subsample" : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    , "min_weight_fraction_leaf" : [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    , "max_depth" : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    , "min_impurity_decrease" : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    , "tol" : [0, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0]
    , "ccp_alpha" : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    }

# スケール
SCALE_CLASSIFIER = {
    "learning_rate" : "log"
    , "subsample" : "linear"
    , "min_weight_fraction_leaf" : "linear"
    , "max_depth" : "linear"
    , "min_impurity_decrease" : "linear"
    , "tol" : "linear"
    , "ccp_alpha" : "log"
    }

