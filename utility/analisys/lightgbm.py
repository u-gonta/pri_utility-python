# パラメータ
PARAM_CLASSIFIER = {
    "reg_alpha" : [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    , "reg_lambda" : [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    , "num_leaves" : [2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256]
    , "colsample_bytree" : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    , "subsample" : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    , "subsample_freq" : [0, 1, 2, 3, 4, 5, 6, 7]
    , "min_child_samples" : [0, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    }

# スケール
SCALE_CLASSIFIER = {
    "reg_alpha" : "log"
    , "reg_lambda" : "log"
    , "num_leaves" : "linear"
    , "colsample_bytree" : "linear"
    , "subsample" : "linear"
    , "subsample_freq" : "linear"
    , "min_child_samples" : "linear"
    }


