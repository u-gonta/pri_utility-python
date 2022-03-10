import pandas
import numpy
import sklearn

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from pandas import Series
from matplotlib import pyplot

import utility

# GradientBoostingClassifierのパラメータ
PARAM_GBC = {
    "learning_rate" : [0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0]
    , "subsample" : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    , "min_weight_fraction_leaf" : [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    , "max_depth" : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    , "min_impurity_decrease" : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    , "tol" : [0, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0]
    , "ccp_alpha" : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    }
# GradientBoostingClassifierのスケール
SCALE_GBC = {
    "learning_rate" : "log"
    , "subsample" : "linear"
    , "min_weight_fraction_leaf" : "linear"
    , "max_depth" : "linear"
    , "min_impurity_decrease" : "linear"
    , "tol" : "linear"
    , "ccp_alpha" : "log"
    }

# データを分析
def classifier(
    model
    , x
    , y
    , cv_params = {}
    , param_scales = {}
    , scorings = ["neg_log_loss"]
    , cv = StratifiedKFold(shuffle = True)
    , directory = "."
):

    # 目的変数をラベル化
    encoder = LabelEncoder()
    label = encoder.fit_transform(y)

    for i, (name, range) in enumerate(cv_params.items()):
        for scoring in scorings:
            # 検証曲線を取得
            train_score, valid_score = validation_curve(model, X = x, y = label
                                                        , param_name = name
                                                        , param_range = range
                                                        , scoring = scoring
                                                        , cv = cv)

            # 学習率のグラフを描画
            utility.plot.draw_learning_rate(train_score, valid_score
                                            , range = range, x_scale = param_scales[name]
                                            , x_label = name, y_label = scoring)
            # 学習率のグラフを画像で保存
            path = directory + "\\analisys_" + scoring + "_" + str(i + 1).zfill(2) + "_" + name + ".png"
            pyplot.savefig(path)

    # クロスバリデーションで評価指標を算出
    scores = cross_validate(model, x, label, scoring = scorings, cv = cv)
    for scoring in scorings:
        target = "test_" + scoring
        message =  scoring + ",スコア:"
        message += " ".join([format(score, ".6f") for score in  scores[target]])
        message += ",平均:{:.6f}".format(scores[target].mean())
        print(message, end = "")

    # 探索
    model.fit(x, label)

    # 描画エリアを閉じる
    pyplot.close()

    # 特徴量の重要度を取得
    importances = Series(data = model.feature_importances_, index = x.columns)
    for importance in importances.items():
        print(importance[0] + ",{:.6f}".format(importance[1]), end = "")

    return scores, importances