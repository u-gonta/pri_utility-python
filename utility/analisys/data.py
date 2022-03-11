import numpy
import pandas
import sklearn
import matplotlib

from pandas import Series
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot

from utility.plot.learning_rate import draw as draw_learning_rate

# データを分析
def classifier(
    model
    , x
    , y
    , cv_params
    , param_scales
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
            draw_learning_rate(train_score, valid_score
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
