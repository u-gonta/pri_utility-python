import numpy
import pandas
import sklearn
import matplotlib
import optuna
import pickle

from pandas import Series
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve
from matplotlib import pyplot

from utility.plot.learning_rate import draw as draw_learning_rate

# ファイル名
FILE_NAME = "tuning_model.pickle"

# モデル
_model = None

# 説明変数
_x = None

# 目的変数
_y = None

# 評価の方法
_scoring = None

# クロスバリデーションの設定
_cv = None

# モデルのパラメータ
_model_params = None

# 評価指標を算出(Optuna)
def optuna_objective(trial):
    # モデルのパラメータを更新
    global _model_params
    model_params = {}
    for key, values in _model_params.items():
        # パラメータの名称
        item = key
        # パラメータの下限値
        lower = values[0]
        # パラメータの上限値
        upper = values[1]
        suggest = None
        if type(lower) is int or type(upper) is int:
            # int型
            suggest = trial.suggest_int(item, lower, upper)
        elif type(lower)is float or type(upper) is float:
            # float型
            suggest = trial.suggest_float(item, lower, upper)
        else:
            # 型不明
            continue

        # パラメータを登録
        model_params[item] = suggest

    # モデルにパラメータを適応
    global _model
    _model.set_params(**model_params)

    # クロスバリデーション
    global _x
    global _y
    global _scoring
    global _cv
    scores = cross_validate(_model, _x, _y, cv = _cv, scoring = _scoring)

    return scores["test_score"].mean()

# チューニング(Optuna)
def optuna_classifier(
    model
    , x
    , y
    , model_params
    , fit_params
    , scoring = "neg_log_loss"
    , cv = StratifiedKFold(shuffle = True)
    , direction = "maximize"
    , seed = 42
    , trials = 600
    , directory = "."
    , file_name = FILE_NAME
):

    # 目的変数をラベル化
    encoder = LabelEncoder()
    label = encoder.fit_transform(y)

    # モデルを更新
    global _model
    _model = model

    # 説明変数を更新
    global _x
    _x = x

    # 目的変数を更新
    global _y
    _y = label

    # 評価の方法を更新
    global _scoring
    _scoring = scoring

    # クロスバリデーションの設定を更新
    global _cv
    _cv = cv

    # モデルのパラメータを更新
    global _model_params
    _model_params = model_params

    # ベイズ最適化を実行
    study = optuna.create_study(direction = direction, sampler = optuna.samplers.TPESampler(seed = seed))
    study.optimize(optuna_objective, n_trials = trials)

    # 最適のパラメータを表示
    best_param = study.best_trial.params
    best_score = study.best_trial.values
    for key, value in study.best_trial.params.items():
        print("{}:{:.6f}".format(key, value), end = "")
    message = "スコア:"
    message += " ".join([format(score, ".6f") for score in best_score])
    print(message, end = "")

    # 最適パラメータをモデルにセット
    _model.set_params(**best_param)

    # 学習曲線の取得
    train_sizes, train_scores, valid_scores = learning_curve(_model, _x, _y
                                                             , train_sizes = numpy.linspace(0.1, 1.0, 10)
                                                             , scoring = _scoring
                                                             , cv = _cv
                                                             , fit_params = fit_params)

    # 学習曲線を描画
    draw_learning_rate(train_scores, valid_scores, range = train_sizes, x_scale = ""
                       , x_label = "train_samples", y_label = _scoring)

    # 学習曲線のグラフを画像で保存
    path = directory + "\\learning_" + scoring + ".png"
    pyplot.savefig(path)

    # 探索
    _model.fit(_x, _y)

    # 特徴量の重要度を取得
    importances = model.feature_importances_
    for (index, name) in enumerate(x.columns):
        print("{},{:.6f}".format(name, importances[index]), end = "")

    # モデルを保存
    pickle.dump(_model, open(directory + "\\" + file_name, "wb"))

    return importances