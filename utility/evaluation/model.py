import numpy
import sklearn
import inspect
import xgboost
import lightgbm
import catboost

from sklearn import model_selection
from sklearn import utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

# 分類のモデルを評価
def classifier(
    x
    , y
    , max_iter = 1000
    , early_stopping = 10
    , scorings = ["neg_log_loss"]
    , cv = StratifiedKFold(shuffle = True)
):

    # 結果
    results = {}

    # 目的変数をラベル化
    encoder = LabelEncoder()
    label_train = encoder.fit_transform(y)

    # モデルの候補
    estimators = utils.all_estimators(type_filter = "classifier")

    # モデルを追加
    estimators.append(("xgboost", xgboost.XGBClassifier))
    estimators.append(("lightgbm", lightgbm.LGBMClassifier))
    estimators.append(("catboost", catboost.CatBoostClassifier))

    # 早期打ち切りを設定
    lightgbm.early_stopping(early_stopping)

    for (name, estimator) in estimators:
        try:
            model = None
            signature = inspect.signature(estimator)
            if signature.parameters.get("max_iter") != None:
                if signature.parameters.get("early_stopping") != None:
                    model = estimator(max_iter = max_iter, early_stopping = early_stopping)

                else:
                    model = estimator(max_iter = max_iter)

            elif name == "xgboost":
                model = estimator(objective = "binary:logistic")

            else:
                model = estimator()

            if "score" not in dir(model):
                print(f"{name},not score", end = "")
                continue

            fit_params = None
            if name == "xgboost" or name == "lightgbm" or name == "catboost":
                fit_params = {}
                fit_params["eval_set"] = [(x_train, label_train)]
                fit_params["verbose"] = 0

                if name == "xgboost":
                    fit_params["early_stopping_rounds"] = early_stopping
                    fit_params["eval_metric"] = "logloss"

                elif name == "lightgbm":
                    fit_params["eval_metric"] = "multi_logloss"

            # クロスバリデーションで評価指標を算出
            scores = cross_validate(model
                                    , X = x_train, y = label_train
                                    , scoring = scorings
                                    , cv = cv
                                    , n_jobs = -1
                                    , fit_params = fit_params)
            results[name] = scores

            message = name
            for scoring in scorings:
                target = "test_" + scoring
                message += "," + scoring + ":"
                message += " ".join([format(score, ".6f") for score in scores[target]])
                message += ",平均 {:.6f}".format(scores[target].mean())
            print(message, end = "")

        except Exception as e:
            print(f"{name},エラー:{e}", end = "")

    return results