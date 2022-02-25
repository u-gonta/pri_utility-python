import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn import utils

# 分類のモデルを評価
def model_classifier(
    x, y
):

    # 目的変数をラベル化
    encoder = LabelEncoder()
    label_train = encoder.fit_transform(y)

    # モデルの候補
    estimators = utils.all_estimators(type_filter = "classifier")

    # モデルを追加
    estimators.append(("xgboost", xgboost.XGBClassifier))
    estimators.append(("lightgbm", lightgbm.LGBMClassifier))
    estimators.append(("catboost", catboost.CatBoostCalssifier))

    for (name, estimator) in estimators:
        try:
            model = None
            if name == "HistGradientBoostingClassifier" or name == "GaussianProcessClassifier" or name == "MultinomialNB":
                model = estimator(max_iter = 200)
            elif name == "xgboost":
                model = estimator(objective = "binary:logistic")
            else:
                model = estimator()

            if "score" not in dir(model):
                print(f"{name},not score")
                continue

            fit_params = None
            if name == "xgboost" or name == "lightgbm" or name == "catboost":
                fit_params = {}
                fit_params["eval_set"] = [(x_train, label_train)]

                if name == "xgboost":
                    fit_params["early_stopping_rounds"] = 10
                    fit_params["eval_metric"] = "logloss"

                elif name == "lightgbm":
                    fit_params["eval_metric"] = "multi_logloss"

            # クロスバリデーションで評価指標を算出
            scores = sklearn.model_selection.cross_val_score(model
                                                             , X = x_train, y = label_train
                                                             , scoring = _scoring
                                                             , cv = _cv
                                                             , n_jobs = -1
                                                             , fit_params = fit_params)
            score = numpy.mean(scores)
            print(f"{name},スコア:{scores},平均:{score}")
            if target:
                if maximum < score:
                    target = name
                    maximum= score
            else:
                target = name
                maximum = score

        except Exception as e:
            print(f"{name},エラー:{e}")