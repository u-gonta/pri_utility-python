import sklearn
import pickle

from sklearn.preprocessing import LabelEncoder

from utility.prepara.data import ID_TITLE as ID
from utility.tuning.data import FILE_NAME as MODEL_FILE_NAME

# ファイル名
FILE_NAME = "submit.csv"

# 予測値を算出してファイルへ出力
def output_file_classifier(
    model
    , x_train
    , y_train
    , id_test
    , x_test
    , invalids = {}
    , id = ID
    , directory = "."
    , model_file_name = MODEL_FILE_NAME
    , save_file_name = FILE_NAME
):

    # 目的変数をラベル化
    encoder = LabelEncoder()
    label = encoder.fit_transform(y_train)

    # 説明変数を絞り込み
    x_train = x_train.drop(columns = invalids)
    x_test = x_test.drop(columns = invalids)

    # モデルを読み込み
    model = pickle.load(open(directory + "\\" + model_file_name, "rb"))

    # モデルを学習
    model.fit(x_train, label)

    # モデルをテスト
    predict = model.predict(x_test)

    # 予測値を結合
    result = id_test.copy()
    result["class"] = encoder.inverse_transform(predict)

    # 予測値を保存
    result[[id, "class"]].to_csv(directory + "\\" + save_file_name, header = False, index = False)

    return predict

