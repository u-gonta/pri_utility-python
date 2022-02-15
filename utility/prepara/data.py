import pandas

# 識別IDのタイトル
ID_TITLE = "id"

# ファイルを読み込みデータに変換
def load(
    training
    , test
    , separate = "\t"
    , identification = 0
    , purpose = ""
):

    # ファイル読み込み
    train = pandas.read_csv(training, sep = separate)
    test = pandas.read_csv(test, sep = separate)

    # 識別IDを取得
    id_train = train.iloc[:, [identification]]
    id_test = test.iloc[:, [identification]]

    # 識別IDのヘッダを更新
    id_train.columns = [ID_TITLE]
    id_test.columns = [ID_TITLE]

    # 識別IDと目的変数を削除
    x_train = train.drop(columns = [train.columns[identification], purpose])
    x_test = test.drop(columns = [test.columns[identification]])

    # 目的変数を取得
    y_train = train[purpose]

    return id_train, x_train, y_train, id_test, x_test