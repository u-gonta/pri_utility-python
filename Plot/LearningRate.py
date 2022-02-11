import numpy
import pyplotlib

# 学習率を描画
def Draw(train = [], valid = [], range = 0, xscale = "log", xlabel = "", ylabel = ""):
    # 学習データに対するスコアの平均±標準偏差を算出
    train_mean = numpy.mean(train, axis = 1)
    train_std  = numpy.std(train, axis = 1)
    train_center = train_mean
    train_high = train_mean + train_std
    train_low = train_mean - train_std

    # テストデータに対するスコアの平均±標準偏差を算出
    valid_mean = numpy.mean(valid, axis = 1)
    valid_std  = numpy.std(valid, axis = 1)
    valid_center = valid_mean
    valid_high = valid_mean + valid_std
    valid_low = valid_mean - valid_std
    
    # 描画エリアを閉じる
    matplotlib.pyplot.close()

    # 学習データをプロット
    matplotlib.pyplot.plot(range, train_center, color = "blue", marker = "o", markersize = 5, label = "training score")
    matplotlib.pyplot.fill_between(range, train_high, train_low, alpha = 0.15, color = "blue")
    # 検証データをプロット
    matplotlib.pyplot.plot(range, valid_center, color = "green", linestyle = "--", marker = "o", markersize = 5, label = "validation score")
    matplotlib.pyplot.fill_between(range, valid_high, valid_low, alpha = 0.15, color = "green")
    if xscale:
        # スケールをxscaleに合わせて変更
        matplotlib.pyplot.xscale(xscale)
    # 軸ラベルおよび凡例の指定
    matplotlib.pyplot.xlabel(xlabel)
    matplotlib.pyplot.ylabel(ylabel)
    # 凡例
    matplotlib.pyplot.legend(loc = "lower right")

