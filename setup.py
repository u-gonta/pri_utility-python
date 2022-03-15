from setuptools import setup, find_packages

setup(
    name = "utility"
    , version = "0.0.5"
    , description = "Python用のライブラリ"
    , long_description = "共通して使用できる関数群"
    , author = "kazu.kara"
    , license = "MIT"
    , classifiers = ["Development Status :: 1 - Planning"]
    , keywords = "utility"
    , install_requires = [
            "numpy", "pandas", "matplotlib", "sklearn"
            , "xgboost", "lightgbm", "catboost", "optuna"]
    , packages = find_packages()
    )
