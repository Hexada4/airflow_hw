import json
import logging
import os
from datetime import datetime

import dill
import pandas as pd
from pandas import DataFrame

path = os.environ.get('PROJECT_PATH', '..')
model_name = os.environ.get('MODEL_FILE', 'cars_pipe.pkl')


def get_path():
    return f'{path}/data/models/{model_name}'


def load_model(path):
    with open(path, 'rb') as f:
        logging.info(f'loading model from path: {path}')
        model = dill.load(f)
        return model


def read_df(path):
    with open(path, 'rb') as j:
        json_record = json.load(j)
        logging.info(json_record)
        res = pd.DataFrame.from_dict([json_record])
    return res


def read_from_folder(path) -> [DataFrame]:
    # read dir and for each json create dataframe
    files = os.listdir(path)
    logging.info(f'to_predict: {files}')
    forms = [read_df(f'{path}/{f}') for f in files]
    return forms


def predict_for_all(model, path):
    forms = read_from_folder(path)
    predictions = [pd.DataFrame.from_dict([{'predicted': model.predict(f)[0]}]) for f in forms]
    logging.info(f'predicted: {predictions}')
    res = pd.concat(predictions, ignore_index=True)
    return res


def save_result(df: DataFrame, path):
    df.to_csv(path, index_label='index')



def predict():
    model_path = get_path()
    model = load_model(model_path)
    form_path = f'{path}/data/test'
    predictions = predict_for_all(model, form_path)
    res_path = f'{path}/data/predictions'
    save_result(predictions, f'{res_path}/prediction_{datetime.now().strftime("%Y%m%d%H%M")}.csv')



if __name__ == '__main__':
    predict()