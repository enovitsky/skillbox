import dill
import glob
import json
import os
import pandas as pd

from datetime import datetime

# Укажем путь к файлам проекта:
# -> $PROJECT_PATH при запуске в Airflow
# -> иначе - текущая директория при локальном запуске

# path = os.environ.get('PROJECT_PATH', '.')
path = '/Users/enovitsky/airflow_hw'

# loading training data
df_train = pd.read_csv('../data/train/homework.csv')

# loading testing data
test_json_files = glob.glob('../data/test/*.json')
df_test = pd.DataFrame()
for file in test_json_files:
    with open(file, 'r') as json_file:
        json_dict = json.load(json_file)

        for key in json_dict.keys():
            json_dict[key] = [json_dict[key]]

        json_df = pd.DataFrame.from_dict(json_dict)
        df_test = pd.concat([df_test, json_df], axis=0, ignore_index=True)


def get_prediction(df):
    with open(f'{path}/data/models/cars_pipe_202310061532.pkl', 'rb') as model_file:
        model = dill.load(model_file)

    df['prediction'] = model.predict(df)
    df = df[['id', 'prediction']]
    return df


def predict():
    df_train_predictions = get_prediction(df_train)
    df_test_predictions = get_prediction(df_test)
    df_predictions = pd.concat([df_train_predictions, df_test_predictions], axis=0, ignore_index=True)
    df_predictions.to_csv(f'{path}/data/predictions/predictions_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)


if __name__ == '__main__':
    predict()
