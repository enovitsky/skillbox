import dill
import glob
import json
import pandas as pd

#loading training data
df_train = pd.read_csv('/Users/enovitsky/airflow_hw/data/train/homework.csv')

#loading testing data
test_json_files = glob.glob('/Users/enovitsky/airflow_hw/data/test/*.json')
df_test = pd.DataFrame()
for file in test_json_files:
    with open(file, 'r') as json_file:
        json_dict = json.load(json_file)

        for key in json_dict.keys():
            json_dict[key] = [json_dict[key]]

        json_df = pd.DataFrame.from_dict(json_dict)
        df_test = pd.concat([df_test, json_df], axis=0, ignore_index=True)


def predict():
    with open('/Users/enovitsky/airflow_hw/data/models/cars_pipe_202310031737.pkl', 'rb') as model_file:
        model = dill.load(model_file)

    train_predictions = pd.Series(model.predict(df_train))
    test_predictions = pd.Series(model.predict(df_test))

    df_predictions = pd.concat([train_predictions, test_predictions], axis=0, ignore_index=True)
    df_predictions.to_csv('/Users/enovitsky/airflow_hw/data/predictions/predictions.csv')


if __name__ == '__main__':
    predict()
