import os
import csv
import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import load_model
from data_utils import load_config, load_and_prepare_data, shuffle_and_combine_data

def evaluate_model(sec, data, name, dataset, dtw_version, config):
    hypo_in, nonhypo_in, hypo_out, nonhypo_out = load_and_prepare_data(sec, data, name, dataset, dtw_version, config)
    shuffled_in, shuffled_out = shuffle_and_combine_data(hypo_in, nonhypo_in, hypo_out, nonhypo_out)

    if sec == '30s':
        shuffled_in = shuffled_in[:, -3000:]
    elif sec == '60s':
        shuffled_in = shuffled_in[:, -6000:]

    shuffled_in = np.expand_dims(shuffled_in, axis=-1)

    model_paths = config['model']['paths'][sec]
    all_model_auc_scores = []

    for path in model_paths:
        loaded_model = load_model(path)
        model_auc_scores = []

        for i in range(10):
            y_pred = loaded_model.predict(shuffled_in)
            auc = roc_auc_score(shuffled_out, y_pred)
            model_auc_scores.append(auc)

        avg_model_auc = np.mean(model_auc_scores)
        all_model_auc_scores.append(avg_model_auc)

    avg_auc_all_models = np.mean(all_model_auc_scores)
    return avg_auc_all_models

def save_results_to_csv(results, output_path):
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, 'auc_results.csv')

    sorted_results = sorted(results, key=lambda x: (x[4], x[3]))

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)

        current_dtw = None
        current_dataset = None

        for result in sorted_results:
            model, sec, data, dataset, dtw, auc = result

            if dtw != current_dtw:
                current_dtw = dtw
                writer.writerow([])
                writer.writerow([f'DTW Version: {dtw}'])

            if dataset != current_dataset:
                current_dataset = dataset
                writer.writerow([f'DataSet: {dataset}'])

            writer.writerow([model, sec, data, auc])

    print(f'Results saved to: {output_file}')

def main():
    config_path = 'config.yaml'
    config = load_config(config_path)

    results = []
    for sec in config['sec']:
        for data in config['data_type']:
            for model in config['name_list']:
                model_results = []
                for dataset in config['dataset']:
                    for dtw_version in config['dtw_version']:
                        avg_auc = evaluate_model(sec, data, model, dataset, dtw_version, config)
                        result = [model, sec, data, dataset, dtw_version, avg_auc]
                        model_results.append(result)
                results.extend(model_results)

    output_path = './results/'
    save_results_to_csv(results, output_path)

if __name__ == '__main__':
    main()