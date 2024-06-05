import numpy as np
import os
from sklearn import metrics
import yaml
import csv
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import load_model

def suffle_combine_data(hypo_input, nonhypo_input, hypo_output, nonhypo_output):
    """Shuffle and combine hypo and nonhypo input data."""
    combined_input = np.concatenate((hypo_input, nonhypo_input), axis=0)
    combined_output = np.concatenate((hypo_output, nonhypo_output), axis=0)
    
    combined_indices = np.arange(len(combined_input))
    np.random.shuffle(combined_indices)
    abp_input = combined_input[combined_indices]
    abp_output = combined_output[combined_indices]
    
    abp_input = np.clip(abp_input, 0, 1)
    
    return abp_input, abp_output

def main():
    # Load configuration from YAML file
    with open('config_30s60s.yaml', 'r') as file:
        config = yaml.safe_load(file)

    datasets = config['datasets']
    times = config['times']
    data_types = config['data_types']
    model_names = config['model_names']

    output_path = config['result_path']
    os.makedirs(output_path, exist_ok=True)

    for dataset in datasets:
        for sec in times:
            results = []
            for data_type in data_types:
                for model_name in model_names:
                    # Generate file paths based on configuration
                    hypo_input_path = config['hypo_input_path_template'].format(data_path=config['data_path'], 
                                            sec=sec, dataset=dataset, model_name=model_name, data_type=data_type)
                    nonhypo_input_path = config['nonhypo_input_path_template'].format(data_path=config['data_path'], 
                                            sec=sec, dataset=dataset, model_name=model_name, data_type=data_type)

                    # Load input data
                    hypo_input = np.load(hypo_input_path)
                    hypo_output = np.ones(hypo_input.shape[0])
                    nonhypo_input = np.load(nonhypo_input_path)
                    nonhypo_output = np.zeros(nonhypo_input.shape[0])

                    # Combine and shuffle input data
                    abp_input, abp_output = suffle_combine_data(hypo_input, nonhypo_input, hypo_output, nonhypo_output)

                    # Adjust input data based on time segment
                    if sec == '30s':
                        abp_input = abp_input[:, -3000:]
                    elif sec == '60s':
                        abp_input = abp_input[:, -6000:]

                    abp_input = np.expand_dims(abp_input, axis=-1)

                    # Load pre-trained model
                    loaded_model = load_model(config['model']['path'][sec])

                    # Calculate AUC scores
                    model_auc_scores = []
                    for i in range(10):
                        y_pred = loaded_model.predict(abp_input)
                        auc = roc_auc_score(abp_output, y_pred)
                        model_auc_scores.append(auc)
                    avg_model_auc = np.mean(model_auc_scores)

                    results.append([dataset, sec, data_type, model_name, avg_model_auc])

            # Write results to CSV file
            output_file = config['output_filename_template'].format(dataset=dataset, sec=sec)
            with open(os.path.join(output_path, output_file), 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Dataset', 'Input Length', 'Data Quality', 'Model', 'AUROC'])
                writer.writerows(results)

if __name__ == "__main__":
    main()
