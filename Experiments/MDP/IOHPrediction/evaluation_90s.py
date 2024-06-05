import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn import metrics
import yaml
import sys
sys.path.append('./utils/')
from model_utils_90s import create_model 
import csv

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

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

def predict(model, abp_input, input_shape, batch_size, verbose):
    """Predict using the loaded model."""
    pred = model.predict(abp_input[:, -input_shape:], batch_size=batch_size, verbose=verbose)
    return pred

def evaluate(abp_output, pred, positive_label, threshold):
    """Evaluate model performance."""
    fpr, tpr, thresholds = metrics.roc_curve(abp_output, pred, pos_label=positive_label)
    auc = metrics.auc(fpr, tpr)
    confusion_matrix = metrics.confusion_matrix(abp_output, np.where(np.reshape(pred, -1) > threshold, 1, 0))
    return auc, confusion_matrix

def main():
    # Load configuration
    config = load_config('config_90s.yaml')
    
    # Extract configuration variables
    sec = config['sec']
    data_types = config['data_types']
    model_names = config['model_names']
    datasets = config['datasets']
    model_path = config['model']['path']
    input_layer_name = config['model']['input_layer_name']
    output_layer_name = config['model']['output_layer_name']
    input_shape = config['model']['input_shape']
    batch_size = config['model']['batch_size']
    verbose = config['model']['verbose']
    positive_label = config['evaluation']['positive_label']
    threshold = config['evaluation']['threshold']

    # Create and load model
    model = create_model(input_shape)
    model.load_weights(model_path)

    for dataset in datasets:
        results = []
        for data_type in data_types:
            for model_name in model_names:
                # Generate file paths based on configuration
                hypo_input_path = config['hypo_input_path_template'].format(
                    data_path=config['data_path'], sec=sec, dataset=dataset, model_name=model_name, data_type=data_type)
                nonhypo_input_path = config['nonhypo_input_path_template'].format(
                    data_path=config['data_path'], sec=sec, dataset=dataset, model_name=model_name, data_type=data_type)
                
                # Load input data
                hypo_input = np.load(hypo_input_path)
                hypo_output = np.ones(hypo_input.shape[0])
                nonhypo_input = np.load(nonhypo_input_path)
                nonhypo_output = np.zeros(nonhypo_input.shape[0])

                # Combine and shuffle input data
                abp_input, abp_output = suffle_combine_data(hypo_input, nonhypo_input, hypo_output, nonhypo_output)
                
                # Predict using the model
                pred = predict(model, abp_input, input_shape, batch_size, verbose)
                
                # Evaluate model performance
                AUROC, _ = evaluate(abp_output, pred, positive_label, threshold)
                
                results.append([dataset, sec, data_type, model_name, AUROC])
        
        # Save results to CSV file
        output_path = config['result_path']
        os.makedirs(output_path, exist_ok=True)
        output_file = config['output_filename_template'].format(dataset=dataset, sec=sec)
        with open(os.path.join(output_path, output_file), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Dataset', 'Input Length', 'Data Quality', 'Model', 'AUROC'])
            writer.writerows(results)

if __name__ == "__main__":
    main()
