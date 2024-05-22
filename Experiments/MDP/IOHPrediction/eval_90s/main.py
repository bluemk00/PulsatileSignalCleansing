import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn import metrics
import yaml
from model import create_model 
import csv

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

data_root = config['data_root']
sec = config['sec']
data_types = config['data_types']
name_list = config['name_list']
dataset_list = config['dataset']
dtw_version_list = config['dtw_version']

model_path = config['model']['path']
input_layer_name = config['model']['input']
output_layer_name = config['model']['output']

input_shape = config['model']['input_shape']

model = create_model(input_shape) 

model.load_weights(model_path) 

def generate_data_paths(name, data_type, sec, dataset, dtw_version):
    dtw_suffix = '_withoutDTW' if dtw_version == 'without_DTW' else ''
    dtw_prefix = '' if dtw_version == 'without_DTW' else 'DTW_'

    if name == 'uncleaned':
        if data_type == 'Clean':
            base_path = config['paths']['uncleaned']['base_path'].format(data_root=data_root, dataset=dataset)
            hypo_input_path = os.path.join(base_path, config['paths']['uncleaned']['file_pattern']['hypo'].format(data_type=data_type, sec=sec))
            nonhypo_input_path = os.path.join(base_path, config['paths']['uncleaned']['file_pattern']['nonhypo'].format(data_type=data_type, sec=sec))
            hypo_input_paths = []
            nonhypo_input_paths = [] 
            hypo_index_path = ''
            nonhypo_index_path = ''
        else:
            base_path = config['paths']['uncleaned']['base_path'].format(data_root=data_root, dataset=dataset)
            noise_base_path = config['paths']['uncleaned']['noise']['base_path'].format(data_root=data_root, dataset=dataset)
            hypo_input_path = os.path.join(base_path, config['paths']['uncleaned']['file_pattern']['hypo'].format(data_type=data_type, sec=sec))
            nonhypo_input_path = os.path.join(base_path, config['paths']['uncleaned']['file_pattern']['nonhypo'].format(data_type=data_type, sec=sec))
            hypo_input_paths = [os.path.join(noise_base_path, config['paths']['uncleaned']['noise']['file_pattern']['hypo'].format(sec=sec))]
            nonhypo_input_paths = [os.path.join(noise_base_path, config['paths']['uncleaned']['noise']['file_pattern']['nonhypo'].format(sec=sec))]
            idx_base_path = config['paths']['uncleaned']['idx']['base_path'].format(data_root=data_root, dataset=dataset)
            hypo_index_path = os.path.join(idx_base_path, config['paths']['uncleaned']['idx']['file_pattern']['hypo'].format(ids=config['paths']['uncleaned']['idx']['ids'][dataset], sec=sec))
            nonhypo_index_path = os.path.join(idx_base_path, config['paths']['uncleaned']['idx']['file_pattern']['nonhypo'].format(ids=config['paths']['uncleaned']['idx']['ids'][dataset], sec=sec))
    else:
        if data_type == 'Clean':
            base_path = config['paths']['others']['base_path'].format(data_root=data_root, dataset=dataset, dtw_suffix=dtw_suffix)
            hypo_input_path = os.path.join(base_path, config['paths']['others']['file_pattern']['hypo'].format(name=name, data_type=data_type, sec=sec, dtw_prefix=dtw_prefix))
            nonhypo_input_path = os.path.join(base_path, config['paths']['others']['file_pattern']['nonhypo'].format(name=name, data_type=data_type, sec=sec, dtw_prefix=dtw_prefix))
            hypo_input_paths = []
            nonhypo_input_paths = []
            hypo_index_path = ''
            nonhypo_index_path = ''
        else:
            base_path = config['paths']['others']['base_path'].format(data_root=data_root, dataset=dataset, dtw_suffix=dtw_suffix)
            noise_base_path = config['paths']['others']['noise']['base_path'].format(data_root=data_root, dataset=dataset)
            hypo_input_path = os.path.join(base_path, config['paths']['others']['file_pattern']['hypo'].format(name=name, data_type=data_type, sec=sec, dtw_prefix=dtw_prefix))
            nonhypo_input_path = os.path.join(base_path, config['paths']['others']['file_pattern']['nonhypo'].format(name=name, data_type=data_type, sec=sec, dtw_prefix=dtw_prefix))
            hypo_input_paths = [os.path.join(noise_base_path, config['paths']['others']['noise']['file_pattern'][dtw_version]['hypo'].format(name=name, sec=sec, dtw_prefix=dtw_prefix))]
            nonhypo_input_paths = [os.path.join(noise_base_path, config['paths']['others']['noise']['file_pattern'][dtw_version]['nonhypo'].format(name=name, sec=sec, dtw_prefix=dtw_prefix))]
            idx_base_path = config['paths']['others']['idx']['base_path'].format(data_root=data_root, dataset=dataset)
            hypo_index_path = os.path.join(idx_base_path, config['paths']['others']['idx']['file_pattern']['hypo'].format(ids=config['paths']['others']['idx']['ids'][dataset], sec=sec))
            nonhypo_index_path = os.path.join(idx_base_path, config['paths']['others']['idx']['file_pattern']['nonhypo'].format(ids=config['paths']['others']['idx']['ids'][dataset], sec=sec))
    
    return hypo_input_path, nonhypo_input_path, hypo_input_paths, nonhypo_input_paths, hypo_index_path, nonhypo_index_path

def load_data(name, data_type, dataset, dtw_version):
    hypo_input_path, nonhypo_input_path, hypo_input_paths, nonhypo_input_paths, hypo_index_path, nonhypo_index_path = generate_data_paths(name, data_type, sec, dataset, dtw_version)
    
    hypo_input = np.load(hypo_input_path)
    nonhypo_input = np.load(nonhypo_input_path)
    hypo_output = np.ones(hypo_input.shape[0])
    nonhypo_output = np.zeros(nonhypo_input.shape[0])
    
    if data_type == 'Noise':
        add_hypo_input = np.concatenate([np.load(path) for path in hypo_input_paths], axis=0)
        add_nonhypo_input = np.concatenate([np.load(path) for path in nonhypo_input_paths], axis=0)
        add_hypo_output = np.ones(add_hypo_input.shape[0])
        add_nonhypo_output = np.zeros(add_nonhypo_input.shape[0])
        
        idx_hypo = np.load(hypo_index_path)
        idx_nonhypo = np.load(nonhypo_index_path)
        
        if dataset == 'vitaldb' and name == 'uncleaned':
            add_hypo_input = (add_hypo_input - 20) / 200
            add_nonhypo_input = (add_nonhypo_input - 20) / 200
        
        hypo_input = np.concatenate((hypo_input, add_hypo_input[idx_hypo]), axis=0)
        nonhypo_input = np.concatenate((nonhypo_input, add_nonhypo_input[idx_nonhypo]), axis=0)
        hypo_output = np.concatenate((hypo_output, add_hypo_output), axis=0)
        nonhypo_output = np.concatenate((nonhypo_output, add_nonhypo_output), axis=0)
    
    if name == 'uncleaned':
        if dataset == 'vitaldb':
            hypo_input = hypo_input * 180.0 / 200.0
            nonhypo_input = nonhypo_input * 180.0 / 200.0
        else:
            hypo_input = (hypo_input - 20) / 200
            nonhypo_input = (nonhypo_input - 20) / 200
        
    return hypo_input, nonhypo_input, hypo_output, nonhypo_output

def shuffle_data(inputs, outputs):
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    shuffled_inputs = inputs[indices]
    shuffled_outputs = outputs[indices]
    return shuffled_inputs, shuffled_outputs

def preprocess_data(hypo_input, nonhypo_input, hypo_output, nonhypo_output):
    shuffled_hypo_input, shuffled_hypo_output = shuffle_data(hypo_input, hypo_output)
    shuffled_nonhypo_input, shuffled_nonhypo_output = shuffle_data(nonhypo_input, nonhypo_output)
    
    combined_input = np.concatenate((shuffled_hypo_input, shuffled_nonhypo_input), axis=0)
    combined_output = np.concatenate((shuffled_hypo_output, shuffled_nonhypo_output), axis=0)
    
    combined_indices = np.arange(len(combined_input))
    np.random.shuffle(combined_indices)
    abp_input = combined_input[combined_indices]
    abp_output = combined_output[combined_indices]
    
    abp_input = np.clip(abp_input, 0, 1)
    
    return abp_input, abp_output

def predict(abp_input):
    pred = model.predict(abp_input[:, -9000:], batch_size=1000, verbose=1)
    return pred

def evaluate(abp_output, pred):
    fpr, tpr, thresholds = metrics.roc_curve(abp_output[:], pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    confusion_matrix = metrics.confusion_matrix(abp_output, np.where(np.reshape(pred, -1) > 0.5, 1, 0))
    return auc, confusion_matrix

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

results = []

for data_type in data_types:
    for name in name_list:
        for dataset in dataset_list:
            for dtw_version in dtw_version_list:
                hypo_input, nonhypo_input, hypo_output, nonhypo_output = load_data(name, data_type, dataset, dtw_version)
                abp_input, abp_output = preprocess_data(hypo_input, nonhypo_input, hypo_output, nonhypo_output)
                
                pred = predict(abp_input)
                auc, confusion_matrix = evaluate(abp_output, pred)
                
                result = {
                    'Data Type': data_type,
                    'Name': name,
                    'Dataset': dataset,
                    'DTW Version': dtw_version,
                    'AUC': auc
                }
                results.append([name, sec, data_type, dataset, dtw_version, auc])

output_path = './results/'
save_results_to_csv(results, output_path)