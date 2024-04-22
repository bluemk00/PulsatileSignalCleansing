import os
import numpy as np
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_and_prepare_data(sec, data, name, dataset, dtw_version, config):
    data_root = config['data_root']
    paths = config['paths']
    dtw_suffix = '_withoutDTW' if dtw_version == 'without_DTW' else ''
    dtw_prefix = '' if dtw_version == 'without_DTW' else 'DTW_'

    if name == 'uncleaned':
        base_path = paths['uncleaned']['base_path'].format(data_root=data_root, dataset=dataset)
        hypo_path = os.path.join(base_path, paths['uncleaned']['file_pattern']['hypo'].format(data=data, sec=sec))
        nonhypo_path = os.path.join(base_path, paths['uncleaned']['file_pattern']['nonhypo'].format(data=data, sec=sec))
        normed = False
    else:
        base_path = paths['others']['base_path'].format(data_root=data_root, dataset=dataset, dtw_suffix=dtw_suffix)
        hypo_path = os.path.join(base_path, paths['others']['file_pattern']['hypo'].format(name=name, data=data, sec=sec, dtw_prefix=dtw_prefix))
        nonhypo_path = os.path.join(base_path, paths['others']['file_pattern']['nonhypo'].format(name=name, data=data, sec=sec, dtw_prefix=dtw_prefix))
        normed = True

    hypo_in = np.load(hypo_path)[:500]
    nonhypo_in = np.load(nonhypo_path)[:500]

    if not normed:
        if dataset == 'vitaldb' and name == 'uncleaned':
            hypo_in = hypo_in * 180.0 / 200.0
            nonhypo_in = nonhypo_in * 180.0 / 200.0
        else:
            hypo_in = (hypo_in - 20.0) / 200.0
            nonhypo_in = (nonhypo_in - 20.0) / 200.0

    if data == 'Noise':
        if name == 'uncleaned':
            noise_base_path = paths['uncleaned']['noise']['base_path'].format(data_root=data_root, dataset=dataset)
            noise_hypo_path = os.path.join(noise_base_path, paths['uncleaned']['noise']['file_pattern']['hypo'].format(sec=sec))
            noise_nonhypo_path = os.path.join(noise_base_path, paths['uncleaned']['noise']['file_pattern']['nonhypo'].format(sec=sec))
            normed = False
        else:
            noise_base_path = paths['others']['noise']['base_path'].format(data_root=data_root, dataset=dataset)
            noise_hypo_path = os.path.join(noise_base_path, paths['others']['noise']['file_pattern'][dtw_version]['hypo'].format(name=name, sec=sec, dtw_prefix=dtw_prefix))
            noise_nonhypo_path = os.path.join(noise_base_path, paths['others']['noise']['file_pattern'][dtw_version]['nonhypo'].format(name=name, sec=sec, dtw_prefix=dtw_prefix))

        noise_hypo_in = np.load(noise_hypo_path)
        noise_nonhypo_in = np.load(noise_nonhypo_path)

        if not normed:
            noise_hypo_in = (noise_hypo_in - 20.0) / 200.0
            noise_nonhypo_in = (noise_nonhypo_in - 20.0) / 200.0

        if sec != '30s':
            idx_base_path = paths['uncleaned' if name == 'uncleaned' else 'others']['idx']['base_path'].format(data_root=data_root, dataset=dataset)
            idx_hypo_path = os.path.join(idx_base_path, paths['uncleaned' if name == 'uncleaned' else 'others']['idx']['file_pattern']['hypo'].format(ids=paths['uncleaned' if name == 'uncleaned' else 'others']['idx']['ids'][dataset], sec=sec))
            idx_nonhypo_path = os.path.join(idx_base_path, paths['uncleaned' if name == 'uncleaned' else 'others']['idx']['file_pattern']['nonhypo'].format(ids=paths['uncleaned' if name == 'uncleaned' else 'others']['idx']['ids'][dataset], sec=sec))

            idx_hypo = np.load(idx_hypo_path)
            idx_nonhypo = np.load(idx_nonhypo_path)

            selected_noise_hypo_in = noise_hypo_in[idx_hypo]
            selected_noise_nonhypo_in = noise_nonhypo_in[idx_nonhypo]
        else:
            selected_noise_hypo_in = noise_hypo_in
            selected_noise_nonhypo_in = noise_nonhypo_in

        All_hypo_in = np.concatenate((hypo_in, selected_noise_hypo_in), axis=0)
        All_nonhypo_in = np.concatenate((nonhypo_in, selected_noise_nonhypo_in), axis=0)
    else:
        All_hypo_in = hypo_in
        All_nonhypo_in = nonhypo_in

    hypo_out = np.ones(All_hypo_in.shape[0])
    nonhypo_out = np.zeros(All_nonhypo_in.shape[0])

    All_hypo_in = np.clip(All_hypo_in, 0.0, 1.0)
    All_nonhypo_in = np.clip(All_nonhypo_in, 0.0, 1.0)

    return All_hypo_in, All_nonhypo_in, hypo_out, nonhypo_out

def shuffle_and_combine_data(All_hypo_in, All_nonhypo_in, hypo_out, nonhypo_out):
    hypo_pairs = list(zip(All_hypo_in, hypo_out))
    nonhypo_pairs = list(zip(All_nonhypo_in, nonhypo_out))
    combined_pairs = hypo_pairs + nonhypo_pairs
    np.random.shuffle(combined_pairs)
    shuffled_in = np.array([pair[0] for pair in combined_pairs])
    shuffled_out = np.array([pair[1] for pair in combined_pairs])
    return shuffled_in, shuffled_out