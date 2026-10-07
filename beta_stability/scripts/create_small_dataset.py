import matplotlib.pyplot as plt
import json
from beta_stability.run import run as run_lca
from run_baseline import run as run_baseline
from run_baseline_topk import run_baseline_topk 
from beta_stability.util.tools import *
from beta_stability.util.init_logger import init_logger
from beta_stability.preprocess import preprocess_data, save_data

species = 'Spermwhale'
x = 'num human'
y='precision'
# x = 'prob_human_error'
# y='error_rate'
config_path = 'configs/config_spermwhale.yaml'






def run_lca(config_path, species):


    results_lca = []

    config = get_config(config_path)
    exp_name = config['exp_name']
    data_params = config['data']

    annotations = data_params['annotation_file']


    name_keys = data_params['name_keys']
    filter_key = '__'.join(name_keys)
    df = preprocess_data(data_params['annotation_file'], 
                        name_keys= name_keys,
                        convert_names_to_ids=True, 
                        viewpoint_list=data_params['viewpoint_list'], 
                        n_filter_min=data_params['n_filter_min'], 
                        n_filter_max=data_params['n_filter_max'],
                        images_dir = data_params['images_dir'], 
                        embedding_uuids = None,
                        format='old'
                    )

    # human_correct_probs, results_lca, results_baseline_topk, results_baseline_threshold = load_pickle(f"/ekaterina/work/src/lca/lca/tmp/plot/{species}__{exp_name}.pickle")
    # print(len(results_lca[0]))
    # (gt_results, r_results, node2uuid) = results_lca[0]

    # print(len(gt_results[-1]))

    # incorrect_clusters = gt_results[-1]['non equal']

    # uuids_set = set()
    # for cluster in incorrect_clusters:
    #     uuids_set.add(node2uuid[list(cluster['GT clustering'])[0]])


    # filtered_df = df[df['uuid_x'].isin(uuids_set)]
    filtered_df = df

    unique_name_keys = filtered_df[filter_key].unique()
    print(len(unique_name_keys))

    # Step 2: Randomly select 5 unique name_keys
    selected_name_keys = random.sample(list(unique_name_keys), 60)

    # Step 3: Filter the DataFrame to keep only the selected name_keys
    filtered_df_by_name_keys = df[df[filter_key].isin(selected_name_keys)]

    filtered_df_by_name_keys['species'] = species

    save_data(filtered_df_by_name_keys, f'/ekaterina/work/src/lca/lca/data/{species}_small.json')

    # print(node2uuid)


        




configs_with_species = [
    # ('./configs/config_grevyszebra.yaml', 'Grevy\'s Zebra'),
    # ('./configs/config_plainszebra.yaml', 'Plains Zebra'),
    # ('./configs/config_forestelephants.yaml', 'Forest Elephants'),
    # ('./configs/config_whaleshark.yaml', 'Whale Shark'),  # Uncomment if needed
    # # ('./configs/config_giraffe.yaml', 'Giraffe'),
    # ('./configs/config_beluga.yaml', 'Beluga Whale')
    ('./configs/config_spermwhale.yaml', 'spermwhale')
]


for config_path, species in configs_with_species:
    run_lca(config_path, species)
    # plot_from_pickle(config_path, x='num human', y='precision',  xlabel="Number of human reviews", ylabel='`Precision', species=species)
   