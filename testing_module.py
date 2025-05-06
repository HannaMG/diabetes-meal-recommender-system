import os
import torch
import torch.nn as nn
from diabetes_module import HealthAndPreferenceRecommender 
from torch.utils.data import Dataset, DataLoader
import random
import recbole.config 
import recbole_debias.config
from recbole.model.general_recommender import BPR, NeuMF, GCMC, DMF, SLIMElastic
from recbole_debias.model.debiased_recommender import MF
import recbole.data
import recbole_debias.data
import pandas as pd
from doc2vec_module import load_foodcom_data
from hybrid_recommender_module import train_model


def get_blood_glucose(num):
    # Return random blood glucose within a certain range based on input number
    if num == 0: # Level 2 hyperglycemia
        return random.uniform(250, 350)
    elif num == 1: # Level 1 hyperglycemia
        return random.uniform(181, 250)
    elif num == 2: # Normal
        return random.uniform(70, 180)
    elif num == 3: # Level 1 hypoglycemia
        return random.uniform(55, 70)
    else: # Level 2 hypoglycemia
        return random.uniform(10, 55)

# Dataset class for testing interactions
class TestingInteractionsDataset(Dataset):
    def __init__(self, user_indices, recipe_indices, ratings, gl, blood_glucose_num):
        self.user_indices = user_indices
        self.recipe_indices = recipe_indices
        self.ratings = ratings
        self.gl = gl
        self.blood_glucose_num = blood_glucose_num

    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, index):
        return (torch.tensor(self.user_indices[index]), torch.tensor(self.recipe_indices[index]), torch.tensor(self.ratings[index], dtype=torch.float32), torch.tensor(self.gl[index], dtype=torch.float32), torch.tensor(get_blood_glucose(self.blood_glucose_num)))


def ndcg_at_k(model, device, test_data, num, k=None, diabetes_model=False):
    final_ndcg = []

    # Set k
    if k is None:
        k = len(test_data[2])
    else:
        k = min(k, len(test_data[2]))

    for i in range(num):
        ndcg_list = []

        for n in range(5):
            # Build a testing dataset 
            test_user_indices, test_recipe_indices, test_ratings, test_gl = test_data
            test_dataset = TestingInteractionsDataset(test_user_indices, test_recipe_indices, test_ratings, test_gl, n)
            test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

            model.eval()
            with torch.no_grad():
                ratings_list = []
                predictions_list = []

                for batch_user, batch_item, batch_rating, batch_gl, batch_blood_glucose in test_dataloader:
                    batch_user = batch_user.to(device)
                    batch_item = batch_item.to(device)
                    batch_rating = batch_rating.to(device)
                    batch_gl = batch_gl.to(device)
                    batch_blood_glucose = batch_blood_glucose.to(device)

                    ratings_list.append(batch_rating)
                    if diabetes_model:
                        predictions_list.append(model(batch_user, batch_item, batch_blood_glucose, batch_gl))
                    else: 
                        predictions_list.append(model(batch_user, batch_item))

                ratings_tensor = torch.cat(ratings_list)
                predictions_tensor = torch.cat(predictions_list)

                order = torch.argsort(predictions_tensor, descending=True)
                target_k = ratings_tensor[order][:k]
                dcg = torch.sum(target_k/torch.log2(torch.arange(2, k+2, dtype=torch.float32, device=device)))

                ideal_order = torch.argsort(ratings_tensor, descending=True)
                ideal_target_k = ratings_tensor[ideal_order][:k]
                idcg = torch.sum(ideal_target_k/torch.log2(torch.arange(2, k+2, dtype=torch.float32, device=device)))

                if idcg == 0:
                    ndcg_list.append(torch.tensor(0.0, device=device))
                else:
                    ndcg_list.append(dcg / idcg)

        final_ndcg.append(torch.mean(torch.stack(ndcg_list)))

    return torch.mean(torch.stack(final_ndcg))
            

def rmse(model, device, test_data, num, diabetes_model=False):
    final_rmse = []

    for i in range(num):
        rmse_list = []

        for n in range(5):
            # Build a testing dataset 
            test_user_indices, test_recipe_indices, test_ratings, test_gl = test_data
            test_dataset = TestingInteractionsDataset(test_user_indices, test_recipe_indices, test_ratings, test_gl, n)
            test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

            model.eval()
            squared_error_sum = 0.0
            N = 0
            with torch.no_grad():
                for batch_user, batch_item, batch_rating, batch_gl, batch_blood_glucose in test_dataloader:
                    batch_user = batch_user.to(device)
                    batch_item = batch_item.to(device)
                    batch_rating = batch_rating.to(device)
                    batch_gl = batch_gl.to(device)
                    batch_blood_glucose = batch_blood_glucose.to(device)

                    predictions = None
                    if diabetes_model:
                        predictions = model(batch_user, batch_item, batch_blood_glucose, batch_gl)
                    else:
                        predictions = model(batch_user, batch_item)
                    squared_error_sum += torch.sum(((batch_rating/5.0) - predictions)**2)
                    N += batch_rating.size(0)
                
                rmse_list.append(torch.sqrt(squared_error_sum / N))

        final_rmse.append(torch.mean(torch.stack(rmse_list)))

    return torch.mean(torch.stack(final_rmse))

def blood_glucose_freq_at_k(model, device, test_data, num, k=5, diabetes_model=False):
    final_freq_list = []

    for i in range(num):
        freq_hyper2 = []
        freq_hyper1 = []
        freq_normal = []
        freq_hypo1 = []
        freq_hypo2 = []

        for n in range(5):
            # Build a testing dataset 
            test_user_indices, test_recipe_indices, test_ratings, test_gl = test_data
            test_dataset = TestingInteractionsDataset(test_user_indices, test_recipe_indices, test_ratings, test_gl, n)
            test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

            model.eval()
            with torch.no_grad():
                predictions_list = []
                gl_list = []
                blood_glucose_list = []

                for batch_user, batch_item, batch_rating, batch_gl, batch_blood_glucose in test_dataloader:
                    batch_user = batch_user.to(device)
                    batch_item = batch_item.to(device)
                    batch_rating = batch_rating.to(device)
                    batch_gl = batch_gl.to(device)
                    batch_blood_glucose = batch_blood_glucose.to(device)

                    if diabetes_model:
                        predictions_list.append(model(batch_user, batch_item, batch_blood_glucose, batch_gl))
                    else:
                        predictions_list.append(model(batch_user, batch_item))
                    gl_list.append(batch_gl)
                    blood_glucose_list.append(batch_blood_glucose)

                predictions_tensor = torch.cat(predictions_list)
                gl_tensor = torch.cat(gl_list)
                blood_glucose_tensor = torch.cat(blood_glucose_list)

                order = torch.argsort(predictions_tensor, descending=True)
                post_meal_predictions = HealthAndPreferenceRecommender.post_meal_blood_glucose(blood_glucose_tensor[order][:k], gl_tensor[order][:k])
                I_hyper2, I_hyper1, I_normal, I_hypo1, I_hypo2 = HealthAndPreferenceRecommender.compute_indicators(post_meal_predictions)

                
                freq_hyper2.append(I_hyper2)
                freq_hyper1.append(I_hyper1)
                freq_normal.append(I_normal)
                freq_hypo1.append(I_hypo1)
                freq_hypo2.append(I_hypo2)
                
        freq_list = []
        freq_list.append(torch.mean(torch.cat(freq_hyper2)))
        freq_list.append(torch.mean(torch.cat(freq_hyper1)))
        freq_list.append(torch.mean(torch.cat(freq_normal)))
        freq_list.append(torch.mean(torch.cat(freq_hypo1)))
        freq_list.append(torch.mean(torch.cat(freq_hypo2)))

        final_freq_list.append(torch.stack(freq_list))

    return torch.mean(torch.stack(final_freq_list), dim=0)

# Wrapper class for Recbole models
class RecBoleWrapper(nn.Module):
    def __init__(self, recbole_model):
        super().__init__()
        self.model = recbole_model

    def forward(self, batch_user, batch_item):
        input_data = {'user_id': batch_user, 'item_id': batch_item}
        score = self.model.predict(input_data)
        if isinstance(score, tuple):
            return score[0]
        return score
    

def preprocess_interactions():
    # Preprocess the interaction data to create a .inter file for Recbole
    dataset_name = 'RecBoleDatasets'
    
    dataset_dir = os.path.join('./Datasets', dataset_name)

    out_inter_file = os.path.join(dataset_dir, f"{dataset_name}.inter")
    
    train_file = os.path.join(dataset_dir, 'train.csv')
    valid_file = os.path.join(dataset_dir, 'valid.csv')
    test_file  = os.path.join(dataset_dir, 'test.csv')  

    train_df = pd.read_csv(train_file)
    valid_df = pd.read_csv(valid_file)
    test_df = pd.read_csv(test_file)
    dataframes = [train_df, valid_df, test_df]
    
    # Concatenate all interaction data
    inter_df = pd.concat(dataframes, ignore_index=True)

    inter_df['timestamp'] = pd.to_datetime(inter_df['timestamp'], errors='coerce')
    inter_df['timestamp'] = inter_df['timestamp'].fillna(pd.Timestamp("1970-01-01"))
    inter_df['timestamp'] = inter_df['timestamp'].astype('int64') // 10**9

    inter_df.rename(columns={
        'user_id':    'user_id:token',
        'item_id':    'item_id:token',
        'timestamp':  'timestamp:float',
        'rating':     'rating:float',
        'u':          'u:float',
        'i':          'i:float'
    }, inplace=True)
    
    inter_df.to_csv(out_inter_file, index=False, sep="\t")

    print("Successfully created processed .inter file:", out_inter_file)

def load_data_from_config(config, mapping, debias=False):
    # Load data in RecBole format

    foodcom_recipes_path = 'Datasets/recipes_with_GL.csv'
    recipe_GL_df = load_foodcom_data(foodcom_recipes_path, "recipes with GL")
    
    train_df = pd.read_csv(os.path.join(config['data_path'], 'train.csv'))
    val_df = pd.read_csv(os.path.join(config['data_path'], 'valid.csv'))
    test_df = pd.read_csv(os.path.join(config['data_path'], 'test.csv'))


    train_df['user_id'] = train_df['user_id'].astype(int)
    train_df['item_id'] = train_df['item_id'].astype(int)
    val_df['user_id'] = val_df['user_id'].astype(int)
    val_df['item_id'] = val_df['item_id'].astype(int)
    test_df['user_id'] = test_df['user_id'].astype(int)
    test_df['item_id'] = test_df['item_id'].astype(int)


    train_df['user_idx'] = train_df['user_id'].map(lambda x: mapping['user'].get(x, -1))
    train_df['item_idx'] = train_df['item_id'].map(lambda x: mapping['item'].get(x, -1))
    
    val_df['user_idx'] = val_df['user_id'].map(lambda x: mapping['user'].get(x, -1))
    val_df['item_idx'] = val_df['item_id'].map(lambda x: mapping['item'].get(x, -1))

    test_df['user_idx'] = test_df['user_id'].map(lambda x: mapping['user'].get(x, -1))
    test_df['item_idx'] = test_df['item_id'].map(lambda x: mapping['item'].get(x, -1))
    
    test_user_indices = []
    test_recipe_indices = []
    test_ratings = []
    test_gl = []

    if debias:
        for i, interaction_row in test_df.iterrows():
            test_user_indices.append(mapping['user'][interaction_row['user_id']])  
            test_recipe_indices.append(mapping['item'][interaction_row['item_id']])
            test_ratings.append(float(interaction_row['label']))
            test_gl.append(float(recipe_GL_df[recipe_GL_df['id'] == interaction_row['item_id']]['GL'].iloc[0]))

    else:
        for i, interaction_row in test_df.iterrows():
            test_user_indices.append(mapping['user'][interaction_row['user_id']])  
            test_recipe_indices.append(mapping['item'][interaction_row['item_id']])
            test_ratings.append(float(interaction_row['rating']))
            test_gl.append(float(recipe_GL_df[recipe_GL_df['id'] == interaction_row['item_id']]['GL'].iloc[0]))

    train_tuples = None
    val_tuples = None
    if debias:
        train_tuples = (
            train_df['user_idx'].values,
            train_df['item_idx'].values,
            train_df['label'].values
        )
        val_tuples = (
            val_df['user_idx'].values,
            val_df['item_idx'].values,
            val_df['label'].values
        )
    else:
        train_tuples = (
            train_df['user_idx'].values,
            train_df['item_idx'].values,
            train_df['rating'].values
        )
        val_tuples = (
            val_df['user_idx'].values,
            val_df['item_idx'].values,
            val_df['rating'].values
        )
    test_tuples = (
        test_user_indices,
        test_recipe_indices,
        test_ratings,
        test_gl
    )
    return train_tuples, val_tuples, test_tuples


def train_baseline(device, model_name, pairwise, normalize, debias=False):
    config = None
    dataset = None

    if debias:
        config = recbole_debias.config.Config(config_dict={
            'RATING_FIELD': 'label',
            'model': model_name,
            'dataset': 'RecBoleDebiasedDatasets',      
            'data_path': './Datasets/',      
            'split_method': 'predefined',   
            'if_exist': 'rebuild',           
            'save_dataset': True,
            'field_separator': "\t",         
            'logging_level': 'DEBUG',
            'load_col': {
                'inter': ['user_id', 'item_id', 'timestamp', 'label', 'u', 'i'],
                'item': ['name', 'item_id', 'minutes', 'contributor_id', 'submitted',
                        'tags', 'nutrition', 'n_steps', 'steps', 'description', 'ingredients', 'n_ingredients']  
            },
            'epochs': 100,
            'learning_rate': 0.001,

            'task_type': 'rating',
            'loss_type':'MSE',
            'eval_metric': 'RMSE,MAE',
            'inter_matrix_type' : 'rating',
            'train_neg_sample_args':None,
            'train_batch_size':256
        })
        dataset = recbole_debias.data.create_dataset(config)
    else:
        config = recbole.config.Config(config_dict={
            'model': model_name,
            'dataset': 'RecBoleDatasets',         
            'data_path': './Datasets/',      
            'split_method': 'predefined',   
            'if_exist': 'rebuild',           
            'save_dataset': True,
            'field_separator': "\t",         
            'logging_level': 'DEBUG',
            'load_col': {
                'inter': ['user_id', 'item_id', 'timestamp', 'rating', 'u', 'i'],
                'item': ['name', 'item_id', 'minutes', 'contributor_id', 'submitted',
                        'tags', 'nutrition', 'n_steps', 'steps', 'description', 'ingredients', 'n_ingredients']  # If you have an item side-information file (item.csv), you can add its configuration here.
            },
            'epochs': 100,
            'learning_rate': 0.001,

            'task_type': 'rating',
            'loss_type':'MSE',
            'eval_metric': 'RMSE,MAE',
            'inter_matrix_type' : 'rating',
            'train_neg_sample_args':None
        })
    
        dataset = recbole.data.create_dataset(config)

    user_arr = dataset.field2id_token['user_id']
    item_arr = dataset.field2id_token['item_id']

    user_mapping = {int(token): idx for idx, token in enumerate(user_arr) if token != "[PAD]"}
    item_mapping = {int(token): idx for idx, token in enumerate(item_arr) if token != "[PAD]"}

    mapping = {'user': user_mapping, 'item': item_mapping}

    model = {
        'BPR':BPR,
        'NeuMF':NeuMF,
        'DMF':DMF,
        'GCMC':GCMC,
        'SLIMElastic':SLIMElastic,
        'MF':MF
    }

    recbole_model = model.get(model_name)(config, dataset)
    
    model = RecBoleWrapper(recbole_model)
    model.to(device)
    
    train_data_tuple = None
    valid_data_tuple = None

    if debias:
        train_data_tuple, valid_data_tuple, _ = load_data_from_config(config, mapping, debias=True)
    else:
        train_data_tuple, valid_data_tuple, _ = load_data_from_config(config, mapping)

    train_model(model, device, train_data_tuple, valid_data_tuple, 256, True, 100, model_name, pairwise=pairwise, normalize=normalize, num_items=len(item_mapping))

def test_baseline(device, model_name, model_path, num=10, debias=False):
    config = None
    dataset = None

    if debias:
        config = recbole_debias.config.Config(config_dict={
            'model': model_name,
            'dataset': 'RecBoleDebiasedDatasets',      
            'data_path': './Datasets/',      
            'split_method': 'predefined',   
            'if_exist': 'rebuild',           
            'save_dataset': True,
            'field_separator': "\t",         
            'logging_level': 'DEBUG',
            'load_col': {
                'inter': ['user_id', 'item_id', 'timestamp', 'label', 'u', 'i'],
                'item': ['name', 'item_id', 'minutes', 'contributor_id', 'submitted',
                        'tags', 'nutrition', 'n_steps', 'steps', 'description', 'ingredients', 'n_ingredients']  
            },
            'epochs': 100,
            'learning_rate': 0.001,

            'task_type': 'rating',
            'loss_type':'MSE',
            'eval_metric': 'RMSE,MAE',
            'inter_matrix_type' : 'rating',
            'train_neg_sample_args':None,
            'train_batch_size':256
        })
        dataset = recbole_debias.data.create_dataset(config)
    else:
        config = recbole.config.Config(config_dict={
            'model': model_name,
            'dataset': 'RecBoleDatasets',         
            'data_path': './Datasets/',      
            'split_method': 'predefined',   
            'if_exist': 'rebuild',           
            'save_dataset': True,
            'field_separator': "\t",         
            'logging_level': 'DEBUG',
            'load_col': {
                'inter': ['user_id', 'item_id', 'timestamp', 'rating', 'u', 'i'],
                'item': ['name', 'item_id', 'minutes', 'contributor_id', 'submitted',
                        'tags', 'nutrition', 'n_steps', 'steps', 'description', 'ingredients', 'n_ingredients']  
            },
            'epochs': 100,
            'learning_rate': 0.001,

            'task_type': 'rating',
            'loss_type':'MSE',
            'eval_metric': 'RMSE,MAE',
            'inter_matrix_type' : 'rating',
            'train_neg_sample_args':None
        })
    
        dataset = recbole.data.create_dataset(config)

    user_arr = dataset.field2id_token['user_id']
    item_arr = dataset.field2id_token['item_id']

    user_mapping = {int(token): idx for idx, token in enumerate(user_arr) if token != "[PAD]"}
    item_mapping = {int(token): idx for idx, token in enumerate(item_arr) if token != "[PAD]"}

    mapping = {'user': user_mapping, 'item': item_mapping}

    model = {
        'BPR':BPR,
        'NeuMF':NeuMF,
        'DMF':DMF,
        'GCMC':GCMC,
        'MF':MF
    }

    recbole_model = model.get(model_name)(config, dataset)
    
    model = RecBoleWrapper(recbole_model)
    model.to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    
    test_data_tuple = None
    if debias:
        _, _, test_data_tuple = load_data_from_config(config, mapping, debias=True)
    else:
        _, _, test_data_tuple = load_data_from_config(config, mapping)

    print(f"\n{model_name}")
    print("--------------------------------------------------------------------------")
    print(f"ndcg@10: {ndcg_at_k(model, device, test_data_tuple, num, k=10, diabetes_model=False)}")
    print(f"blood glucose frequencies@10: {blood_glucose_freq_at_k(model, device, test_data_tuple, num, k=10, diabetes_model=False)}")
    print("--------------------------------------------------------------------------")

def test_my_model(device, model, model_path, test_data, num=10):

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    print(f"\nHealth and Preference Recommender")
    print("--------------------------------------------------------------------------")
    print(f"ndcg@10: {ndcg_at_k(model, device, test_data, num, k=10, diabetes_model=True)}")
    print(f"blood glucose frequencies@10: {blood_glucose_freq_at_k(model, device, test_data, num, k=10, diabetes_model=True)}")
    print("--------------------------------------------------------------------------")


