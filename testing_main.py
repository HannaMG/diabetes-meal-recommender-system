import os
import torch
from diabetes_module import HealthAndPreferenceRecommender, HealthRecommender, create_dataset_info_gl
from doc2vec_module import create_doc2vec_model, get_global_rating_mean, get_recipe_embeddings, get_user_embeddings, load_foodcom_data, map_ids_to_idx_recipes, map_ids_to_idx_users
from hybrid_recommender_module import HybridRecommender, MemoryBasedCollaborativeFiltering, ModelBasedCollaborativeFiltering, create_dataset_info, create_item_embedding_matrix, create_rating_matrix, create_user_embedding_matrix
from testing_module import preprocess_interactions, test_baseline, test_my_model, train_baseline
from gensim.models.doc2vec import Doc2Vec


def main():
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

    dataset_name = 'RecBoleDatasets'
    dataset_dir = os.path.join('./Datasets', dataset_name)
    
    processed_file = os.path.join(dataset_dir, f"{dataset_name}.inter")
    
    # If the processed .inter file is missing, run the preprocessing step
    if not os.path.exists(processed_file):
        print("Processed .inter file not found, running preprocessing...")
        preprocess_interactions()
    else:
        print("Processed .inter file found:", processed_file)

    # Datasets
    foodcom_recipes_path = 'Datasets/recipes_with_GL.csv'
    foodcom_interactions_train_path = 'Datasets/Food.com/interactions_train.csv'
    foodcom_interaction_validation_path = 'Datasets/Food.com/interactions_validation.csv'
    foodcom_interactions_test_path = 'Datasets/Food.com/interactions_test.csv'

    foodcom_recipes_df = load_foodcom_data(foodcom_recipes_path, "recipes with GL")
    foodcom_interactions_train_df = load_foodcom_data(foodcom_interactions_train_path, "interactions training")
    foodcom_interaction_val_df = load_foodcom_data(foodcom_interaction_validation_path, "interactions validation")
    foodcom_interactions_test_df = load_foodcom_data(foodcom_interactions_test_path, "interactions testing")
    
    if os.path.exists("doc2vec_recipe_model.model"):
        print("Existing doc2vec_recipe_model.model was found.")
    else:
        print("No doc2vec_recipe_model.model was found. Creating one now.")
        create_doc2vec_model(foodcom_recipes_df, 100, 5, 2, 40)

    # Load doc2vec model
    doc2vec_model = Doc2Vec.load('doc2vec_recipe_model.model')

    # Create recipe embeddings
    recipe_embeddings = get_recipe_embeddings(doc2vec_model, foodcom_recipes_df)
    print(f"example recipe embedding: {recipe_embeddings['137739']}")

    # Create user embeddings
    train_user_embeddings = get_user_embeddings(doc2vec_model, foodcom_interactions_train_df, recipe_embeddings)
    val_user_embeddings = get_user_embeddings(doc2vec_model, foodcom_interaction_val_df, recipe_embeddings)
    test_user_embeddings = get_user_embeddings(doc2vec_model, foodcom_interactions_test_df, recipe_embeddings)
    print(f"example user embedding: {train_user_embeddings['76535']}")

    train_global_rating_mean = get_global_rating_mean(train_user_embeddings)
    print(f"Global rating mean: {train_global_rating_mean}")
   
    # Get mapped indices
    recipe_mapped_idx = map_ids_to_idx_recipes(recipe_embeddings)
    train_user_mapped_idx = map_ids_to_idx_users(train_user_embeddings, True)
    val_user_mapped_idx = map_ids_to_idx_users(val_user_embeddings, False)
    test_user_mapped_idx = map_ids_to_idx_users(test_user_embeddings, False)

    # Get embedding matrices
    recipe_embedding_matrix = create_item_embedding_matrix(recipe_embeddings, recipe_mapped_idx)
    train_user_embedding_matrix = create_user_embedding_matrix(train_user_embeddings, train_user_mapped_idx)
    
    # Get necessary info for dataset
    train_interactions_data = create_dataset_info(foodcom_interactions_train_df, train_user_mapped_idx, recipe_mapped_idx)
    val_interactions_data_gl = create_dataset_info_gl(foodcom_interaction_val_df, foodcom_recipes_df, val_user_mapped_idx, recipe_mapped_idx)
    train_interactions_data_gl = create_dataset_info_gl(foodcom_interactions_train_df,foodcom_recipes_df, train_user_mapped_idx, recipe_mapped_idx)
    test_interactions_data_gl = create_dataset_info_gl(foodcom_interactions_test_df,foodcom_recipes_df, test_user_mapped_idx, recipe_mapped_idx)

    
    content_model = ModelBasedCollaborativeFiltering(len(train_user_mapped_idx), len(recipe_mapped_idx), train_global_rating_mean, train_user_embedding_matrix, recipe_embedding_matrix)
    content_model.to(device)

    rating_matrix = create_rating_matrix(train_interactions_data, len(train_user_mapped_idx), len(recipe_mapped_idx))
    collab_model = MemoryBasedCollaborativeFiltering(rating_matrix, train_user_embedding_matrix)
    collab_model.to(device)

    hybrid_model = HybridRecommender(content_model, collab_model, len(train_user_mapped_idx))
    hybrid_model.to(device)

    health_model = HealthRecommender()
    health_model.to(device)

    model = HealthAndPreferenceRecommender(hybrid_model, health_model, len(train_user_mapped_idx))
    model.to(device)

    train_baseline(device, 'BPR', True, False)
    train_baseline(device, 'NeuMF', False, False)
    train_baseline(device, 'GCMC', False, False)
    train_baseline(device, 'DMF', False, True)
    train_baseline(device, 'MF', False, False, debias=True)

    test_my_model(device, model, './best_health_and_preference_recommender.pth', test_interactions_data_gl)
    test_baseline(device, 'BPR', './best_BPR.pth')
    test_baseline(device, 'NeuMF', './best_NeuMF.pth')
    test_baseline(device, 'GCMC', './best_GCMC.pth')
    test_baseline(device, 'DMF', './best_DMF.pth')
    test_baseline(device, 'MF', './best_MF.pth', debias=True)


if __name__ == '__main__':
    main()
