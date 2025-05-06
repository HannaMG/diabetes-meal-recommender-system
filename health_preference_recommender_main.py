import os
import torch
import nltk
from gensim.models.doc2vec import Doc2Vec
from diabetes_module import HealthAndPreferenceRecommender, HealthRecommender, create_dataset_info_gl, train_model
from doc2vec_module import load_foodcom_data, create_doc2vec_model, get_recipe_embeddings, get_user_embeddings, get_global_rating_mean, map_ids_to_idx_recipes, map_ids_to_idx_users
from hybrid_recommender_module import create_item_embedding_matrix, create_user_embedding_matrix, ModelBasedCollaborativeFiltering, create_dataset_info, MemoryBasedCollaborativeFiltering, create_rating_matrix, HybridRecommender

def main():
    nltk.download('punkt_tab')

    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

    # Datasets
    foodcom_recipes_path = 'Datasets/recipes_with_GL.csv'
    foodcom_interactions_train_path = 'Datasets/Food.com/interactions_train.csv'
    foodcom_interaction_validation_path = 'Datasets/Food.com/interactions_validation.csv'
    
    foodcom_recipes_df = load_foodcom_data(foodcom_recipes_path, "recipes with GL")
    foodcom_interactions_train_df = load_foodcom_data(foodcom_interactions_train_path, "interactions training")
    foodcom_interaction_val_df = load_foodcom_data(foodcom_interaction_validation_path, "interactions validation")

    print("Columns in the dataset:", foodcom_recipes_df.columns.tolist())

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
    print(f"example user embedding: {train_user_embeddings['76535']}")

    train_global_rating_mean = get_global_rating_mean(train_user_embeddings)
    print(f"Global rating mean: {train_global_rating_mean}")

    # Get mapped indices
    recipe_mapped_idx = map_ids_to_idx_recipes(recipe_embeddings)
    train_user_mapped_idx = map_ids_to_idx_users(train_user_embeddings, True)
    val_user_mapped_idx = map_ids_to_idx_users(val_user_embeddings, False)

    # Get embedding matrices
    recipe_embedding_matrix = create_item_embedding_matrix(recipe_embeddings, recipe_mapped_idx)
    train_user_embedding_matrix = create_user_embedding_matrix(train_user_embeddings, train_user_mapped_idx)

    # Get necessary info for dataset
    train_interactions_data = create_dataset_info(foodcom_interactions_train_df, train_user_mapped_idx, recipe_mapped_idx)
    val_interactions_data_gl = create_dataset_info_gl(foodcom_interaction_val_df, foodcom_recipes_df, val_user_mapped_idx, recipe_mapped_idx)
    train_interactions_data_gl = create_dataset_info_gl(foodcom_interactions_train_df,foodcom_recipes_df, train_user_mapped_idx, recipe_mapped_idx)

    # Create models
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

    # Train model
    train_model(model, device, train_interactions_data_gl, val_interactions_data_gl, 256, True, 100)


if __name__ == '__main__':
    main()
