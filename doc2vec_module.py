import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def load_foodcom_data(path, name):
    # Load Food.com data from CSV file
    try:
        foodcom_df = pd.read_csv(path)
        print(f"Food.com {name} data loaded successfully.")
        return foodcom_df
    except Exception as e:
        print(f"Error loading Food.com {name} data:", e)
        return pd.DataFrame()
    
def generate_recipe_doc_lis(recipes_df):
    recipe_doc_lis = []
        
    # Create TaggedDocument for each recipe
    for i, recipe_row in recipes_df.iterrows():
        elements = []
        for col in ['name', 'minutes', 'contributor_id', 'submitted', 'tags', 'nutrition', 'n_steps', 'steps', 'description', 'ingredients', 'n_ingredients']:
            elements.append(str(recipe_row[col]))
        recipe_text = " ".join(elements)

        tokens = word_tokenize(recipe_text.lower())
        recipe_doc_lis.append(TaggedDocument(words=tokens, tags=[str(recipe_row['id'])]))

    return recipe_doc_lis
    
def create_doc2vec_model(recipes_df, vector_size, window_size, min_count, epochs):
    recipe_doc_lis = generate_recipe_doc_lis(recipes_df)

    # Initialize and build vocabulary
    doc2vec_model = Doc2Vec(vector_size=vector_size, window=window_size, min_count=min_count, workers=28, epochs=epochs)
    doc2vec_model.build_vocab(recipe_doc_lis)
    print("Vocabulary built. Total unique tokens:", len(doc2vec_model.wv.index_to_key))

    # Train and save model
    doc2vec_model.train(recipe_doc_lis, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
    print("Doc2Vec model training complete.")
    doc2vec_model.save("doc2vec_recipe_model.model")
    print("Model saved as doc2vec_recipe_model.model.")

def get_recipe_embeddings(model, recipes_df):
    recipe_doc_lis = generate_recipe_doc_lis(recipes_df)
    recipe_embedding_dict = {}

    # Get or create recipe embeddings using Doc2Vec model
    for doc in recipe_doc_lis:
        try:
            recipe_embedding_dict[doc.tags[0]] = model.dv[str(doc.tags[0])]
        except KeyError:
            print(f"Recipe embedding for recipe id {doc.tags[0]} not found. Creating new embedding.")
            recipe_embedding_dict[doc.tags[0]] = model.infer_vector(doc.words)

    return recipe_embedding_dict

def get_user_embeddings(model, interactions_df, recipe_embedding_dict):
    user_embedding_dict = {}

    # Create user embeddings using interactions
    for i, interaction_row in interactions_df.iterrows():
        user_id = str(interaction_row['user_id'])
        recipe_id = str(interaction_row['recipe_id'])
        rating = interaction_row['rating']
        
        if recipe_id not in recipe_embedding_dict:
            print(f"Recipe embedding for recipe id {recipe_id} not found. Skipping...")
            continue

        if user_id not in user_embedding_dict: # Create new user entry in dictionary
            user_embedding_dict[user_id] = {
                "embedding": rating*recipe_embedding_dict[recipe_id],
                "recipes": {recipe_id: rating}
            }
        else:
            if recipe_id not in user_embedding_dict[user_id]["recipes"]:
                user_embedding_dict[user_id]["embedding"] += rating*recipe_embedding_dict[recipe_id]
                user_embedding_dict[user_id]["recipes"][recipe_id] = rating
            else:
                print(f"User id {user_id} already has recipe id {recipe_id}. Skipping...")
                continue
    # Compute final averages
    for user_id, info in user_embedding_dict.items():
        ratings_sum = 0
        for recipe_id, rating in info["recipes"].items():
            ratings_sum += rating
        if ratings_sum > 0:
            user_embedding_dict[user_id]["embedding"] /= ratings_sum
        else:
            user_embedding_dict[user_id]["embedding"] = np.zeros(model.vector_size)

    return user_embedding_dict

def get_global_rating_mean(user_embedding_dict):
    num_ratings = 0
    sum_ratings = 0

    # Iterate through user embeddings and calculate the global mean rating
    for user_id, info in user_embedding_dict.items():
        # Iterate through the recipes rated by the user
        for recipe_id, rating in info["recipes"].items():
            num_ratings += 1
            sum_ratings += rating

    return sum_ratings/num_ratings

def map_ids_to_idx_users(embedding_dict, create):
    # Initialize mapping dictionary and index, if not done so already
    if not hasattr(map_ids_to_idx_users, "mapping_dict"):
        map_ids_to_idx_users.mapping_dict = {}
    if not hasattr(map_ids_to_idx_users, "idx"):
        map_ids_to_idx_users.idx = 0

    ret_mapping_dict = {}

    # Iterate through user embedding dictionary and map user IDs to indices
    for id, data in embedding_dict.items():
        if id not in map_ids_to_idx_users.mapping_dict:
            if create: # Create new mapping for unseen user IDs
                map_ids_to_idx_users.mapping_dict[id] = map_ids_to_idx_users.idx
                map_ids_to_idx_users.idx += 1
            else: # If not creating, assign -1 index for unseen user IDs
                print(f"User not seen before.")
                ret_mapping_dict[id] = -1
                continue
        ret_mapping_dict[id] = map_ids_to_idx_users.mapping_dict[id]  

    return ret_mapping_dict

def map_ids_to_idx_recipes(embedding_dict):
    # Initialize mapping dictionary and index, if not done so already
    if not hasattr(map_ids_to_idx_recipes, "mapping_dict"):
        map_ids_to_idx_recipes.mapping_dict = {}
    if not hasattr(map_ids_to_idx_recipes, "idx"):
        map_ids_to_idx_recipes.idx = 0

    ret_mapping_dict = {}

    # Iterate through recipe embedding dictionary and map recipe IDs to indices
    for id, data in embedding_dict.items():
        if id not in map_ids_to_idx_recipes.mapping_dict:
            map_ids_to_idx_recipes.mapping_dict[id] = map_ids_to_idx_recipes.idx
            map_ids_to_idx_recipes.idx += 1
        ret_mapping_dict[id] = map_ids_to_idx_recipes.mapping_dict[id]  

    return ret_mapping_dict


