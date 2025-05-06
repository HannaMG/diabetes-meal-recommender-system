import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from recbole.model.loss import BPRLoss
import random


class ModelBasedCollaborativeFiltering(nn.Module):
    def __init__(self, n_users, n_items, global_mean, user_embeddings, item_embeddings):
        super().__init__()
        self.global_mean = nn.Parameter(torch.tensor([global_mean]), requires_grad=False)
        self.item_bias = nn.Embedding(n_items, 1)
        self.user_bias = nn.Embedding(n_users, 1)

        self.item_embeddings = nn.Embedding.from_pretrained(item_embeddings, freeze=False)
        self.user_embeddings = nn.Embedding.from_pretrained(user_embeddings, freeze=False)

    def forward(self, user_indices, item_indices):
        b_i = self.item_bias(item_indices).squeeze(dim=1)
        b_u = self.user_bias(user_indices).squeeze(dim=1)
        q_i = self.item_embeddings(item_indices)
        p_u = self.user_embeddings(user_indices)

        return self.global_mean + b_i + b_u + torch.sum(torch.mul(q_i, p_u), dim=1)
        
    def predict(self, user, item):
        return self.forward(user, item)
    
class MemoryBasedCollaborativeFiltering(nn.Module):
    def __init__(self, rating_matrix, user_embeddings):
        super().__init__()
        self.rating_matrix = torch.tensor(rating_matrix, dtype=torch.float32)
        
        # Calculate average ratings for each user
        avg_ratings_lis = []
        for i in range(self.rating_matrix.shape[0]):
            row = self.rating_matrix[i]
            valid = ~torch.isnan(row)
            if valid.sum() > 0:
                avg_ratings_lis.append(row[valid].mean())
            else:
                avg_ratings_lis.append(torch.tensor(0.0))

        self.user_embeddings = nn.Embedding.from_pretrained(user_embeddings, freeze=False)
        self.avg_ratings = torch.stack(avg_ratings_lis).to(self.user_embeddings.weight.device)
    
    def forward_single(self, user_idx, item_idx):
        device=self.user_embeddings.weight.device

        user_idx = int(user_idx)
        item_idx = int(item_idx)

        avg_ratings = self.avg_ratings.to(device)

        u = self.user_embeddings(torch.tensor([user_idx], device=device, dtype=torch.long)).squeeze(0)
        avg_u = avg_ratings[user_idx].to(device)
        all_ratings_i = self.rating_matrix[:, item_idx].to(device)

        # Get valid neighbors and valid associated data
        valid_neighbor_indices = torch.where(~torch.isnan(all_ratings_i))[0]
        valid_neighbor_indices = valid_neighbor_indices[valid_neighbor_indices != user_idx]
        valid_v = self.user_embeddings(valid_neighbor_indices)
        valid_avg_v = avg_ratings[valid_neighbor_indices].to(device)
        valid_ratings_i = all_ratings_i[valid_neighbor_indices]

        # Cosine Similarity
        dot_product = torch.matmul(valid_v, u).view(-1)
        magnitude_u = torch.norm(u)
        all_magnitude_v = torch.norm(valid_v, dim=1)
        all_sim = (dot_product / ((magnitude_u * all_magnitude_v) + 1e-8)).view(-1)

        diff_ratings = valid_ratings_i - valid_avg_v

        if torch.sum(torch.abs(all_sim)) == 0:
            return avg_u
        else:
            return avg_u + ((torch.sum(torch.mul(diff_ratings, all_sim)))/torch.sum(torch.abs(all_sim)))

    
    def forward(self, user_indices, item_indices):
        # Go through every user-item pair and get the prediction
        predictions = [
            self.forward_single(u, i)
            for u, i in zip(user_indices, item_indices)
        ]
        return torch.stack(predictions)
    
    def predict(self, user, item):
        return self.forward(user, item)


class HybridRecommender(nn.Module):
    def __init__(self, mod_model, memory_model, n_users):
        super().__init__()
        self.mod_model = mod_model
        self.memory_model = memory_model

        self.alpha = nn.Embedding(n_users, 1)
    
    def forward(self, user_indices, item_indices):
        model_pred = self.mod_model(user_indices, item_indices)
        memory_pred = self.memory_model(user_indices, item_indices)
        
        # Weighted average
        hybrid_pred = self.alpha(user_indices).squeeze(-1) * model_pred + (1 - self.alpha(user_indices).squeeze(-1)) * memory_pred
        return hybrid_pred


    def predict(self, user, item):
        return self.forward(user, item)

# Dataset class for user-item interactions
class InteractionsDataset(Dataset):
    def __init__(self, user_indices, recipe_indices, ratings):
        self.user_indices = user_indices
        self.recipe_indices = recipe_indices
        self.ratings = ratings

    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, index):
        return (torch.tensor(self.user_indices[index]), torch.tensor(self.recipe_indices[index]), torch.tensor(self.ratings[index], dtype=torch.float32))

# Early stopping class
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
def create_user_embedding_matrix(user_embedding_dict, user_mapped_idx):
    matrix = np.zeros((len(user_mapped_idx), len(next(iter(user_embedding_dict.values()))["embedding"])), dtype=np.float32)

    # User indices map to user embeddings
    for user_id, idx in user_mapped_idx.items():
        matrix[idx] = user_embedding_dict[user_id]["embedding"]

    return torch.tensor(matrix)

def create_item_embedding_matrix(item_embedding_dict, item_mapped_idx):
    matrix = np.zeros((len(item_mapped_idx), len(next(iter(item_embedding_dict.values())))), dtype=np.float32) 

    # Item indices map to item embeddings
    for item_id, idx in item_mapped_idx.items():
        matrix[idx] = item_embedding_dict[item_id]

    return torch.tensor(matrix)

def create_rating_matrix(interactions_data, num_users, num_items):
    user_indices, recipe_indices, ratings = interactions_data

    rating_matrix = np.full((num_users, num_items), np.nan, dtype=np.float32)
    
    # User and recipe indices map to ratings
    for user_idx, recipe_idx, rating in zip(user_indices, recipe_indices, ratings):
        rating_matrix[user_idx, recipe_idx] = float(rating)
    
    return rating_matrix

def create_dataset_info(interactions_df, user_mapped_idx, recipe_mapped_idx):
    user_indices = []
    recipe_indices = []
    ratings = []

    # Create tuples of user indices, recipe indices, and ratings
    for i, interaction_row in interactions_df.iterrows():
      user_indices.append(user_mapped_idx[str(interaction_row['user_id'])])  
      recipe_indices.append(recipe_mapped_idx[str(interaction_row['recipe_id'])])
      ratings.append(float(interaction_row['rating']))

    return (user_indices, recipe_indices, ratings)

def build_user_positive_map(train_data):
    user_indices, item_indices, _ = train_data  
    user_positive = {}

    # User indices map to all interacted item indices
    for u, i in zip(user_indices, item_indices):
        if u not in user_positive:
            user_positive[u] = set()
        user_positive[u].add(i)
    return user_positive

def sample_negative(batch_user, num_items, user_positive):
    negative_items = []

    # Iterate through each user in batch and sample negative items
    for u in batch_user.tolist():
        while True:
            rand_item = random.randint(0, num_items - 1)
            # Check if random item is not in user's positive item set
            if rand_item not in user_positive.get(u, set()):
                negative_items.append(rand_item)
                break
    return torch.LongTensor(negative_items).to(batch_user.device)

def train_model(model, device, train_data, val_data, batch_size, shuffle, num_epochs, name, pairwise=False, normalize=False, num_items=None, requires_grad=True): 
    print("Begin model training...")

    # Build a training dataset 
    train_user_indices, train_recipe_indices, train_ratings = train_data
    train_dataset = InteractionsDataset(train_user_indices, train_recipe_indices, train_ratings)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # Build a validation dataset
    val_user_indices, val_recipe_indices, val_ratings = val_data
    val_dataset = InteractionsDataset(val_user_indices, val_recipe_indices, val_ratings)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Set up loss function and optimizer
    criterion = nn.MSELoss()
    user_positive_map = None
    if pairwise:
        criterion = BPRLoss()
        user_positive_map = build_user_positive_map(train_data)
    optimizer = optim.Adam(model.parameters(), lr=0.001) 

    early_stopper = EarlyStopper(patience=3, min_delta=0.001)

    train_loss_list = []
    val_loss_list = []

    best_val_loss = float('inf')

    # Training loop
    
    for epoch in range(num_epochs):
        # Train batch
        model.train()
        train_loss = 0.0
        for batch_user, batch_item, batch_rating in train_dataloader:
            batch_user = batch_user.to(device)
            batch_item = batch_item.to(device)
            batch_rating = batch_rating.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_user, batch_item)
            if normalize:
                batch_rating = batch_rating / 5.0
                #predictions = torch.sigmoid(predictions)
            loss = 0.0
            if pairwise:
                neg_items = sample_negative(batch_user, num_items, user_positive_map)
                neg_scores = model(batch_user, neg_items)
                loss = criterion(predictions, neg_scores)
            else:
                loss = criterion(predictions, batch_rating)
            if requires_grad:
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item() * batch_user.size(0)
            
        avg_train_loss = train_loss / len(train_dataset)
        train_loss_list.append(avg_train_loss)

        # Validation batch
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_user, batch_item, batch_rating in val_dataloader:
                batch_user = batch_user.to(device)
                batch_item = batch_item.to(device)
                batch_rating = batch_rating.to(device)

                predictions = model(batch_user, batch_item)
                loss = 0.0
                if normalize:
                    batch_rating = batch_rating / 5.0
                    #predictions = torch.sigmoid(predictions)
                if pairwise:
                    neg_items = sample_negative(batch_user, num_items, user_positive_map)
                    neg_scores = model(batch_user, neg_items)
                    loss = criterion(predictions, neg_scores)
                else:
                    loss = criterion(predictions, batch_rating)

                val_loss += loss.item() * batch_user.size(0)
            
            avg_val_loss = val_loss / len(val_dataset)
            val_loss_list.append(avg_val_loss)


        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_" + name + ".pth")
            
        torch.save(model.state_dict(), name + ".pth")

        # if early_stopper.early_stop(avg_val_loss):
        #     print("Early stopping triggered.")
        #     break

    # Plot training and validation loss every 25 epochs
        if (epoch+1) % 25 == 0:
            epochs = range(1, len(train_loss_list) + 1)
            plt.figure(figsize=(8, 5))
            plt.plot(epochs, train_loss_list, label='Train Loss')
            plt.plot(epochs, val_loss_list, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss per Epoch for ' + name)
            plt.legend()
            plt.show()

    return model
        