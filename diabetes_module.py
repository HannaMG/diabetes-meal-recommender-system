import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from hybrid_recommender_module import EarlyStopper
import random
import matplotlib.pyplot as plt

class HealthRecommender(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, blood_glucose, gl):
        blood_glucose = blood_glucose.float()
        gl = gl.float()

        # Define conditions for hyperglycemia, normal, and hypoglycemia
        condition_hyper = (blood_glucose > 180) & (gl <= 10)
        condition_normal = ((blood_glucose >= 70) & (blood_glucose <= 180)) & (gl < 15)
        condition_hypo = (blood_glucose < 70) & (gl >= 15)

        # Calculate health score based on conditions
        # Hyperglycemia and Normal: health score decreases as glycemic load increases 
        # Hypoglycemia: health score increases as glycemic load increases
        health_score = torch.zeros_like(blood_glucose)
        health_score = torch.where(condition_hyper, torch.exp(-torch.sqrt(gl)), health_score)
        health_score = torch.where(condition_normal, torch.exp(-torch.sqrt(gl)), health_score)
        health_score = torch.where(condition_hypo, 1.0 - torch.exp(-torch.sqrt(gl)), health_score)

        return health_score
            
    
class HealthAndPreferenceRecommender(nn.Module):
    def __init__(self, hybrid_recommender, health_recommender, n_users):
        super(HealthAndPreferenceRecommender, self).__init__()
        # Recommender models
        self.hybrid_recommender = hybrid_recommender
        self.health_recommender = health_recommender

        # Blood glucose level alpha parameters
        self.alpha_hyper2 = nn.Embedding(n_users, 1)
        self.alpha_hyper1 = nn.Embedding(n_users, 1)
        self.alpha_normal  = nn.Embedding(n_users, 1)
        self.alpha_hypo1   = nn.Embedding(n_users, 1)
        self.alpha_hypo2   = nn.Embedding(n_users, 1)
        self.bias = nn.Embedding(n_users, 1)
    
    @staticmethod
    def post_meal_blood_glucose(blood_glucose, gl):
        # Approximate post-meal blood glucose level calculated for testing purposes
        return blood_glucose + (gl * 4)
    
    @staticmethod
    def compute_indicators(post_meal):
        # Compute indicators for level 2 hyperglycemia, level 1 hyperglycemia, normal blood glucose level, level 1 hypoglycemia, and level 2 hypoglycemia
        I_hyper2 = (post_meal >= 250).float()
        I_hyper1 = ((post_meal < 250) & (post_meal > 180)).float()
        I_normal   = ((post_meal <= 180) & (post_meal >= 70)).float()
        I_hypo1    = ((post_meal < 70) & (post_meal >= 55)).float()
        I_hypo2    = (post_meal < 55).float()
        return I_hyper2, I_hyper1, I_normal, I_hypo1, I_hypo2

    def forward(self, user_indices, item_indices, blood_glucose, gl):
        preference_score = self.hybrid_recommender(user_indices, item_indices) / 5.0 # Normalized
        health_score = self.health_recommender(blood_glucose, gl) 

        post_meal = self.post_meal_blood_glucose(blood_glucose, gl)
        
        I_hyper2, I_hyper1, I_normal, I_hypo1, I_hypo2 = self.compute_indicators(post_meal)
        
        alpha = torch.sigmoid(self.alpha_hyper2(user_indices).squeeze(-1) * I_hyper2 + 
                 self.alpha_hyper1(user_indices).squeeze(-1) * I_hyper1 + 
                 self.alpha_normal(user_indices).squeeze(-1)  * I_normal + 
                 self.alpha_hypo1(user_indices).squeeze(-1)   * I_hypo1 + 
                 self.alpha_hypo2(user_indices).squeeze(-1)   * I_hypo2 + 
                 self.bias(user_indices).squeeze(-1))
        
        final_score = alpha * health_score + (1 - alpha) * preference_score
        return final_score

# Dataset class for user interactions with diabetes components
class DiabetesInteractionsDataset(Dataset):
    def __init__(self, user_indices, recipe_indices, ratings, gl):
        self.user_indices = user_indices
        self.recipe_indices = recipe_indices
        self.ratings = ratings
        self.gl = gl

    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, index):
        return (torch.tensor(self.user_indices[index]), torch.tensor(self.recipe_indices[index]), torch.tensor(self.ratings[index], dtype=torch.float32), torch.tensor(self.gl[index], dtype=torch.float32), torch.tensor(generate_random_blood_glucose(1)[0]))


def generate_random_blood_glucose(n_users):
    # Generate random blood glucose values, from a range of 10 to 350 mg/dL, for n_users
    # In a real scenario, you would replace this with actual blood glucose data
    blood_glucose_lis = []
    for i in range(n_users):
        blood_glucose_lis.append(random.uniform(10, 350))
    return blood_glucose_lis

def create_dataset_info_gl(interactions_df, recipe_GL_df, user_mapped_idx, recipe_mapped_idx):
    user_indices = []
    recipe_indices = []
    ratings = []
    gl = []

    # Iterate through data and extract user and recipe indices, ratings, and GL values
    for i, interaction_row in interactions_df.iterrows():
      user_indices.append(user_mapped_idx[str(interaction_row['user_id'])])  
      recipe_indices.append(recipe_mapped_idx[str(interaction_row['recipe_id'])])
      ratings.append(float(interaction_row['rating']))
      gl.append(float(recipe_GL_df[recipe_GL_df['id'] == interaction_row['recipe_id']]['GL'].iloc[0]))

    return (user_indices, recipe_indices, ratings, gl)

def train_model(model, device, train_data, val_data, batch_size, shuffle, num_epochs):    
    print("Begin model training...")

    # Build a training dataset 
    train_user_indices, train_recipe_indices, train_ratings, train_gl = train_data
    train_dataset = DiabetesInteractionsDataset(train_user_indices, train_recipe_indices, train_ratings, train_gl)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # Build a validation dataset
    val_user_indices, val_recipe_indices, val_ratings, val_gl = val_data
    val_dataset = DiabetesInteractionsDataset(val_user_indices, val_recipe_indices, val_ratings, val_gl)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Set up loss function and optimizer
    criterion = nn.MSELoss()
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
        for batch_user, batch_item, batch_rating, batch_gl, batch_blood_glucose in train_dataloader:
            batch_user = batch_user.to(device)
            batch_item = batch_item.to(device)
            batch_rating = batch_rating.to(device)
            batch_gl = batch_gl.to(device)
            batch_blood_glucose = batch_blood_glucose.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_user, batch_item, batch_blood_glucose, batch_gl)
            order = torch.argsort(predictions, descending=True) 
            post_meal_predictions = HealthAndPreferenceRecommender.post_meal_blood_glucose(batch_blood_glucose[order][:10], batch_gl[order][:10])
            I_hyper2, I_hyper1, I_normal, I_hypo1, I_hypo2 = HealthAndPreferenceRecommender.compute_indicators(post_meal_predictions)
            pref_loss = 0.0
            pref_loss = criterion(predictions, batch_rating/5.0)
            health_loss = torch.relu(torch.mean(I_hyper2) - 0.05) + torch.relu(torch.mean(I_hyper1) - 0.25) + torch.relu(0.70 - torch.mean(I_normal)) + torch.relu(torch.mean(I_hypo1) - 0.04) + torch.relu(torch.mean(I_hypo2) - 0.01)
            loss = pref_loss + health_loss  
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_user.size(0)
            
        avg_train_loss = train_loss / len(train_dataset)
        train_loss_list.append(avg_train_loss)

        # Validation batch
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_user, batch_item, batch_rating, batch_gl, batch_blood_glucose in val_dataloader:
                batch_user = batch_user.to(device)
                batch_item = batch_item.to(device)
                batch_rating = batch_rating.to(device)
                batch_gl = batch_gl.to(device)
                batch_blood_glucose = batch_blood_glucose.to(device)

                predictions = model(batch_user, batch_item, batch_blood_glucose, batch_gl)
                order = torch.argsort(predictions, descending=True) 
                post_meal_predictions = HealthAndPreferenceRecommender.post_meal_blood_glucose(batch_blood_glucose[order][:10], batch_gl[order][:10])
                I_hyper2, I_hyper1, I_normal, I_hypo1, I_hypo2 = HealthAndPreferenceRecommender.compute_indicators(post_meal_predictions)
                pref_loss = 0.0
                pref_loss = criterion(predictions, batch_rating/5.0)
                health_loss = torch.relu(torch.mean(I_hyper2) - 0.05) + torch.relu(torch.mean(I_hyper1) - 0.25) + torch.relu(0.70 - torch.mean(I_normal)) + torch.relu(torch.mean(I_hypo1) - 0.04) + torch.relu(torch.mean(I_hypo2) - 0.01)
                loss = pref_loss + health_loss          

                val_loss += loss.item() * batch_user.size(0)
            
            avg_val_loss = val_loss / len(val_dataset)
            val_loss_list.append(avg_val_loss)


        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_health_and_preference_recommender.pth")
            
        torch.save(model.state_dict(), "health_and_preference_recommender.pth")

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
            plt.title('Training and Validation Loss per Epoch for Health and Preference Recommender')
            plt.legend()
            plt.show()

    return model

