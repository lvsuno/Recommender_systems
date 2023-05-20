# The rating here is the recommandation criteria
 #%% import librairies
import pandas as pd
from sklearn import model_selection, preprocessing
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from sklearn.metrics import mean_squared_error

#%% data import
df = pd.read_csv("ratings.csv")
df.head()
# %%
print(f"Unique Users: {df.userId.nunique()}, Unique Movies : {df.movieId.nunique()}")
# %%  Data class

class MovieDataset (Dataset):
    def __init__(self, users, movies, ratings) -> None:
        self.users = users # user_id
        self.movies = movies # movie_id
        self.ratings = ratings # rating

    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, index):
        users = self.users[index]
        movies = self.movies[index]
        ratings = self.ratings[index]
        return torch.tensor(users, dtype=torch.long), torch.tensor(movies, dtype=torch.long), torch.tensor(ratings, dtype=torch.long)

#%% Model class
class RecSysModel(nn.Module):
    def __init__(self, n_users, n_movies, n_embeddings = 32) -> None:
        super().__init__()
        # Embed the users data and the item (movies) data
        # Here it's a simplified situation. Users and movies data can be more
        # than that.
        self.user_embed = nn.Embedding(n_users, n_embeddings)
        self.movie_embed = nn.Embedding(n_movies, n_embeddings)
        self.out = nn.Linear(n_embeddings* 2, 1)

    def forward(self, users, movies):
        user_embeds = self.user_embed(users) # embed users
        movie_embeds = self.movie_embed(movies) # embed movies
        x = torch.cat([user_embeds, movie_embeds], dim=1)
        x = self.out(x)
        return x
    
#%% encode user and movie id to start from 0
lbl_user = preprocessing.LabelEncoder()
lbl_movie = preprocessing.LabelEncoder()
df.userId = lbl_user.fit_transform(df.userId.values)
df.movieId = lbl_movie.fit_transform(df.movieId.values)

#%% Create Training and Testing data
df_train, df_test = model_selection.train_test_split(df, test_size= 0.2,
                                                     random_state=42, stratify=df.rating.values)

#%% Dataset instances
train_dataset = MovieDataset(
    users=df_train.userId.values,
    movies=df_train.movieId.values,
    ratings=df_train.rating.values
)

test_dataset = MovieDataset(
    users=df_test.userId.values,
    movies=df_test.movieId.values,
    ratings=df_test.rating.values
)

#%% DataLoader
BATCH_SIZE = 4
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

#%% Model Instance, Optimizer, and Loss Function

model = RecSysModel(n_users=len(lbl_user.classes_), n_movies=len(lbl_movie.classes_))
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.MSELoss()

#%% Model Training
NUM_EPOCHS = 1
model.train()
for epoch in range(NUM_EPOCHS):
    for users, movies, ratings in train_loader:
        optimizer.zero_grad()
        y_pred = model(users, movies)
        
        y_true = ratings.unsqueeze(dim=1).to(torch.float32)

        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()

#%% Model Evaluation 
y_preds = []
y_trues = []
user_movie_test = defaultdict(list)  # empty dictionary of list
model.eval()

with torch.no_grad():
    for users, movies, ratings, in test_loader:
        y_pred = model(users, movies)
        for i in range(len(users)):
            user_id = users[i].item()
            movie_id = movies[i].item()
            pred_rating = y_pred[i][0].item()
            true_rating = ratings[i].item()
            print(f"User: {user_id}, Movie: {movie_id}, Pred: {pred_rating}, True: {true_rating}")
            user_movie_test[user_id].append((pred_rating, true_rating))


#%% Precision and Recall

precisions = {}
recalls = {}

k = 10
thres = 3.5

for uid, user_ratings in user_movie_test.items():
    # sort user_ratings by rating
    user_ratings.sort(key=lambda x: x[0], reverse=True) 

    # count of the relevant items
    n_rel = sum((true_rating>thres) for (_, true_rating) in user_ratings)
    # count recommended items that are predicted relevent within Top K
    n_reck = sum((pred_rating>thres) for (pred_rating,_) in user_ratings[:k])

   # count recommended items that are relevant

    n_rel_and_nreck = sum(((true_rating>thres) and (pred_rating>thres)) 
                         for (pred_rating, true_rating) in user_ratings[:k])

    print(f"uid {uid},  n_rel {n_rel}, n_rec_k {n_reck}, n_rel_and_rec_k {n_rel_and_nreck}")

    precisions[uid] = n_rel_and_nreck / n_reck if n_reck != 0 else 0

    recalls[uid] = n_rel_and_nreck / n_rel if n_rel != 0 else 0

print(f"Precision @ {k}: {sum(precisions.values()) / len(precisions)}")

print(f"Recall @ {k} : {sum(recalls.values()) / len(recalls)}")  
# %%
