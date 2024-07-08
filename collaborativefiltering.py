import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import zipfile
import os
import urllib.request

def create_params(size):
    return nn.Parameter(torch.zeros(*size).normal_(0, 0.01))

def get_embedding_sizes(data):
    cat_columns = data.select_dtypes(include=['object', 'category']).columns
    embedding_sizes = []
    for col in cat_columns:
        num_unique_values = data[col].nunique()
        embedding_size = min(600, round(1.6 * num_unique_values**0.56))
        embedding_sizes.append((num_unique_values, embedding_size))
    return embedding_sizes

class RatingsDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['user'].cat.codes.values, dtype=torch.long)
        self.movies = torch.tensor(df['title'].cat.codes.values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

class PMF(nn.Module):
    def __init__(self, n_users, n_movies, n_factors, y_range=(0, 5.5)):
        super().__init__()
        self.user_factors = create_params([n_users,n_factors])
        self.user_bias =  create_params([n_users])
        self.movie_factors = create_params([n_movies,n_factors])
        self.movie_bias = create_params([n_movies])
        self.y_range = y_range

    def forward(self, x):
        users = self.user_factors[x[:, 0]]
        movies = self.movie_factors[x[:, 1]]

        res = (users * movies).sum(dim=1)
        res += self.user_bias[x[:, 0]] + self.movie_bias[x[:, 1]]
        return torch.sigmoid(res) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]

class PMFNN(nn.Module):
    def __init__(self, userEmbSize, itemEmbSize, y_range=(0, 5.5),n_act=100):
        super().__init__()
        self.user_factors = nn.Embedding(*userEmbSize)
        self.movie_factors = nn.Embedding(*itemEmbSize)
        self.layers = nn.Sequential(
            nn.Linear(userEmbSize[1] + itemEmbSize[1], n_act, bias=True),
            nn.ReLU(),
            nn.Linear(n_act, 1, bias=True),
        )
        self.y_range = y_range

    def forward(self, x):
        embs = self.user_factors(x[:, 0]), self.movie_factors(x[:, 1])
        x = self.layers(torch.cat(embs, dim=1)).squeeze()
        return torch.sigmoid(x) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]


def train(model, train_loader, val_loader, loss_func, optimizer, scheduler, epochs=5):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for users, movies, ratings in train_loader:
            x = torch.stack([users, movies], dim=1)
            y = ratings

            optimizer.zero_grad()
            preds = model(x).squeeze()
            loss = loss_func(preds, y)        
            loss.backward()
            optimizer.step()
            scheduler.step()            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for users, movies, ratings in val_loader:
                x = torch.stack([users, movies], dim=1)
                y = ratings
                preds = model(x)
                loss = loss_func(preds, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")


#data retrieval and prep
url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
zip_file = 'ml-100k.zip'
urllib.request.urlretrieve(url, zip_file)


with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall('ml-100k')

ratings_path = os.path.join('ml-100k/ml-100k', 'u.data')
movies_path = os.path.join('ml-100k/ml-100k', 'u.item')

ratings_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
movies_columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 
                  'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
                  'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']


ratings = pd.read_csv(ratings_path, sep='\t', names=ratings_columns, encoding='latin-1')
movies = pd.read_csv(movies_path, sep='|', names=movies_columns, encoding='latin-1')

movies = movies.drop_duplicates(subset='movie_id').sort_values(by='title')
ratings = ratings.merge(movies)

ratings['user'] = ratings['user_id'].astype('category')
ratings['title'] = ratings['title'].astype('category')
ratings.drop(['timestamp','movie_id','user_id'], axis=1, inplace=True)
ratings = ratings[['user','title','rating']]


# Split data into training and validation sets
train_df, val_df = train_test_split(ratings, test_size=0.2, random_state=42)

train_dataset = RatingsDataset(train_df)
val_dataset = RatingsDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


#Probabilistic Matrix Factorization
epochs = 10
#manual embedding sizes
n_users = len(np.unique(ratings['user'].cat.codes.values))
n_movies = len(np.unique(ratings['title'].cat.codes.values))
n_factors = 50
model = PMF(n_users, n_movies, 50)
loss_func = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3,weight_decay=.2)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=5e-3, steps_per_epoch=len(train_loader), epochs=epochs, pct_start=0.25, anneal_strategy='cos', div_factor=25.0, final_div_factor=100000.0
)
train(model, train_loader, val_loader, loss_func, optimizer,scheduler, epochs=epochs)

#worst and best biases
movie_bias = model.movie_bias.squeeze()
idxs = movie_bias.argsort(descending=True)[:5]
[ratings['title'].cat.categories[i.item()] for i in idxs]

movie_bias = model.movie_bias.squeeze()
idxs = movie_bias.argsort(descending=False)[:5]
[ratings['title'].cat.categories[i.item()] for i in idxs]

#pca plot
top_title_ids = ratings.groupby('title')['rating'].count().sort_values(ascending=False).index.codes[:1000]
top_title_tensor = torch.tensor(top_title_ids)

movie_factors = model.movie_factors.detach().cpu().numpy()
top_movie_factors = movie_factors[top_title_tensor]

pca = PCA(n_components=3)
movie_pca = pca.fit_transform(top_movie_factors)
fac0, fac1, fac2 = movie_pca[:, 0], movie_pca[:, 1], movie_pca[:, 2]



idxs = list(range(50))
X = fac0[idxs]
Y = fac2[idxs]

plt.figure(figsize=(12, 12))
plt.scatter(X, Y)

for i, x, y in zip(idxs, X, Y):
    title = f"{top_title_tensor[i].item()} - {ratings['title'].cat.categories[top_title_tensor[i].item()]}"
    plt.text(x, y, title, color=np.random.rand(3)*0.7, fontsize=11)

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 3')
plt.title('PCA of Movie Factors')
plt.grid(True)
plt.show()


#NN variant

epochs = 10

embedding_sizes = get_embedding_sizes(ratings)
model = PMFNN(*embedding_sizes)
loss_func = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3,weight_decay=.2)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=5e-3, steps_per_epoch=len(train_loader), epochs=epochs, pct_start=0.25, anneal_strategy='cos', div_factor=25.0, final_div_factor=100000.0
)

train(model, train_loader, val_loader, loss_func, optimizer,scheduler, epochs=epochs)
