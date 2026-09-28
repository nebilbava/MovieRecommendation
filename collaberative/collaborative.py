import numpy as np
import pandas as pd

df=pd.read_csv("../Data/Ratings.csv")
print("DATASET SAMPLE")
print(df.head())
print()
print(f"Size Of The Dataset:{df.shape}")

num_movies=df["movieId"].max()
num_users=df["userId"].max()
print(f"Size Of the Users:{num_users}")
print(f"Size Of The Movies:{num_movies}")
print()

Y=np.zeros((num_movies,num_users))
R=np.zeros((num_movies,num_users))
for row in df.itertuples():
    movies=row.movieId-1
    users=row.userId-1
    ratings=row.rating
    Y[movies,users]=ratings
R=(Y!=0).astype(int) 

global_mean=np.sum(np.sum(Y)/np.sum(R))
print(f"Global Mean:{global_mean:.4f}")
print()

num_featutes=10
X=np.random.normal(0,0.1, (num_movies,num_featutes))
W=np.random.normal(0,0.1,   (num_users,num_featutes))
b=np.zeros(num_users)                  #bias for users
c=np.zeros((num_movies,1))                 #bias for movies

def cost_function(x,w,b,c,r,y,global_mean,lambda_=0):

    prediction=(x @ w.T) + b + c + global_mean
    error=(prediction-y) * r
    cost=0.5 * np.sum(error**2)

    cost+=(lambda_/2) * (
        np.sum(x**2) + np.sum(w**2) + np.sum(c**2) + np.sum(b**2) )
    return cost

def gradient_calculation(x,w,b,c,r,y,global_mean,lambda_=0):

    prediction=(x @ w.T)+ b + c + global_mean
    error=(prediction-y) * r

    w_gradient= error.T @ x + lambda_ * w
    x_gradient= error @ w + lambda_ * x
    b_gradient=np.sum(error,axis=0) + lambda_ * b
    c_gradient=np.sum(error,axis=1 ,keepdims=True) + lambda_ * c

    return w_gradient,x_gradient,b_gradient,c_gradient

def compute_descent(x,w,b,c,r,y,global_mean,iteration,learning_rate,lambda_=0):

    for i in range(iteration):
        cost=cost_function(x,w,b,c,r,y,global_mean,lambda_)
        w_gradient,x_gradient,b_gradient,c_gradient=gradient_calculation(x,w,b,c,r,y,global_mean,lambda_)

        w= w - w_gradient * learning_rate
        x= x - x_gradient * learning_rate
        b= b - b_gradient * learning_rate  
        c= c - c_gradient * learning_rate
        if i%100==0:
            print(f"Iteration:{i:2d}|Cost:{cost:.2f}")
    return w,x,b,c 
  
learning_rate=0.0001
lambda_=0.01
iteration=2000

W_new,X_new,b_new,c_new=compute_descent(X,W,b,c,R,Y,global_mean,iteration,learning_rate,lambda_)

print()
print("Movie Feature Matrice")
print(W_new)
print()
print("User Feature Metrice")
print(X_new)
print()
print("User Bias")
print(b_new)
print()
print("Movie Bias")
print(c_new)
print()


np.set_printoptions(precision=2, suppress=True)
prediction= (X_new @ W_new.T) + b_new + c_new + global_mean     
print("Predicted Ratings")
print(prediction)


movie_df=pd.read_csv("../Data/Movies.csv")
movie_title=dict(zip(movie_df["movie_id"],movie_df["title"]))

for user in range(num_users):

    unrated_movies=R[:,user]==0
    movie_indices=np.where(unrated_movies)[0]
    predicted_rating=prediction[movie_indices,user]

    sorted_rating=np.argsort(predicted_rating)[::-1]
    recommended_movies=movie_indices[sorted_rating][:5]
    top_rating=predicted_rating[sorted_rating][:5]
    print(f"Recommended Movies for user:{user + 1}")
    for movie,rate in zip(recommended_movies,top_rating):

        movie_id=movie + 1
        movie_name=movie_title.get(int(movie_id),f"Unknown Movie(Id:{movie_id})")
        print(f"Movie Name:{movie_name:<50}|Predicted rating:{rate:.2f}")
        print()

    
    
