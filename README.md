# 🎬 Movie Recommendation System

A movie recommendation system built using **Machine Learning** with two different recommendation approaches:

* **Collaborative Filtering**
* **Content-Based Filtering**

The project demonstrates how different recommendation techniques can be used to recommend movies based on either **user ratings** or **movie characteristics**.

---

## 📌 Project Overview

This project contains two independent recommendation systems.

### 1. Collaborative Filtering

The collaborative filtering model recommends movies based on **user rating patterns**.

It learns:

* Movie features
* User preferences
* User bias
* Movie bias
* Global rating average

The model uses **Matrix Factorization** to learn relationships between users and movies.

### 2. Content-Based Filtering

The content-based model recommends movies that are similar to a movie selected by the user.

It uses information such as:

* Movie title
* Genres
* Keywords
* Overview
* Original language
* Cast
* Crew

The movie information is combined into a single text representation and converted into numerical features using **TF-IDF**.

**Cosine Similarity** is then used to find movies with similar content.

---

## 📂 Project Structure

```text
MovieRecommendation/
│
├── Data/
│   ├── Movies.csv
│   ├── Ratings.csv
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
│
├── collaborative/
│   └── collaborative.py
│
├── content/
│   └── content.py
│
└── README.md
```

> Dataset filenames and folder names can be adjusted according to the final project structure.

---

## 🤝 Collaborative Filtering

The collaborative filtering system uses user ratings to learn movie and user representations.

The ratings dataset contains information such as:

```text
userId
movieId
rating
timestamp
```

The ratings are converted into a matrix where:

* Rows represent movies
* Columns represent users
* Values represent ratings

Missing ratings are represented using a separate matrix.

The model then learns latent features for users and movies.

### Main concepts used

* Matrix Factorization
* Gradient Descent
* User Bias
* Movie Bias
* Global Mean
* Regularization

The predicted rating can be represented conceptually as:

```text
Predicted Rating =
Global Mean
+ User Bias
+ Movie Bias
+ User Features × Movie Features
```

The system then recommends movies that the user has not rated based on their predicted ratings.

---

## 🎥 Content-Based Filtering

The content-based system recommends movies based on their characteristics.

Movie information from the TMDB dataset is processed and combined into a single text field.

For example:

```text
title + keywords + overview + language + genres + cast + crew
```

This combined text is converted into TF-IDF vectors.

### Main concepts used

* Text preprocessing
* TF-IDF
* Cosine Similarity
* Feature extraction

The similarity between movies is calculated using:

```text
Cosine Similarity
```

When a movie is selected, the system finds other movies with the highest similarity scores and recommends them.

For example:

```text
Input:
Avatar

Recommendations:
Aliens
Alien³
Alien
Moonraker
...
```

---

## 🧠 Technologies Used

* **Python**
* **NumPy**
* **Pandas**
* **Scikit-learn**

### Machine Learning

* Matrix Factorization
* Gradient Descent
* TF-IDF
* Cosine Similarity

---

## 📊 Datasets

### Collaborative Filtering

The collaborative filtering system uses a **MovieLens-style ratings dataset** containing users, movies, and ratings.

### Content-Based Filtering

The content-based system uses the **TMDB 5000 Movies** and **TMDB 5000 Credits** datasets.

The two recommendation systems use different datasets
