# Hybrid-Recommendation-System
Movie &amp; Anime Recommender: Watch Smart, Not Hard! 🍿🤖

*(Unsupervised Clustering & Content-Based Filtering)*

### **1. Project Overview**

This project focuses on building a **hybrid recommendation system** that combines **unsupervised clustering (K-Means)** and **content-based filtering** to recommend both **Netflix movies and anime**. The motivation is to provide personalized recommendations in domains where **explicit user feedback (ratings, watch history)** is sparse, leveraging instead the **intrinsic properties of the content** (e.g., genres, themes, keywords, descriptions).

The system is capable of:

* Segmenting movies and anime into **clusters** based on latent content similarities.
* Recommending items from the **most relevant cluster** for a given user query.
* Offering **content-based recommendations** using textual and numeric features.

---

### **2. Problem Statement**

Traditional collaborative filtering struggles in cold-start situations or when ratings are limited. To overcome this, our system:

* Uses **K-Means clustering** to segment titles into groups that capture hidden patterns (e.g., genres, themes, audience styles).
* Employs **TF-IDF text embeddings** and similarity metrics to recommend content aligned with user preferences.
* Provides a **hybrid approach**, combining **global cluster-level insights** with **local content similarity** for better personalization.

---

### **3. Data Sources & Preprocessing**

#### **Movie Dataset (4,803 rows, 20 columns)**

* Columns include: *budget, genres, overview, popularity, revenue, runtime, tagline, vote\_average, vote\_count, etc.*
* **Missing values**: 3,941 across 5 columns (homepage, overview, release\_date, runtime, tagline).
* **Transformations**:

  * Log transformation on highly skewed columns: `budget`, `revenue`, `popularity`, `vote_count`.
  * Square-root transformation on moderately skewed column: `runtime`.
* **Observations**: Budget, revenue, and popularity are right-skewed; runtime distribution is fairly centered.

#### **Anime Dataset (6,668 rows, 33 columns)**

* Columns include: *anime\_id, title, episodes, score, members, favorites, genre, aired\_from\_year, duration\_min, etc.*
* **Missing values**: 25,258 across 12 columns (title\_english, rating, premiered, broadcast, producer, licensor, etc.).
* **Transformations**:

  * Log transformations on `members`, `favorites`, `episodes`, `scored_by`, etc.
  * Square-root transformations on `popularity` and `rank`.
* **Observations**: Many columns are right-skewed (popularity, episodes, favorites), suggesting wide variability in audience size and engagement.

---

### **4. Methodology**

#### **Step 1: Feature Engineering**

* Combined key **textual fields** (overview, tagline, genres, keywords for movies; synopsis, genre, themes for anime).
* Vectorized text data using **TF-IDF embeddings**.
* Normalized numerical features (budget, revenue, runtime, score, members).
* Constructed **content vectors** by concatenating text embeddings with scaled numerical features.

#### **Step 2: Clustering (Unsupervised)**

* Applied **K-Means clustering**.
* Determined optimal clusters via **Elbow Method** and **Silhouette Scores**.
* Movies: **2 clusters** offered best separation.
* Anime: PCA visualization showed **clear latent groupings** (Explained Variance: PC1 = 73.04%, PC2 = 2.86%).

#### **Step 3: Recommendation System**

* **Cluster Recommender**: Given a target item (e.g., "Vampire Hunter"), the system retrieves **similar items from the same cluster**.
* **Content-Based Filtering**: Using cosine similarity over TF-IDF vectors to recommend top-k similar titles.
* **Hybrid Function** (`hybrid_recommender_kmeans`): Combines both approaches — first identifies the **most relevant cluster**, then retrieves **content-similar titles** from within it.

---

### **5. Evaluation & Insights**

#### **Cluster Quality**

* Metrics: Silhouette Score, Davies-Bouldin Index confirmed well-separated clusters.
* PCA visualization showed strong variance explained in **movies & anime clusters**.

#### **Recommendations Example**

* Input Query: *"vampire"*

  * **Top 5 Movie Recommendations:** `The Dead Undead, Bled, Like Crazy, Alien Zone, The Mighty`
  * **Top 5 Anime Recommendations:** `Chikyuu to no Yakusoku, Watashitachi no Mirai, Ongeki, Gum Shaara, A Tang Qi Yu`

* Cluster-Based Recommendation:

  * Input: *"Vampire Hunter"*
  * Output: `Inu x Boku SS, Seto no Hanayome, Shugo Chara!! Doki, Princess Tutu, Bakuman. 3rd Season`

---

### **6. Business Value & Applications**

* **Streaming Platforms**: Personalized recommendations even for **new users (cold start problem)**.
* **Marketing Teams**: Ability to identify **customer clusters** and target them with tailored promotions.
* **Content Curation**: Insights into **latent themes & audience preferences** for new acquisitions.
* **Cross-Domain Synergy**: Recommending **anime to movie lovers** (and vice versa) by mapping shared clusters.

---

### **7. Conclusion**

This project demonstrates how **unsupervised learning + content-based filtering** can be combined into a **hybrid recommendation system** that is both scalable and effective.

* **Unsupervised clustering** revealed hidden content groupings.
* **Content similarity** ensured recommendations remain relevant to user queries.
* **Hybrid integration** allows personalization even when explicit ratings are absent.
