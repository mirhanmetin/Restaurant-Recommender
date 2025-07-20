Medium Blog: https://medium.com/@sadi.akdemir/%EF%B8%8F-personalized-restaurant-recommendation-system-78a825dc970a

🍽️ Personalized Restaurant Recommendation System — Sadi Akdemir & Yusuf Mirhan Metin

In a world overwhelmed by choices, users often spend too much time deciding where to eat, drink, or shop. This project introduces a Personalized Restaurant Recommendation System designed to reduce decision fatigue by offering tailored suggestions based on user preferences, similar user behavior, and food category interest.

🧠 Problem Definition

Main challenge: Users waste time searching for suitable restaurants among thousands of options.

Goal: Deliver accurate and personalized restaurant suggestions based on:

-Similar restaurants to a user’s highly-rated restaurants (content-based filtering)

-Similar users’ preferences (collaborative filtering)

-Category-based search (manual input)

-And the combination of all above (hybrid model)

📊 Dataset Information

Source: Yelp Open Dataset

Data Preprocessing:

-Businesses filtered by category "Restaurant"

-Irrelevant columns dropped and null values handled

-Review set reduced to the top 1,000 most active users for performance

-Each user was assigned a random American-style full name

-Numerical attributes normalized (stars, review_count)

-Textual category data encoded using TF-IDF

Processed Files Used:

File NameDescriptioncleaned_restaurants.csvRestaurant info (name, categories, rating)review_top_1000_users_named_cleaned.jsonReviews by top 1000 users with usernames

🧪 Recommendation Techniques Used
We implemented three recommendation approaches and an extra search function:

1. 📘 Content-Based Filtering (CBF)

-Recommends restaurants similar to those the user liked before

-Based on restaurant categories + ratings/review counts

-Categories vectorized using TF-IDF

-Cosine similarity used to find closeness between vectors

2. 👥 Collaborative Filtering (CF)

-Uses SVD matrix factorization from the surprise library

-Learns from ratings by similar users to predict preferences

-User–item interaction matrix built from rating data

3. 🧠 Hybrid Model

-Weighted combination of CF and CBF scores:

-Hybrid Score = 0.5 CF Score + 0.5 CBF Score

-Offers better performance in sparse data situations and cold-start mitigation

4. 🔍 Category Search Feature

-Allows users to manually enter a category like "pizza" or "sushi"

-Returns the top 5 highest-rated restaurants matching the keyword

💻 Technologies Used

Tool / LibraryPurposePythonCore programming languagepandas, numpyData preprocessing and handlingscikit-learnTF-IDF vectorizer & normalizationsurpriseCollaborative Filtering (SVD)GradioInteractive UI for the userHugging Face SpacesPublic deployment

🌐 You can test the full system (with all three models + category search) at the link below

🔗 Live Demo: https://huggingface.co/spaces/mirhanmetin/Personalized_Restaurant_Recommendation_System

Enter a full name such as:

James Smith /
Mary Johnson /
Michael Brown /
Emily Davis /
Robert Miller /

Or search with categories like:

"bakery", "pizza", "vegan", "cafe"

👥 Team Members

Sadi Akdemir:

Data preprocessing, content-based filtering, Gradio/Hugging Face integration, documentation

Yusuf Mirhan Metin:

Collaborative filtering, hybrid scoring logic, UI optimization, evaluation metrics

📈 Evaluation & Observations

CF performance was measured using RMSE and MAE from 5-fold cross-validation

CBF and Hybrid results were evaluated through user feedback and test runs

The Hybrid model consistently outperformed single models in balancing relevance and novelty

🎓 Conclusion

This project demonstrates how intelligent recommendation systems can improve user decision-making in crowded domains like food and hospitality.

✅ CBF brings personalization through familiarity

✅ CF discovers hidden gems by learning from others

✅ Hybrid offers the best of both worlds

✅ Category search helps with direct exploration

We developed a fully-functional, interactive system accessible to everyone via Hugging Face Spaces.

🔗 Resources

-Yelp Open Dataset

-Surprise Library for Recommender Systems

-Gradio Docs

-Hugging Face Spaces
