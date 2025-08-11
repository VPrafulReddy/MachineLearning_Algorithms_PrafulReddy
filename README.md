1_logistic_regression.py

Logistic regression is a simple and effective machine learning method used to predict yes-or-no outcomes—like whether someone will buy a product, pass an exam, or click on an ad. It works by taking input data (such as age or hours studied), combining them with learned weights, and passing the result through a special math function called a sigmoid. This function squashes the output into a number between 0 and 1, which represents the probability of a “yes.” If the probability is above 0.5, it predicts “yes”; otherwise, it predicts “no.” For example, if someone studied for 7 hours, logistic regression might say there's a 90% chance they'll pass. It’s fast, easy to use, and great for making simple predictions.


#Output:

Accuracy: 1.00

Classification Report:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00         9
  versicolor       1.00      1.00      1.00        13
   virginica       1.00      1.00      1.00         8

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30

Sample Input: [[5.1, 3.5, 1.4, 0.2]]
Predicted Class: setosa


#The accuracy might not always be 1.00 — it depends on the train-test split.

#The prediction for the sample input [5.1, 3.5, 1.4, 0.2] will always be "setosa" because those measurements match that species closely.



2_decision_tree.py 

A decision tree is like a flowchart that helps a computer make decisions by asking a series of questions. Imagine you're trying to decide whether to go outside: the tree might first ask, “Is it raining?” If yes, then “Do you have an umbrella?”—and so on. In machine learning, it works the same way: it splits data into branches based on features (like age, income, etc.) and keeps asking questions until it reaches a final decision, like “yes” or “no.” It’s easy to understand, kind of like playing 20 Questions, and great for both simple and complex decision-making tasks.

3_knn.py

The K-Nearest Neighbors (KNN) algorithm is like asking your closest friends for advice. When you want to predict something—say, whether someone likes a movie—KNN looks at the “k” most similar people (neighbors) based on things like age, interests, or ratings. If most of them liked the movie, it assumes you probably will too. It doesn’t do any fancy calculations ahead of time; it just compares new data to existing examples and goes with the majority vote. Simple, intuitive, and surprisingly effective—just like good old peer pressure!

4_svm.py

Support Vector Machine (SVM) is like drawing the perfect line to separate two groups of things. Imagine you have a bunch of red and blue dots scattered on a paper, and you want to split them with a straight line so that all reds are on one side and blues on the other. SVM finds the **best possible line** (or even a curve, if needed) that not only separates the groups but also keeps the widest possible gap between them—like giving each team their own space. It’s great for classification tasks and works well even when the data isn’t perfectly clean or simple. Think of it as a super picky referee making sure both sides stay in their lanes.

5_random_forest.py

Random Forest is like asking a bunch of experts for their opinion and then going with the majority vote. Instead of relying on one decision tree (which might be biased or overconfident), it builds **many trees**, each trained on slightly different slices of the data. When it’s time to make a prediction—like whether someone will default on a loan—each tree gives its answer, and the forest picks the most common one. This teamwork makes Random Forest more accurate and reliable, kind of like crowd wisdom: one tree might make a mistake, but a whole forest? Much harder to fool.
