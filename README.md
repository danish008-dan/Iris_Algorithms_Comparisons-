Iris Dataset Classification – Machine Learning Project
📘 Project Overview

This project demonstrates the use of multiple machine learning algorithms to classify iris flowers into their respective species — Iris-setosa, Iris-versicolor, and Iris-virginica.
The dataset used is the classic Iris dataset, which contains 150 samples with four features:

Sepal Length

Sepal Width

Petal Length

Petal Width

The main goal is to compare the performance of different classification algorithms on the same dataset.

🤖 Algorithms Used

Logistic Regression

K-Nearest Neighbors (KNN)

Decision Tree Classifier

Random Forest Classifier

These algorithms represent a mix of linear, instance-based, and tree-based models — allowing comparison between linear and non-linear classifiers.

⚙️ Tech Stack / Libraries

Python 🐍

Pandas

NumPy

Scikit-learn

Matplotlib / Seaborn (optional for visualization)

📂 Project Structure
├── iris_classification.py   # Main source code
├── Iris_Classification_Report.pdf  # Project report summary
├── Iris.csv                 # Dataset (if included)
└── README.md                # Project documentation

🚀 How to Run the Project

Clone the repository:

git clone https://github.com/yourusername/Iris-Classification-Project.git
cd Iris-Classification-Project


Install dependencies:

pip install -r requirements.txt


(or manually install: pip install pandas scikit-learn numpy matplotlib)

Run the script:

python iris_classification.py

📊 Results

All models were trained and tested on a 70/30 split of the dataset.

Algorithm	Accuracy
Logistic Regression	97%
KNN	100%
Decision Tree	97%
Random Forest	100%

Note: Accuracy may slightly vary depending on random state and dataset split.

🧠 Key Insights

The Iris dataset is simple and linearly separable, so even basic models perform extremely well.

Random Forest provides the best generalization due to its ensemble nature.

Comparing multiple models helps understand how different algorithms behave on the same data.

📄 Project Report

A detailed project summary and comparison of model performances can be found in the PDF report:
Iris_Classification_Report.pdf

💬 Author

👤 Danish Khatri
📧 danishkhatri885@gmail.com
🔗 LinkedIn Profile : https://www.linkedin.com/in/danish-khatri-72543b381/

🏷️ Tags

#MachineLearning #DataScience #Python #IrisDataset #MLProjects
