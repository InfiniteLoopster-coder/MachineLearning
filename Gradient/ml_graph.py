import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Create directed graph
G = nx.DiGraph()

# Pipeline nodes and edges
pipeline = ["Data Collection", "Data Preprocessing", "Feature Engineering", "Model Training", 
            "Model Evaluation", "Hyperparameter Tuning", "Model Deployment"]
G.add_edge("Machine Learning", "Data Collection")
for u, v in zip(pipeline[:-1], pipeline[1:]):
    G.add_edge(u, v)

# Taxonomy under Model Training
categories = {
    "Supervised Learning": [
        "Linear Regression", "Logistic Regression", "Decision Tree", "Random Forest", 
        "Support Vector Machine", "K-Nearest Neighbors", "Naive Bayes", 
        "Gradient Boosting", "XGBoost", "LightGBM"
    ],
    "Unsupervised Learning": [
        "K-Means", "Hierarchical Clustering", "DBSCAN", 
        "Principal Component Analysis", "Gaussian Mixture Model", "Autoencoder"
    ],
    "Reinforcement Learning": [
        "Q-Learning", "SARSA", "Deep Q-Network", "Policy Gradients", "Proximal Policy Optimization"
    ],
    "Semi-Supervised Learning": ["Label Propagation", "Self-Training"],
    "Deep Learning": [
        "Convolutional Neural Network", "Recurrent Neural Network", 
        "Long Short-Term Memory", "Gated Recurrent Unit", "Transformer", "Generative Adversarial Network"
    ]
}

# Add taxonomy edges
for cat, algos in categories.items():
    G.add_edge("Model Training", cat)
    for algo in algos:
        G.add_edge(cat, algo)

# Position nodes manually by level
pos = {}
pos["Machine Learning"] = (0, 0)

# Level 1: pipeline nodes
x1 = np.linspace(-6, 6, len(pipeline))
for x, node in zip(x1, pipeline):
    pos[node] = (x, -1)

# Level 2: categories
x2 = np.linspace(-4, 4, len(categories))
for x, node in zip(x2, categories.keys()):
    pos[node] = (x, -2)

# Level 3: algorithms
for cat, algos in categories.items():
    x_parent, y_parent = pos[cat]
    N = len(algos)
    span = max(N / 2, 1)
    x3 = np.linspace(x_parent - span, x_parent + span, N)
    for x, algo in zip(x3, algos):
        pos[algo] = (x, -3)

# Draw graph
plt.figure(figsize=(16, 12))
nx.draw(G, pos, with_labels=True, arrows=True, node_size=2500, font_size=8)
plt.title("Machine Learning Workflow and Algorithm Taxonomy")
plt.axis('off')
plt.show()
