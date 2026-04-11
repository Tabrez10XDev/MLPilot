from sklearn.datasets import make_classification
import pandas as pd

X, y = make_classification(
    n_samples=1200,
    n_features=8,
    n_informative=4,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    class_sep=1.2,
    flip_y=0.03,
    random_state=42,
)
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
df["target"] = y
df.to_csv("data/tree_data.csv", index=False)