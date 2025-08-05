import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


y_true = np.load("labels.npy")
y_pred = np.load("preds.npy")

classes = ["NN", "YN", "YY"]
labels = [0, 1, 2]

# REPORT
report = classification_report(
    y_true, y_pred,
    labels=labels,
    target_names=classes,
    zero_division=0
)
print(report)


with open("eval_report.txt", "w") as f:
    f.write(report)

# confusion MATRIX
cm = confusion_matrix(y_true, y_pred, labels=labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("eval_confusion_matrix.png")
plt.close()
print("save to eval_report.txt and eval_confusion_matrix.png")
