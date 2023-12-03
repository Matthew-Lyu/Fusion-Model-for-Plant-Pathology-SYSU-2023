import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

exp_number = 'exp8'

data = pd.read_csv(f'/home/hbenke/Project/Lvwc/Project/cv/leaf_disease_classifier/runs/test_result/{exp_number}/{exp_number}.txt', dtype=float)
y_pred = data.iloc[:, :6].values
y_true = data.iloc[:, 6:].values

conf_matrix = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))

plt.figure(figsize=(9, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='ocean_r', xticklabels=['healthy', 'scab', 'frog_eye_leaf_spot', 'rust', 'complex', 'powdery_mildew'], yticklabels=['healthy', 'scab', 'frog_eye_leaf_spot', 'rust', 'complex', 'powdery_mildew'])
plt.xlabel('pred')
plt.ylabel('label')
plt.title('conf_matrix')
plt.savefig(f'/home/hbenke/Project/Lvwc/Project/cv/leaf_disease_classifier/runs/test_result/{exp_number}/{exp_number}.png', dpi=1000)