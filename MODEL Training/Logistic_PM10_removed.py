import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve

# ==========================================
# 0. CONFIGURATION
# ==========================================
sns.set_context("talk")
plt.rcParams.update({'font.size': 12})

# ==========================================
# 1. SETUP & LOAD DATA
# ==========================================
print("Loading data...")
train_df = pd.read_csv('final_train_data.csv')
test_df = pd.read_csv('final_test_data.csv')

# ==========================================
# 2. DEFINE FEATURES (NO PM10)
# ==========================================
feature_cols = ['PM2.5', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed']
target_col = 'AQI_Category'

X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_test = test_df[feature_cols]
y_test = test_df[target_col]

# Binarize for ROC/PR (Good vs Polluted)
y_test_binary = (y_test != 'Good').astype(int)

print(f"Features (PM10 REMOVED): {feature_cols}")
print(f"Training on {len(X_train)} samples. Testing on {len(X_test)} samples.")

# ==========================================
# 3. INITIALIZE MODELS
# ==========================================
models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000),
    "Decision_Tree": DecisionTreeClassifier(random_state=42),
    "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier()
}

# ==========================================
# 4. CREATE OUTPUT FOLDER
# ==========================================
output_folder = 'PM10_removed'
os.makedirs(output_folder, exist_ok=True)

# ==========================================
# 5. TRAIN, EVALUATE & SAVE RESULTS
# ==========================================
output_file = os.path.join(output_folder, 'model_evaluation_PM10_removed.txt')

print(f"\nStarting execution... Results will be saved to '{output_folder}'")

with open(output_file, 'w') as f:
    f.write("=======================================================\n")
    f.write("      AIR QUALITY MODEL EVALUATION (PM10 REMOVED)      \n")
    f.write("=======================================================\n\n")

    for name, model in models.items():
        print(f" -> Processing {name}...")
        
        # A. TRAIN & PREDICT
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Get Probabilities for ROC
        if hasattr(model, "predict_proba"):
            good_idx = np.where(model.classes_ == 'Good')[0][0]
            y_prob_pollution = 1 - model.predict_proba(X_test)[:, good_idx]
        else:
            y_prob_pollution = None

        # B. CALCULATE METRICS
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Generate report
        full_report = classification_report(y_test, y_pred, zero_division=0)
        
        # C. WRITE TO TEXT FILE
        f.write(f"--- {name.replace('_', ' ')} Results ---\n")
        f.write(f"Accuracy: {acc:.2%}\n")
        f.write(f"Weighted F1-Score: {f1:.4f}\n\n")
        f.write(full_report)
        f.write("\n" + "="*50 + "\n\n")
        
        # D. GENERATE DASHBOARD IMAGE
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        plt.suptitle(f"Model Performance: {name} (No PM10)", fontsize=20, weight='bold', y=0.98)
        
        # 1. Confusion Matrix
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[0, 0], cmap='Blues', normalize='true', colorbar=False)
        axes[0, 0].set_title("Confusion Matrix (Accuracy)", fontsize=14)
        
        # 2. Classification Report Heatmap
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report_dict).transpose().drop(columns=['support'])
        if 'accuracy' in report_df.index: report_df = report_df.drop('accuracy')
        sns.heatmap(report_df, annot=True, cmap='RdYlGn', fmt='.2f', vmin=0, vmax=1, ax=axes[0, 1], cbar=False)
        axes[0, 1].set_title("Precision, Recall & F1 Scores", fontsize=14)
        
        # 3. ROC Curve
        if y_prob_pollution is not None:
            fpr, tpr, _ = roc_curve(y_test_binary, y_prob_pollution)
            roc_auc = auc(fpr, tpr)
            axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
            axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[1, 0].set_xlabel('False Positive Rate')
            axes[1, 0].set_ylabel('True Positive Rate')
            axes[1, 0].set_title("ROC Curve (Detecting Pollution)", fontsize=14)
            axes[1, 0].legend(loc="lower right")
        else:
            axes[1, 0].text(0.5, 0.5, "ROC Not Available", ha='center')
            
        # 4. PR Curve
        if y_prob_pollution is not None:
            precision, recall, _ = precision_recall_curve(y_test_binary, y_prob_pollution)
            axes[1, 1].plot(recall, precision, color='purple', lw=2)
            axes[1, 1].set_xlabel('Recall')
            axes[1, 1].set_ylabel('Precision')
            axes[1, 1].set_title("Precision-Recall Curve", fontsize=14)
        else:
            axes[1, 1].text(0.5, 0.5, "PR Curve Not Available", ha='center')

        # Save Image
        img_name = f"dashboard_{name}.png"
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_folder, img_name))
        plt.close()

print(f"\nSuccess! Results and images saved in '{output_folder}'.")
