# # # import pandas as pd
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # import seaborn as sns
# # # import os

# # # # Machine Learning Libraries
# # # from sklearn.linear_model import LogisticRegression
# # # from sklearn.tree import DecisionTreeClassifier
# # # from sklearn.ensemble import RandomForestClassifier
# # # from sklearn.svm import SVC
# # # from sklearn.neighbors import KNeighborsClassifier
# # # from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score, f1_score

# # # # ==========================================
# # # # 1. SETUP & LOAD DATA
# # # # ==========================================
# # # print("Loading data...")
# # # train_df = pd.read_csv('final_train_data.csv')
# # # test_df = pd.read_csv('final_test_data.csv')

# # # # ==========================================
# # # # 2. DEFINE FEATURES (NO PM2.5)
# # # # ==========================================
# # # feature_cols = ['PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed']
# # # target_col = 'AQI_Category'

# # # X_train = train_df[feature_cols]
# # # y_train = train_df[target_col]

# # # X_test = test_df[feature_cols]
# # # y_test = test_df[target_col]

# # # print(f"Training on {len(X_train)} samples. Testing on {len(X_test)} samples.")

# # # # ==========================================
# # # # 3. INITIALIZE MODELS
# # # # ==========================================
# # # models = {
# # #     "Logistic_Regression": LogisticRegression(max_iter=1000),
# # #     "Decision_Tree": DecisionTreeClassifier(random_state=42),
# # #     "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
# # #     "SVM": SVC(),
# # #     "KNN": KNeighborsClassifier()
# # # }

# # # # ==========================================
# # # # 4. TRAIN, EVALUATE & SAVE TO TXT
# # # # ==========================================
# # # output_file = 'model_evaluation_summary.txt'

# # # print(f"\nStarting execution... Results will be saved to '{output_file}'")

# # # # Open the text file once to write all results
# # # with open(output_file, 'w') as f:
# # #     f.write("==============================================\n")
# # #     f.write("      AIR QUALITY MODEL EVALUATION REPORT     \n")
# # #     f.write("==============================================\n\n")

# # #     for name, model in models.items():
# # #         print(f" -> Processing {name}...")
        
# # #         # A. TRAIN & PREDICT
# # #         model.fit(X_train, y_train)
# # #         y_pred = model.predict(X_test)
        
# # #         # B. CALCULATE METRICS
# # #         acc = accuracy_score(y_test, y_pred)
# # #         f1 = f1_score(y_test, y_pred, average='weighted')
        
# # #         # Generate the full detailed report text
# # #         full_report = classification_report(y_test, y_pred, zero_division=0)
        
# # #         # C. WRITE TO TEXT FILE
# # #         f.write(f"--- {name.replace('_', ' ')} Results ---\n")
# # #         f.write(f"Accuracy: {acc:.2%}\n")
# # #         f.write(f"Weighted F1-Score: {f1:.4f}\n\n")
# # #         f.write(full_report)
# # #         f.write("\n" + "="*50 + "\n\n") # Separator line
        
# # #         # D. GENERATE IMAGE 1: CONFUSION MATRIX
# # #         fig, ax = plt.subplots(figsize=(6, 5))
# # #         ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues', normalize='true')
# # #         plt.title(f'Confusion Matrix: {name}')
# # #         plt.tight_layout()
# # #         plt.savefig(f'confusion_matrix_{name}.png')
# # #         plt.close()
        
# # #         # E. GENERATE IMAGE 2: HEATMAP OF CLASSIFICATION REPORT
# # #         report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
# # #         report_df = pd.DataFrame(report_dict).transpose()
# # #         if 'accuracy' in report_df.index: report_df = report_df.drop('accuracy')
# # #         if 'support' in report_df.columns: report_df = report_df.drop(columns=['support'])
            
# # #         fig, ax = plt.subplots(figsize=(8, 6))
# # #         sns.heatmap(report_df, annot=True, cmap='RdYlGn', fmt='.2f', vmin=0, vmax=1)
# # #         plt.title(f'Performance Heatmap: {name}')
# # #         plt.tight_layout()
# # #         plt.savefig(f'heatmap_{name}.png')
# # #         plt.close()

# # # print(f"\nSuccess! Check '{output_file}' for the detailed text report.")
# # # print("Check your folder for the 10 PNG images.")
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.tree import DecisionTreeClassifier
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.svm import SVC
# # from sklearn.neighbors import KNeighborsClassifier
# # from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
# # from sklearn.preprocessing import label_binarize

# # # ==========================================
# # # 1. LOAD DATA
# # # ==========================================
# # print("Loading data...")
# # train_df = pd.read_csv('final_train_data.csv')
# # test_df = pd.read_csv('final_test_data.csv')

# # feature_cols = ['PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed']
# # target_col = 'AQI_Category'

# # X_train = train_df[feature_cols]
# # y_train = train_df[target_col]
# # X_test = test_df[feature_cols]
# # y_test = test_df[target_col]

# # # Binarize the target for ROC/PR Curves (We treat "Good" as Class 0, Everything else as Class 1)
# # # This makes the curves easier to understand: "Can the model detect Pollution?"
# # y_test_binary = (y_test != 'Good').astype(int)

# # # ==========================================
# # # 2. DEFINE MODELS
# # # ==========================================
# # # Note: SVM needs probability=True to draw ROC curves
# # models = {
# #     "Logistic_Regression": LogisticRegression(max_iter=1000),
# #     "Decision_Tree": DecisionTreeClassifier(random_state=42),
# #     "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
# #     "SVM": SVC(probability=True, random_state=42),
# #     "KNN": KNeighborsClassifier()
# # }

# # # ==========================================
# # # 3. GENERATE 5 DASHBOARD IMAGES
# # # ==========================================
# # print("\nGenerating 5 Dashboard Images (This may take a minute)...")

# # for name, model in models.items():
# #     print(f" -> Creating Dashboard for {name}...")
    
# #     # Train
# #     model.fit(X_train, y_train)
    
# #     # Predict (Categories)
# #     y_pred = model.predict(X_test)
    
# #     # Predict (Probabilities for ROC/PR) - Probability of NOT being 'Good'
# #     # We find the index of the 'Good' class to get the inverse probability
# #     good_idx = np.where(model.classes_ == 'Good')[0][0]
# #     # Probability of Pollution = 1 - Probability of Good
# #     y_prob_pollution = 1 - model.predict_proba(X_test)[:, good_idx]

# #     # --- CREATE FIGURE WITH 4 SUBPLOTS ---
# #     fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# #     plt.suptitle(f"Model Performance Dashboard: {name}", fontsize=20, weight='bold')
    
# #     # 1. TOP-LEFT: CONFUSION MATRIX
# #     ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[0, 0], cmap='Blues', normalize='true', colorbar=False)
# #     axes[0, 0].set_title("Confusion Matrix (Accuracy)", fontsize=14)
    
# #     # 2. TOP-RIGHT: CLASSIFICATION REPORT HEATMAP
# #     report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
# #     report_df = pd.DataFrame(report_dict).transpose().drop(columns=['support'])
# #     if 'accuracy' in report_df.index: report_df = report_df.drop('accuracy')
# #     sns.heatmap(report_df, annot=True, cmap='RdYlGn', fmt='.2f', vmin=0, vmax=1, ax=axes[0, 1])
# #     axes[0, 1].set_title("Precision, Recall & F1 Scores", fontsize=14)
    
# #     # 3. BOTTOM-LEFT: ROC CURVE (Detection of Pollution)
# #     fpr, tpr, _ = roc_curve(y_test_binary, y_prob_pollution)
# #     roc_auc = auc(fpr, tpr)
# #     axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
# #     axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# #     axes[1, 0].set_xlabel('False Alarm Rate')
# #     axes[1, 0].set_ylabel('Detection Rate')
# #     axes[1, 0].set_title("ROC Curve (Detecting Pollution)", fontsize=14)
# #     axes[1, 0].legend(loc="lower right")
# #     axes[1, 0].grid(True, alpha=0.3)
    
# #     # 4. BOTTOM-RIGHT: PRECISION-RECALL CURVE
# #     precision, recall, _ = precision_recall_curve(y_test_binary, y_prob_pollution)
# #     axes[1, 1].plot(recall, precision, color='purple', lw=2)
# #     axes[1, 1].set_xlabel('Recall (Sensitivity)')
# #     axes[1, 1].set_ylabel('Precision')
# #     axes[1, 1].set_title("Precision-Recall Curve (Pollution)", fontsize=14)
# #     axes[1, 1].grid(True, alpha=0.3)
    
# #     # Save
# #     filename = f"dashboard_{name}.png"
# #     plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Make space for title
# #     plt.savefig(filename)
# #     plt.close()
# #     print(f"    Saved {filename}")

# # print("\nDone! You now have 5 Dashboard images (dashboard_*.png).")
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
# from sklearn.preprocessing import label_binarize

# # ==========================================
# # 0. CONFIGURATION FOR HIGH RESOLUTION
# # ==========================================
# # This sets the global font scale to be readable in large images
# sns.set_context("talk") 
# plt.rcParams.update({'font.size': 12}) 

# # ==========================================
# # 1. LOAD DATA
# # ==========================================
# print("Loading data...")
# train_df = pd.read_csv('final_train_data.csv')
# test_df = pd.read_csv('final_test_data.csv')

# feature_cols = ['PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed']
# target_col = 'AQI_Category'

# X_train = train_df[feature_cols]
# y_train = train_df[target_col]
# X_test = test_df[feature_cols]
# y_test = test_df[target_col]

# # Binarize for ROC/PR (Good vs Polluted)
# y_test_binary = (y_test != 'Good').astype(int)

# # ==========================================
# # 2. DEFINE MODELS
# # ==========================================
# models = {
#     "Logistic_Regression": LogisticRegression(max_iter=1000),
#     "Decision_Tree": DecisionTreeClassifier(random_state=42),
#     "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
#     "SVM": SVC(probability=True, random_state=42),
#     "KNN": KNeighborsClassifier()
# }

# # ==========================================
# # 3. GENERATE HIGH-RES DASHBOARDS
# # ==========================================
# print("\nGenerating High-Resolution Dashboard Images...")

# for name, model in models.items():
#     print(f" -> Creating Dashboard for {name}...")
    
#     # Train & Predict
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
    
#     # Get Probabilities for ROC (Inverse of 'Good' probability)
#     good_idx = np.where(model.classes_ == 'Good')[0][0]
#     y_prob_pollution = 1 - model.predict_proba(X_test)[:, good_idx]

#     # --- CREATE FIGURE (Large Size for Quality) ---
#     fig, axes = plt.subplots(2, 2, figsize=(20, 16)) # Increased size slightly
    
#     # Main Title
#     plt.suptitle(f"Model Performance Dashboard: {name.replace('_', ' ')}", fontsize=24, weight='bold', y=0.98)
    
#     # 1. TOP-LEFT: CONFUSION MATRIX
#     ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[0, 0], cmap='Blues', normalize='true', colorbar=False)
#     axes[0, 0].set_title("Confusion Matrix (Accuracy)", fontsize=18, pad=15)
#     axes[0, 0].grid(False) # Turn off grid for CM
    
#     # 2. TOP-RIGHT: CLASSIFICATION REPORT HEATMAP
#     report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
#     report_df = pd.DataFrame(report_dict).transpose().drop(columns=['support'])
#     if 'accuracy' in report_df.index: report_df = report_df.drop('accuracy')
    
#     sns.heatmap(report_df, annot=True, cmap='RdYlGn', fmt='.2f', vmin=0, vmax=1, ax=axes[0, 1], cbar=False, annot_kws={"size": 14})
#     axes[0, 1].set_title("Precision, Recall & F1 Scores", fontsize=18, pad=15)
#     axes[0, 1].set_yticks(axes[0, 1].get_yticks()) # Fix ticker warning
#     axes[0, 1].set_yticklabels(axes[0, 1].get_yticklabels(), rotation=0)
    
#     # 3. BOTTOM-LEFT: ROC CURVE
#     fpr, tpr, _ = roc_curve(y_test_binary, y_prob_pollution)
#     roc_auc = auc(fpr, tpr)
#     axes[1, 0].plot(fpr, tpr, color='darkorange', lw=3, label=f'AUC = {roc_auc:.2f}')
#     axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
#     axes[1, 0].set_xlabel('False Positive Rate', fontsize=16)
#     axes[1, 0].set_ylabel('True Positive Rate', fontsize=16)
#     axes[1, 0].set_title("ROC Curve (Detecting Pollution)", fontsize=18, pad=15)
#     axes[1, 0].legend(loc="lower right", fontsize=14)
#     axes[1, 0].grid(True, alpha=0.3)
    
#     # 4. BOTTOM-RIGHT: PRECISION-RECALL CURVE
#     precision, recall, _ = precision_recall_curve(y_test_binary, y_prob_pollution)
#     axes[1, 1].plot(recall, precision, color='purple', lw=3)
#     axes[1, 1].set_xlabel('Recall (Sensitivity)', fontsize=16)
#     axes[1, 1].set_ylabel('Precision', fontsize=16)
#     axes[1, 1].set_title("Precision-Recall Curve (Pollution)", fontsize=18, pad=15)
#     axes[1, 1].grid(True, alpha=0.3)
    
#     # SAVE WITH HIGH DPI
#     filename = f"dashboard_{name}_HighRes.png"
#     # bbox_inches='tight' ensures no text is cut off
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
#     plt.savefig(filename, dpi=300, bbox_inches='tight') 
#     plt.close()
    
#     print(f"    Saved {filename} (High Quality)")

# print("\nDone! Check for files ending in '_HighRes.png'.")

###With feature pm2.5


import pandas as pd
import numpy as np
import os

# Machine Learning Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

# ==========================================
# 1. SETUP & LOAD DATA
# ==========================================
print("Loading data...")
train_df = pd.read_csv('final_train_data.csv')
test_df = pd.read_csv('final_test_data.csv')

# ==========================================
# 2. DEFINE FEATURES (WITH PM2.5)
# ==========================================
# We include PM2.5 this time to see the high accuracy
feature_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed']
target_col = 'AQI_Category'

X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_test = test_df[feature_cols]
y_test = test_df[target_col]

print(f"Features: {feature_cols}")
print(f"Training on {len(X_train)} samples. Testing on {len(X_test)} samples.")

# ==========================================
# 3. INITIALIZE MODELS
# ==========================================
models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000),
    "Decision_Tree": DecisionTreeClassifier(random_state=42),
    "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

# ==========================================
# 4. TRAIN, EVALUATE & SAVE TO TXT
# ==========================================
output_file = 'model_evaluation_summary_WITH_PM25.txt'

print(f"\nStarting execution... Results will be saved to '{output_file}'")

with open(output_file, 'w') as f:
    f.write("=======================================================\n")
    f.write("      AIR QUALITY MODEL EVALUATION (WITH PM2.5)        \n")
    f.write("=======================================================\n\n")

    for name, model in models.items():
        print(f" -> Processing {name}...")
        
        # A. TRAIN & PREDICT
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # B. CALCULATE METRICS
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Generate the full detailed report text
        full_report = classification_report(y_test, y_pred, zero_division=0)
        
        # C. WRITE TO TEXT FILE
        f.write(f"--- {name.replace('_', ' ')} Results ---\n")
        f.write(f"Accuracy: {acc:.2%}\n")
        f.write(f"Weighted F1-Score: {f1:.4f}\n\n")
        f.write(full_report)
        f.write("\n" + "="*50 + "\n\n")

print(f"\nSuccess! Open '{output_file}' to see the high accuracy results.")