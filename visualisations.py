import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, ConfusionMatrixDisplay
from sklearn.tree import plot_tree


# ── Colour palette ────────────────────────────────────────────────────────────
GOOD_COL = "#2196F3"
BAD_COL  = "#F44336"
PALETTE  = [GOOD_COL, BAD_COL]


# ══════════════════════════════════════════════════════════════════════════════
# 1. ORIGINAL DATA VISUALS
# ══════════════════════════════════════════════════════════════════════════════

def plot_original_data(df: pd.DataFrame):
    """
    Four-panel overview of the raw German Credit dataset.
      • Class balance (bar)
      • Age distribution by class (histogram)
      • Credit amount distribution by class (histogram)
      • Correlation heatmap (numerical features)
    """
    y = df['class'].map({1: 'good', 2: 'bad'})
    numerical_cols = ['duration', 'credit_amount', 'installment_rate',
                      'residence_since', 'age', 'existing_credits', 'dependents']

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("German Credit Dataset – Exploratory Overview", fontsize=16, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel 1: class balance ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    counts = y.value_counts()
    bars = ax1.bar(counts.index, counts.values, color=PALETTE, edgecolor='white', width=0.5)
    ax1.set_title("Class Balance\n(900 train + 100 test = 1,000 total)", fontsize=11)
    ax1.set_xlabel("Credit Rating")
    ax1.set_ylabel("Count")
    for bar, val in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                 str(val), ha='center', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, max(counts.values) * 1.15)

    # ── Panel 2: age distribution by class ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for label, col in zip(['good', 'bad'], PALETTE):
        ax2.hist(df.loc[y == label, 'age'], bins=20, alpha=0.65,
                 color=col, label=label, edgecolor='white')
    ax2.set_title("Age Distribution by Class", fontsize=11)
    ax2.set_xlabel("Age (years)")
    ax2.set_ylabel("Count")
    ax2.legend()

    # ── Panel 3: credit amount by class ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for label, col in zip(['good', 'bad'], PALETTE):
        ax3.hist(df.loc[y == label, 'credit_amount'], bins=25, alpha=0.65,
                 color=col, label=label, edgecolor='white')
    ax3.set_title("Credit Amount Distribution by Class", fontsize=11)
    ax3.set_xlabel("Credit Amount (DM)")
    ax3.set_ylabel("Count")
    ax3.legend()

    # ── Panel 4: correlation heatmap ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    corr = df[numerical_cols].corr()
    sns.heatmap(corr, ax=ax4, annot=True, fmt=".2f", cmap='coolwarm',
                center=0, linewidths=0.5, annot_kws={"size": 8},
                cbar_kws={"shrink": 0.8})
    ax4.set_title("Numerical Feature Correlations", fontsize=11)
    ax4.tick_params(axis='x', rotation=45, labelsize=8)
    ax4.tick_params(axis='y', rotation=0,  labelsize=8)

    plt.savefig("plot_original_data.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: plot_original_data.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. DECISION TREE VISUALS
# ══════════════════════════════════════════════════════════════════════════════

def plot_decision_tree_results(model, X_test_processed, y_test, feature_names):
    """
    Two-panel figure:
      • Confusion matrix
      • ROC curve
    """
    y_pred  = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Decision Tree – Evaluation", fontsize=14, fontweight='bold')

    # Confusion matrix
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                           display_labels=['good', 'bad']).plot(ax=axes[0], colorbar=False,
                                                                 cmap='Blues')
    axes[0].set_title("Confusion Matrix")

    # ROC curve
    axes[1].plot(fpr, tpr, color='#4CAF50', lw=2, label=f"AUC = {auc:.3f}")
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig("plot_decision_tree.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: plot_decision_tree.png")


def plot_decision_tree_structure(model, feature_names):
    """
    Plot the trained decision tree structure (up to depth 4 for readability).
    """
    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(model, feature_names=feature_names, class_names=['good', 'bad'],
              filled=True, rounded=True, fontsize=9, max_depth=4, ax=ax)
    ax.set_title("Decision Tree Structure (max display depth = 4)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("plot_decision_tree_structure.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: plot_decision_tree_structure.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. SVM VISUALS
# ══════════════════════════════════════════════════════════════════════════════

def plot_svm_results(model, X_test_processed, y_test):
    """
    Two-panel figure:
      • Confusion matrix
      • ROC curve
    """
    y_pred  = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("SVM (RBF Kernel) – Evaluation", fontsize=14, fontweight='bold')

    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                           display_labels=['good', 'bad']).plot(ax=axes[0], colorbar=False,
                                                                 cmap='Oranges')
    axes[0].set_title("Confusion Matrix")

    axes[1].plot(fpr, tpr, color='#FF9800', lw=2, label=f"AUC = {auc:.3f}")
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig("plot_svm.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: plot_svm.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. KNN VISUALS
# ══════════════════════════════════════════════════════════════════════════════

def plot_knn_results(model, X_train_processed, y_train, X_test_processed, y_test):
    """
    Three-panel figure:
      • k vs CV accuracy
      • Confusion matrix
      • ROC curve
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier

    k_range = range(1, 21)
    cv_scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k),
                                 X_train_processed, y_train,
                                 cv=5, scoring='accuracy').mean()
                 for k in k_range]

    y_pred  = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    best_k = model.n_neighbors

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("K-Nearest Neighbours – Evaluation", fontsize=14, fontweight='bold')

    # k vs accuracy
    axes[0].plot(list(k_range), cv_scores, marker='o', lw=2, color='#9C27B0')
    axes[0].axvline(best_k, color='red', linestyle='--', label=f"Best k = {best_k}")
    axes[0].set_xlabel("k (number of neighbours)")
    axes[0].set_ylabel("5-Fold CV Accuracy")
    axes[0].set_title("Accuracy vs k")
    axes[0].legend()

    # Confusion matrix
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                           display_labels=['good', 'bad']).plot(ax=axes[1], colorbar=False,
                                                                 cmap='Purples')
    axes[1].set_title("Confusion Matrix")

    # ROC curve
    axes[2].plot(fpr, tpr, color='#9C27B0', lw=2, label=f"AUC = {auc:.3f}")
    axes[2].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("True Positive Rate")
    axes[2].set_title("ROC Curve")
    axes[2].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig("plot_knn.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: plot_knn.png")