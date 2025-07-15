import os
import json

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)

from src.predict import (
    predict_lr_test,
    predict_lgbm_test,
)
from src.train import y_test
from src.evaluate import y_scores_lr_test, y_scores_lgbm_fraud_test



def run(pred_lr=predict_lr_test(), pred_lgbm=predict_lgbm_test()):


    # Confusion Matrix
    cm_lr = confusion_matrix(y_test, pred_lr)
    cm_lgbm = confusion_matrix(y_test, pred_lgbm)

    def save_confusion_matrix(cm, labels, name):
        path = f'outputs/plots/confusion_matrix_{name}.png'
        display = ConfusionMatrixDisplay(confusion_matrix=cm,
                                        display_labels=labels)
        display.plot()
        plt.title(f'{name} Confusion Matrix')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close()

    save_confusion_matrix(cm=cm_lr, labels=['Valid', 'Fraud'], name='lr')
    save_confusion_matrix(cm=cm_lgbm, labels=['Valid', 'Fraud'], name='lgbm')

    print('Confusion Matrix Saved.')



    # Metrics
    metrics_lr = {
        "accuracy": accuracy_score(y_test, pred_lr),
        "precision": precision_score(y_test, pred_lr),
        "recall": recall_score(y_test, pred_lr),
        "f1_score": f1_score(y_test, pred_lr),
        "roc_auc": roc_auc_score(y_test, y_score=y_scores_lr_test),
    }

    metrics_lgbm = {
        "accuracy": accuracy_score(y_test, pred_lgbm),
        "precision": precision_score(y_test, pred_lgbm),
        "recall": recall_score(y_test, pred_lgbm),
        "f1_score": f1_score(y_test, pred_lgbm),
        "roc_auc": roc_auc_score(y_test, y_score=y_scores_lgbm_fraud_test),
    }

    with open('outputs/metrics_lr.json', 'w') as f:
        json.dump(metrics_lr, f, indent=4)

    with open('outputs/metrics_lgbm.json', 'w') as f:
        json.dump(metrics_lgbm, f, indent=4)
    print('Metrics Saved.')

    # Comparison Bars
    def save_bar_comparison_plot(values, labels, name, ylabel, title='Score'):
        os.makedirs('outputs/plots', exist_ok=True)
        plt.figure()
        plt.bar(labels, values, color=['royalblue', 'gray'])
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(f'outputs/plots/{name}.png')
        plt.close()

    # Accuracy
    save_bar_comparison_plot(
        values=[
            accuracy_score(y_test, pred_lr),
            accuracy_score(y_test, pred_lgbm)
        ],
        labels=['LR', 'LGBM'],
        title='Accuracy Score Comparison',
        name='accuracy_score_comparison',
        ylabel='Accuracy'
    )
    # F1 Score
    save_bar_comparison_plot(
        values=[
            f1_score(y_test, pred_lr),
            f1_score(y_test, pred_lgbm)
        ],
        labels=['LR', 'LGBM'],
        title='F1 Score Comparison',
        name='f1_score_comparison',
        ylabel='F1 Score'
    )
    print('Accuracy and F1 Score Diagrams Saved.')


    # Recall_Precision Diagram
    precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_scores_lr_test)
    precision_lgbm, recall_lgbm, _ = precision_recall_curve(y_test, y_scores_lgbm_fraud_test)

    models = ([
        ('LR', recall_lr, precision_lr),
        ('LGBM', recall_lgbm, precision_lgbm)
    ])
    plt.figure()
    os.makedirs(f'outputs/plots', exist_ok=True)
    for model_name, recall, precision in models:
        plt.plot(recall, precision, label=model_name)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall Comparison')
    plt.grid(True)
    plt.legend()
    filename='precision_vs_recall_comparison'
    plt.savefig(f'outputs/plots/{filename}.png')
    plt.close()
    print('Recall_Precision Diagram Saved.')


    # roc_curve Diagram
    fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_score=y_scores_lr_test)
    fpr_lgbm, tpr_lgbm, thresholds_lgbm = roc_curve(y_test, y_score=y_scores_lgbm_fraud_test)

    plt.plot(fpr_lr, tpr_lr, label='LR', color='royalblue')
    plt.plot(fpr_lgbm, tpr_lgbm, label='LGBM', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig('outputs/plots/roc_curve_comparison.png')
    plt.close()
    print('ROC_Curve Diagram Saved.')
    print('Visualization has been done.')