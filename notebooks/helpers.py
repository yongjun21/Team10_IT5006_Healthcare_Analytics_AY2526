from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc
)
import time


def evaluate_model(get_model, train_X, train_y, test_X, test_y, get_decision_score=None):
    model = get_model()

    start_time = time.time()
    model.fit(train_X, train_y)
    training_time = time.time() - start_time

    train_pred = model.predict(train_X)
    test_pred = model.predict(test_X)

    results = {
        "model": model,
        "train_pred": train_pred,
        "test_pred": test_pred,
        "train_accuracy": accuracy_score(train_y, train_pred),
        "test_accuracy": accuracy_score(test_y, test_pred),
        "train_precision_score": precision_score(train_y, train_pred),
        "test_precision_score": precision_score(test_y, test_pred),
        "train_recall_score": recall_score(train_y, train_pred),
        "test_recall_score": recall_score(test_y, test_pred),
        "train_f1": f1_score(train_y, train_pred),
        "test_f1": f1_score(test_y, test_pred),
        "train_cm": confusion_matrix(train_y, train_pred),
        "test_cm": confusion_matrix(test_y, test_pred),
        "training_time": training_time
    }

    if get_decision_score is not None:
        train_decision_score = get_decision_score(model, train_X)
        test_decision_score = get_decision_score(model, test_X)

        # Calculate ROC curves
        train_fpr, train_tpr, train_thresholds = roc_curve(train_y, train_decision_score)
        test_fpr, test_tpr, test_thresholds = roc_curve(test_y, test_decision_score)

        # Calculate precision-recall curves
        train_precision, train_recall, train_pr_thresholds = precision_recall_curve(train_y, train_decision_score)
        test_precision, test_recall, test_pr_thresholds = precision_recall_curve(test_y, test_decision_score)

        results["train_decision_score"] = train_decision_score
        results["test_decision_score"] = test_decision_score
        results["train_roc_auc"] = auc(train_fpr, train_tpr)
        results["test_roc_auc"] = auc(test_fpr, test_tpr)
        results["train_pr_auc"] = auc(train_recall, train_precision)
        results["test_pr_auc"] = auc(test_recall, test_precision)
        results["train_roc_curve"] = train_fpr, train_tpr
        results["test_roc_curve"] = test_fpr, test_tpr
        results["train_pr_curve"] = train_precision, train_recall
        results["test_pr_curve"] = test_precision, test_recall

    return results

