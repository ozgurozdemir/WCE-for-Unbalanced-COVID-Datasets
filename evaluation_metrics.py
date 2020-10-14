from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


def experiment_report(groundtruths, predictions):

    cnfs = confusion_matrix(groundtruths, predictions)
    tn = cnfs[0,0]; tp = cnfs[1,1]
    fn = cnfs[1,0]; fp = cnfs[0,1]
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    bacc = (sensitivity + specificity) / 2.
    ppcr = (tp + fp) / (tp + fp + tn + fn)

    log =  classification_report(groundtruths, predictions)
    log += "\n\n>> Confusion Matrix:\n" + str(cnfs)
    log += "\n\n>> Scores:\n"
    log += f":: Sensitivity: {sensitivity}\n"
    log += f":: Specificity: {specificity}\n"
    log += f":: bACC:        {bacc}\n"
    log += f":: PPCR:        {ppcr}\n"

    return log
