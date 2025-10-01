import time
from sklearn.svm import SVC


def classic_SVC(train_features, train_labels) -> SVC:
    '''
    We train a classical Support Vector Classifier from scikit-learn. For the sake of simplicity, we don't tweak any parameters and rely on the default values.
    '''
    svc = SVC()
    _ = svc.fit(train_features, train_labels) 

    return svc

def get_classic_SVC_score(mod_svc: SVC, features, labels):
    score = mod_svc.score(features, labels)
    return score

def run_classic_SVC(train_features, train_labels, test_features, test_labels):
    start = time.time()
    svc = classic_SVC(train_features, train_labels)
    elapsed = time.time() - start

    train_score = get_classic_SVC_score(svc, train_features, train_labels)
    test_score = get_classic_SVC_score(svc, test_features, test_labels)

    return train_score, test_score, elapsed


