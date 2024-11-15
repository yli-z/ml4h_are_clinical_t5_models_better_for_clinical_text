def compute_confidence_interval(m):
    # m is a list of test scores in each fold (fold 1 - 5)
    ## Confidence Interval
    import numpy as np
    seed = np.random.RandomState(42)
    n_iter = 100
    m_ci = np.percentile([np.mean(seed.choice(m, 5, replace=True)) for _ in range(n_iter)], q=[2.5,97.5])

    return m_ci

def compute_ttest(m1, m2):
    # m1, m2 are lists of test scores in each fold for model 1 (fold 1 - 5)
    ## T-test
    from scipy.stats import ttest_rel
    test_res = ttest_rel(m1, m2)

    return test_res