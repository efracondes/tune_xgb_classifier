from sklearn.model_selection import cross_validate
import itertools
import xgboost as xgb
import numpy as np
import random 

def perform_cross_validation(
        model_constructor, 
        model_hyperparams, 
        data, 
        y_col, 
        metrics = ['f1', 'accuracy', 'balanced_accuracy'], 
        n_folds=5
    ):
    model = model_constructor(**model_hyperparams)
    data_cv = data.sample(frac = 1, random_state=1235)
    x_data = data_cv.drop(columns=[y_col])
    y_data = data_cv[y_col]
    cv_results = cross_validate(
        estimator = model, 
        X=x_data,
        y=y_data,
        scoring=metrics,
        cv=n_folds
    )
    return cv_results

def get_all_hyperparam_combinations(hyperparams_space):
    all_hyperparams_comb = [params for params in hyperparams_space.values()]
    all_hyperparams_comb = list(itertools.product(*all_hyperparams_comb))
    param_names = hyperparams_space.keys()
    all_hyperparams_comb = [dict(zip(param_names, param_vals)) for param_vals in all_hyperparams_comb]
    return all_hyperparams_comb

def fine_tune_clf(
    model_constructor, 
    hyperparams_space,
    track_metric,
    metrics,
    data,
    y_col,
    random_search = None
):
    '''
    - random_search (int): Random search of n combs in the hyperparams 
        space, set to None for a complete search
    '''

    all_params_comb = get_all_hyperparam_combinations(hyperparams_space)
    if random_search is not None:
        random.seed(1234)
        all_params_comb = random.sample(all_params_comb, random_search).copy()
    
    # Init tracking metrics
    best_score = -np.inf
    best_params = None
    best_results = None
    num_combs = len(all_params_comb)
    # Searching
    for cnt, param_comb in enumerate(all_params_comb, 1):
        #print(param_comb)
        results = perform_cross_validation(
            model_constructor=model_constructor, 
            model_hyperparams=param_comb, 
            data=data, 
            y_col=y_col,
            metrics=metrics
        )
        # Track best results
        iter_performance = results[f'test_{track_metric}'].mean()
        if iter_performance > best_score:
            best_score = iter_performance
            best_params = param_comb.copy()
            best_results = results.copy()
            print(f'Performances improved! Iter {cnt}/{num_combs}, best {track_metric}={round(best_score, 3)}')
        
    print('--------------------------------')
    print('FINISHED TRAINING!!!!')
    print(f'Best params performance')
    agg_best_res = {metric: round(value.mean(), 3) for metric, value in best_results.items()}
    for met, value in agg_best_res.items(): print(f'{met} = {value}')

    return best_params, best_score, best_results