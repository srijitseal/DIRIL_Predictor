import numpy as np
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import ast  # Import the Abstract Syntax Trees (ast) module
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from sklearn.ensemble import RandomForestClassifier

from skactiveml.classifier import SklearnClassifier
from skactiveml.regressor import NICKernelRegressor
from skactiveml.pool import ExpectedModelVarianceReduction
from skactiveml.pool import RandomSampling, MonteCarloEER
from skactiveml.utils import unlabeled_indices, labeled_indices, MISSING_LABEL
from skactiveml.visualization import plot_decision_boundary, plot_utilities

from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_predict, StratifiedKFold

import warnings
mlp.rcParams["figure.facecolor"] = "white"
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.datasets import make_blobs
from skactiveml.pool import UncertaintySampling, BatchBALD

from sklearn.model_selection import train_test_split

from tqdm import tqdm

import deepchem as dc


if('df' not in locals()):
    print("Reading data")
    df = pd.read_csv("data/DILIst_features_v2.csv.gz", compression="gzip")
    df['morgan_fingerprint'] = df['morgan_fingerprint'].apply(ast.literal_eval)
    df['mordred_descriptors'] = df['mordred_descriptors'].apply(ast.literal_eval)
else:
    print("Data already loaded")

param_distributions = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'n_jobs': [10],
    'random_state': [42]
}

def generate_features(df, featurespace):
    
    X = np.array(df[featurespace].tolist())
    y = np.array(df['Label'].tolist())
    smiles = np.array(df['protonated_Output'].tolist())
        
    return X, y, smiles

def random_datasplits(X, y, random_state=42):
    
    # Identify non-NaN indices
    non_nan_indices = ~np.isnan(y)

    # Split only the non-NaN parts into test and part of train
    X_non_nan = X[non_nan_indices]
    y_non_nan = y[non_nan_indices]
    X_train_partial, X_test, y_train_partial, y_test = train_test_split(
        X_non_nan, y_non_nan, test_size=0.25, random_state=random_state, stratify=y_non_nan
    )

    # Combine the non-selected non-NaN data back with NaN-containing rows for the full train set
    # Identify indices for rows used in X_train_partial (inverse operation might be needed depending on how you track selected indices)
    # This is a conceptual step; specifics depend on ensuring we don't double-count or omit any rows

    # For simplicity, let's include all original data in X_train, then remove X_test entries
    X_train = np.concatenate((X_train_partial, X[~non_nan_indices]), axis=0)
    y_train = np.concatenate((y_train_partial, y[~non_nan_indices]), axis=0)

    # Verifications
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    print(f"Number of NaNs in y_test: {np.isnan(y_test).sum()}")  # Should be 0
    print(f"Number of NaNs in y_train: {np.isnan(y_train).sum()}")  # Original number minus the ones in y_test
    
    return (X_train, y_train, X_test, y_test)

def scaffold_datasplits(X, y, smiles, random_state=42):
    
    # Identify non-NaN indices
    non_nan_indices = ~np.isnan(y)

    # Split only the non-NaN parts into test and part of train
    X_non_nan = X[non_nan_indices]
    y_non_nan = y[non_nan_indices]
    smiles_non_nan = smiles[non_nan_indices]
    
    # creation of a deepchem dataset with the smile codes in the ids field
    dataset = dc.data.DiskDataset.from_numpy(X=X_non_nan,y=y_non_nan,ids=smiles_non_nan)

    butinasplitter = dc.splits.ButinaSplitter()
    scaffoldsplitter = dc.splits.ScaffoldSplitter()
    
    train_dataset, test_dataset = butinasplitter.train_test_split(dataset, frac_train = 0.75, seed =42)
    # train_dataset, test_dataset = butinasplitter.train_test_split(dataset, frac_train = 0.75, frac_valid = 0,
    #                                                               frac_test = 0.25, seed =42)
    
    X_train_partial = train_dataset.X
    X_test = test_dataset.X
    y_train_partial = train_dataset.y
    y_test = test_dataset.y

    # Combine the non-selected non-NaN data back with NaN-containing rows for the full train set
    # Identify indices for rows used in X_train_partial (inverse operation might be needed depending on how you track selected indices)
    # This is a conceptual step; specifics depend on ensuring we don't double-count or omit any rows

    # For simplicity, let's include all original data in X_train, then remove X_test entries
    X_train = np.concatenate((X_train_partial, X[~non_nan_indices]), axis=0)
    y_train = np.concatenate((y_train_partial, y[~non_nan_indices]), axis=0)

    # Verifications
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    print(f"Number of NaNs in y_test: {np.isnan(y_test).sum()}")  # Should be 0
    print(f"Number of NaNs in y_train: {np.isnan(y_train).sum()}")  # Original number minus the ones in y_test
    
    return (X_train, y_train, X_test, y_test)


def initial_model(X_train, y_train, X_test, y_test):
    
    non_nan_indices_y_train = ~np.isnan(y_train)
    # Getting unique values and their counts
    unique_values, counts = np.unique(y_train[non_nan_indices_y_train], return_counts=True)
    # Combining unique values and counts into a dictionary for a similar output to pandas.Series.value_counts()
    value_counts = dict(zip(unique_values, counts))
    print("Total data value counts : " , value_counts)


    clf = RandomForestClassifier(random_state=42, n_jobs=-1)

    searcher = RandomizedSearchCV(clf, param_distributions, n_iter=10, scoring='balanced_accuracy', 
                                          cv=5, n_jobs =-1, random_state=42)
    searcher.fit(X_train[non_nan_indices_y_train], y_train[non_nan_indices_y_train])

    # Update clf to the best estimator
    clf = SklearnClassifier(
                searcher.best_estimator_,
                classes=np.unique(y_test),
                random_state=0
            )

    clf.fit(X_train[non_nan_indices_y_train], y_train[non_nan_indices_y_train])
    
    non_nan_indices_y_train = ~np.isnan(y_train)
    optimal_threshold = find_opt_threshold(clf, X_train[non_nan_indices_y_train], y_train[non_nan_indices_y_train])
                    
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class
    # Apply the optimal threshold to determine final predictions
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)

    balanced_accuracy = balanced_accuracy_score(y_test, y_pred_optimal)
    auc_score = roc_auc_score(y_test, y_proba) 
   
    print(f'The balanced accuracy score is {balanced_accuracy}.')
    print(f'The AUC score is {auc_score}.')
    
    return


def find_opt_threshold(clf, X_train, y_train, use_cvpred=False):
    
    # StratifiedKFold preserves the percentage of samples for each class.
    cv = StratifiedKFold(n_splits=5)
    # Use cross_val_predict to get the probability predictions for the positive class
    if (use_cvpred):
        y_train_proba = cross_val_predict(clf, X_train, y_train, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
        
    else:
        y_train_proba = clf.predict_proba(X_train)[:, 1] 

    # Get false positive rates (fpr), true positive rates (tpr), and thresholds from ROC curve
    fpr, tpr, thresholds = roc_curve(y_train, y_train_proba)

    # Find the optimal threshold: the one closest to (0,1) on the ROC plot, or use another criterion
    # This is a simple method to find such a threshold; you might adopt a more sophisticated one
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    return(optimal_threshold)

def active_learning(featurespace, n_cycles, heldoutsplit, optimiser):
    
    X, y, smiles = generate_features(df, featurespace)
    
    match heldoutsplit:
        case "random":
            X_train, y_train, X_test, y_test = random_datasplits(X, y, random_state=42)
        case "scaffold":
            X_train, y_train, X_test, y_test = scaffold_datasplits(X, y, smiles, random_state=42)
        case _:
            raise ValueError(f"Splitting {heldoutsplit} not supported.")

    # Getting unique values and their counts
    unique_values, counts = np.unique(y_train, return_counts=True)
    # Combining unique values and counts into a dictionary for a similar output to pandas.Series.value_counts()
    value_counts = dict(zip(unique_values, counts))
    print("Total data value counts train: " , value_counts)
        
    # Getting unique values and their counts
    unique_values, counts = np.unique(y_test, return_counts=True)
    # Combining unique values and counts into a dictionary for a similar output to pandas.Series.value_counts()
    value_counts = dict(zip(unique_values, counts))
    print("Total data value counts test: " , value_counts)
    
    # Create classifier and query strategy.
    clf = SklearnClassifier(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        classes=np.unique(y_test),
        random_state=42
        )
    
    match optimiser:
        case "entropy":
            qs = UncertaintySampling(method='entropy', random_state=42)
        case "RandomSampling":
            qs = RandomSampling(random_state=42)
        case "margin_sampling":
            qs = UncertaintySampling(method='margin_sampling', random_state=42)
        case "BatchBALD":
            qs = BatchBALD(random_state=42)
        case _:
            raise ValueError(f"Optimizer {optimiser} not supported.")
                  
    # initial_model(X_train, y_train, X_test, y_test)
    
    accuracy_scores = []
    auc_scores = []
    sample_counts = []

    non_nan_indices_y_train_original = ~np.isnan(y_train)
    
    for c in tqdm(range(n_cycles), desc='Processing cycles'):

        # Optimize hyperparameters using RandomizedSearchCV
        clf = RandomForestClassifier(random_state=42)
        non_nan_indices_y_train = ~np.isnan(y_train)
        searcher = RandomizedSearchCV(clf, param_distributions, n_iter=10, scoring='balanced_accuracy', 
                                      cv=5, n_jobs =-1, random_state=42)
        searcher.fit(X_train[non_nan_indices_y_train], y_train[non_nan_indices_y_train])

        # Update clf to the best estimator
        clf = SklearnClassifier(
            searcher.best_estimator_,
            classes=np.unique(y_test),
            random_state=0
        )

        clf.fit(X_train, y_train)
        
        # plotting
        unlbld_idx = unlabeled_indices(y_train)
        lbld_idx = labeled_indices(y_train)

        # print(f'After {c} iterations:')

        #Optimise threshold 
        #Option 1 Use code below if you want to optimise on original data _and_ metabolites
        #non_nan_indices_y_train = ~np.isnan(y_train)
        #optimal_threshold = find_opt_threshold(clf, X_train[non_nan_indices_y_train], y_train[non_nan_indices_y_train])

        #Option 2 Use code below if you want to optimise on original data only, no metabolites (recommended)
        optimal_threshold = find_opt_threshold(clf, X_train[non_nan_indices_y_train_original], 
                                               y_train[non_nan_indices_y_train_original],
                                               use_cvpred= True)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class
        # Apply the optimal threshold to determine final predictions
        y_pred_optimal = (y_proba >= optimal_threshold).astype(int)

        balanced_accuracy = balanced_accuracy_score(y_test, y_pred_optimal)
        auc_score = roc_auc_score(y_test, y_proba)  # Ensure y_test is appropriately encoded for binary classification

        if(c==0):
            print(f'Initial model without metabolites:')
            print(f'The balanced accuracy score is {balanced_accuracy}.')
            print(f'The AUC score is {auc_score}.')

        accuracy_scores.append(balanced_accuracy)
        auc_scores.append(auc_score)
        sample_counts.append(len(lbld_idx))

        match optimiser:
            case "entropy":
                query_idx = qs.query(X=X_train, y=y_train, clf=clf, batch_size=1)
            case "RandomSampling":
                query_idx = qs.query(X=X_train, y=y_train)
            case "margin_sampling":
                query_idx = qs.query(X=X_train, y=y_train, clf=clf, batch_size=1)
            case "BatchBALD":
                query_idx = qs.query(X=X_train, y=y_train, ensemble=clf)
            case _:
                raise ValueError(f"Optimizer {optimiser} not supported.")
        
       
        y_train[query_idx] = clf.predict_proba(X_train[query_idx])[:, 1]
        # print(query_idx)
        # print(y_train[query_idx])
        
        y_train[query_idx] = (y_train[query_idx] >= optimal_threshold).astype(int)
        # print(optimal_threshold)
        # print(y_train[query_idx])

        

    plt.figure(figsize=(10, 5))
    plt.plot(sample_counts, accuracy_scores, marker='o', label='Balanced Accuracy')
    plt.xlabel('Number of Labeled Samples + Metabolites')
    plt.ylabel('Balanced Accuracy Score')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'./results_plots/accuracy_vs_samples{featurespace}_{heldoutsplit}_{optimiser}_{n_cycles}.png') 
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.plot(sample_counts, auc_scores, marker='o', label='Balanced Accuracy')
    plt.xlabel('Number of Labeled Samples + Metabolites')
    plt.ylabel('AUCROC')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'./results_plots/auc_vs_samples{featurespace}_{heldoutsplit}_{optimiser}_{n_cycles}.png') 
    plt.show()
    
    # Assuming accuracy_scores, auc_scores, and sample_counts are already populated lists
    results_df = pd.DataFrame({
        'Sample_Counts': sample_counts,
        'Accuracy_Scores': accuracy_scores,
        'AUC_Scores': auc_scores
    })
    
    results_df["featurespace"] = featurespace

    # Save the DataFrame to a CSV file
    results_df.to_csv(f'./results_plots/model_performance_results{featurespace}_{heldoutsplit}_{optimiser}_{n_cycles}.csv', index=False)

    return

print("Running Models")

active_learning(featurespace ="mordred_descriptors", n_cycles=150 , heldoutsplit="scaffold" , optimiser="entropy")
active_learning(featurespace ="morgan_fingerprint", n_cycles=150 , heldoutsplit="scaffold", optimiser="entropy")
active_learning(featurespace ="mordred_descriptors", n_cycles=150 , heldoutsplit="random" , optimiser="entropy")
active_learning(featurespace ="morgan_fingerprint", n_cycles=150 , heldoutsplit="random", optimiser="entropy")
active_learning(featurespace ="mordred_descriptors", n_cycles=150 , heldoutsplit="scaffold" , optimiser="RandomSampling")
active_learning(featurespace ="morgan_fingerprint", n_cycles=150 , heldoutsplit="scaffold", optimiser="RandomSampling")
active_learning(featurespace ="mordred_descriptors", n_cycles=150 , heldoutsplit="random" , optimiser="RandomSampling")
active_learning(featurespace ="morgan_fingerprint", n_cycles=150 , heldoutsplit="random", optimiser="RandomSampling")
active_learning(featurespace ="mordred_descriptors", n_cycles=150 , heldoutsplit="scaffold" , optimiser="margin_sampling")
active_learning(featurespace ="morgan_fingerprint", n_cycles=150 , heldoutsplit="scaffold", optimiser="margin_sampling")
active_learning(featurespace ="mordred_descriptors", n_cycles=150 , heldoutsplit="random" , optimiser="margin_sampling")
active_learning(featurespace ="morgan_fingerprint", n_cycles=150 , heldoutsplit="random", optimiser="margin_sampling")
active_learning(featurespace ="mordred_descriptors", n_cycles=150 , heldoutsplit="scaffold" , optimiser="BatchBALD")
active_learning(featurespace ="morgan_fingerprint", n_cycles=150 , heldoutsplit="scaffold", optimiser="BatchBALD")
active_learning(featurespace ="mordred_descriptors", n_cycles=150 , heldoutsplit="random" , optimiser="BatchBALD")
active_learning(featurespace ="morgan_fingerprint", n_cycles=150 , heldoutsplit="random", optimiser="BatchBALD")