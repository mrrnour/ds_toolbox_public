"""Feature importance + interpretability: tree importance, partial dependence, SHAP."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def feature_importance_batch(umodel, X, y):
    """
    Calculate and plot feature importance for a given model and dataset.
    """
    from sklearn.pipeline import Pipeline

    umodel1 = umodel[-1] if isinstance(umodel, Pipeline) else umodel
    model_name = umodel1.__class__.__name__

    ##TODO: generalize for any tree model:
    if "xgb" in model_name.lower():
        from xgboost import plot_importance

        umodel.fit(X, y)
        plt.figure(
            figsize=(100, 100),
            dpi=150,
        )
        sns.set(rc={"figure.figsize": (20, 10)}, font_scale=1)
        plot_importance(umodel, max_num_features=30)
        plt.show()
        plt.close()

        feature_importance = pd.Series(
            umodel.feature_importances_, index=X.columns
        ).sort_values(ascending=False)
        print(
            "non zero features:",
            feature_importance[feature_importance >= 0.01].index.tolist(),
        )
        print(
            "zero features:",
            feature_importance[feature_importance < 0.01].index.tolist(),
        )

        plt.figure(figsize=(5, 2), dpi=150)

        importance_hist = feature_importance.hist()

        sns.set(rc={"figure.figsize": (20, 10)}, font_scale=1.5)
        importance_hist

        sel_features = feature_importance[feature_importance > 0.01].index.tolist()
    else:
        print("The model is not tree based- returning None")
        feature_importance = None
        sel_features = X.columns

    return feature_importance, sel_features


def pdp_plot_batch(X, umodel, sel_features):
    """
    Generate partial dependence plots (PDP) and individual conditional expectation (ICE) plots for a given model and selected features.
    """
    from sklearn.inspection import PartialDependenceDisplay

    print(
        "Computing partial dependence plots and individual conditional expectation..."
    )

    _, ax = plt.subplots(
        figsize=(30, 30),
        sharey=False,
        constrained_layout=True,
    )

    features_info = {
        "features": sel_features,
        "kind": "both",
        "centered": True,
    }
    display = PartialDependenceDisplay.from_estimator(
        umodel,
        X,
        **features_info,
        ax=ax,
    )


def shap_plots_batch(X, y, umodel, test_size=0.2, kmeans=None, random_state=100):
    """
    Generates SHAP plots for a given model and dataset.
    """
    import shap
    from sklearn.model_selection import RandomizedSearchCV, train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
                                                        X, y,
                                                        test_size=test_size,
                                                        random_state=random_state
                                                      )

    umodel.fit(X_train.values, y_train)

    if kmeans is not None:
       udata= shap.kmeans(X_train, kmeans)
    else:
       udata=X_train.values
    explainer = shap.KernelExplainer(
                                      model=umodel.predict,
                                      data=udata,
                                    )
    shap_values = explainer.shap_values(
                                          X_test,
                                      )

    shap.summary_plot(shap_values, X_test, plot_type="violin")

    return shap_values
