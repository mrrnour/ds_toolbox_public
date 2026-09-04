"""Linear regression assumptions checker (linearity, independence, multicollinearity, homoscedasticity, normality, outliers)."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.stats import anderson, jarque_bera, shapiro
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

warnings.filterwarnings("ignore")


###TODO: test it before using:
class LinearRegressionAssumptionsChecker:
    """
    A comprehensive class to check assumptions of Linear Regression Models
    including OLS, Ridge, and Lasso based on Analytics Vidhya methodology:
    1. Linear relationship between X and Y
    2. Independence of residuals (No Autocorrelation)
    3. Multicollinearity (adapted for regularized models)
    4. Homoscedasticity (Equal Variance)
    5. Normality of residuals
    6. No influential outliers

    Note: For Ridge/Lasso, multicollinearity checks are less critical
    as these models handle correlated features differently.

    Usage:
    # Step 0: Fit your model
    model = LinearRegression()
    model.fit(X, y)

    # Step 1: Initialize the checker
    checker = LinearRegressionAssumptionsChecker(
        model=model,
        X=X,
        y=y,
        feature_names=feature_names
    )

    # Step 2: Run full diagnostic
    results = checker.run_full_diagnostic()

    # Step 3: Individual checks (optional)
    checker.check_linearity()
    checker.check_independence()
    checker.check_multicollinearity()
    checker.check_homoscedasticity()
    checker.check_normality()
    checker.check_outliers_influence()
    """

    def __init__(self, model, X, y, feature_names=None, model_type="ols"):
        """
        Initialize the checker with fitted model and data

        Parameters:
        -----------
        model : fitted regression model (sklearn or statsmodels)
        X : array-like, feature matrix
        y : array-like, target variable
        feature_names : list, names of features (optional)
        model_type : str, type of model ('ols', 'ridge', 'lasso')
        """
        self.model = model
        self.X = np.array(X)
        self.y = np.array(y)
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(self.X.shape[1])]
        self.model_type = model_type.lower()

        # Get predictions and residuals
        self.y_pred = model.predict(self.X)
        self.residuals = self.y - self.y_pred

        # Standardized residuals
        self.std_residuals = self.residuals / np.std(self.residuals)

        # For statsmodels compatibility
        if hasattr(model, "resid"):
            self.residuals = model.resid
            self.y_pred = model.fittedvalues

    def check_linearity(self, plot=True):
        """
        Check Assumption 1: Linear relationship between X and Y
        Using residuals vs fitted values plot
        """
        print("=" * 60)
        print("ASSUMPTION 1: LINEAR RELATIONSHIP")
        print("=" * 60)

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            # Residuals vs Fitted Plot
            axes[0].scatter(self.y_pred, self.residuals, alpha=0.7, color="blue")
            axes[0].axhline(y=0, color="red", linestyle="--")
            axes[0].set_xlabel("Fitted Values")
            axes[0].set_ylabel("Residuals")
            axes[0].set_title("Residuals vs Fitted Values\n(Check for Linearity)")

            # Add smooth line
            z = np.polyfit(self.y_pred, self.residuals, 2)
            p = np.poly1d(z)
            x_smooth = np.linspace(self.y_pred.min(), self.y_pred.max(), 100)
            axes[0].plot(x_smooth, p(x_smooth), color="red", linewidth=2)

            # Actual vs Predicted Plot
            axes[1].scatter(self.y, self.y_pred, alpha=0.7, color="green")
            min_val = min(self.y.min(), self.y_pred.min())
            max_val = max(self.y.max(), self.y_pred.max())
            axes[1].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
            axes[1].set_xlabel("Actual Values")
            axes[1].set_ylabel("Predicted Values")
            axes[1].set_title("Actual vs Predicted Values")

            plt.tight_layout()
            plt.show()

        # Analysis
        residual_pattern = np.polyfit(self.y_pred, self.residuals, 2)[0]

        print("📊 Analysis:")
        print(f"   • Residual pattern coefficient: {residual_pattern:.6f}")

        if abs(residual_pattern) < 0.001:
            print("   ✅ PASSED: Linear relationship assumption is satisfied")
            print("   • Residuals are randomly scattered around zero")
        else:
            print("   ❌ VIOLATED: Non-linear relationship detected")
            print("   • Consider polynomial features or non-linear transformations")
            print("   • Suggested: log(X), √X, or X² transformations")

        return abs(residual_pattern) < 0.001

    def check_independence(self, plot=True):
        """
        Check Assumption 2: Independence of residuals (No Autocorrelation)
        Using Durbin-Watson test
        """
        print("\n" + "=" * 60)
        print("ASSUMPTION 2: INDEPENDENCE OF RESIDUALS")
        print("=" * 60)

        # Durbin-Watson Test
        dw_stat = durbin_watson(self.residuals)

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            # Residuals vs Index (Time Series Plot)
            axes[0].plot(self.residuals, marker="o", linestyle="-", alpha=0.7)
            axes[0].axhline(y=0, color="red", linestyle="--")
            axes[0].set_xlabel("Observation Index")
            axes[0].set_ylabel("Residuals")
            axes[0].set_title("Residuals vs Index\n(Check for Autocorrelation)")

            # ACF Plot (simplified)
            from statsmodels.tsa.stattools import acf

            autocorr = acf(self.residuals, nlags=20, fft=True)
            axes[1].bar(range(len(autocorr)), autocorr, alpha=0.7)
            axes[1].axhline(y=0, color="red", linestyle="--")
            axes[1].set_xlabel("Lag")
            axes[1].set_ylabel("Autocorrelation")
            axes[1].set_title("Autocorrelation Function")

            plt.tight_layout()
            plt.show()

        print("📊 Durbin-Watson Test:")
        print(f"   • DW Statistic: {dw_stat:.4f}")
        print("   • Range: 0 to 4 (2 = no autocorrelation)")

        if 1.5 <= dw_stat <= 2.5:
            print("   ✅ PASSED: No significant autocorrelation detected")
        elif dw_stat < 1.5:
            print("   ❌ VIOLATED: Positive autocorrelation detected")
            print("   • Solution: Add lag variables or use time series models")
        else:
            print("   ❌ VIOLATED: Negative autocorrelation detected")
            print("   • Solution: Check model specification")

        return 1.5 <= dw_stat <= 2.5

    def check_multicollinearity(self, plot=True):
        """
        Check Assumption 3: No Multicollinearity
        Using VIF (Variance Inflation Factor) and correlation matrix

        Note: For Ridge/Lasso models, multicollinearity is less problematic:
        - Ridge: Handles multicollinearity by shrinking coefficients
        - Lasso: Performs automatic feature selection
        """
        print("\n" + "=" * 60)
        print("ASSUMPTION 3: NO MULTICOLLINEARITY")
        if self.model_type in ["ridge", "lasso"]:
            print(f"📝 Note: {self.model_type.upper()} regression handles multicollinearity")
        print("=" * 60)

        # Calculate VIF
        X_with_const = sm.add_constant(self.X)
        vif_data = pd.DataFrame()
        vif_data["Feature"] = ["const"] + self.feature_names
        vif_data["VIF"] = [
            variance_inflation_factor(X_with_const, i) for i in range(X_with_const.shape[1])
        ]

        # Correlation Matrix
        corr_matrix = np.corrcoef(self.X.T)

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            # VIF Plot
            vif_features = vif_data[vif_data["Feature"] != "const"]
            axes[0].barh(vif_features["Feature"], vif_features["VIF"], color="skyblue")
            axes[0].axvline(x=4, color="orange", linestyle="--", label="Moderate (VIF=4)")
            axes[0].axvline(x=10, color="red", linestyle="--", label="High (VIF=10)")
            axes[0].set_xlabel("VIF Value")
            axes[0].set_title("Variance Inflation Factor")
            axes[0].legend()

            # Correlation Heatmap
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(
                corr_matrix,
                mask=mask,
                annot=True,
                cmap="coolwarm",
                center=0,
                xticklabels=self.feature_names,
                yticklabels=self.feature_names,
                ax=axes[1],
            )
            axes[1].set_title("Feature Correlation Matrix")

            plt.tight_layout()
            plt.show()

        print("📊 VIF Analysis:")
        for _, row in vif_data.iterrows():
            if row["Feature"] != "const":
                vif_val = row["VIF"]
                if vif_val <= 4:
                    status = "✅ Good"
                elif vif_val <= 10:
                    status = "⚠️  Moderate"
                else:
                    status = "❌ High"
                print(f"   • {row['Feature']}: {vif_val:.3f} {status}")

        # Check for high correlations
        high_corr_pairs = []
        for i in range(len(self.feature_names)):
            for j in range(i + 1, len(self.feature_names)):
                if abs(corr_matrix[i, j]) > 0.8:
                    high_corr_pairs.append(
                        (self.feature_names[i], self.feature_names[j], corr_matrix[i, j])
                    )

        max_vif = vif_data[vif_data["Feature"] != "const"]["VIF"].max()

        if self.model_type == "ols":
            # Standard OLS interpretation
            if max_vif <= 4 and len(high_corr_pairs) == 0:
                print("   ✅ PASSED: No multicollinearity issues detected")
            else:
                print("   ❌ VIOLATED: Multicollinearity detected")
                if high_corr_pairs:
                    print("   • High correlations found:")
                    for feat1, feat2, corr in high_corr_pairs:
                        print(f"     - {feat1} & {feat2}: {corr:.3f}")
                print("   • Solutions: Remove highly correlated features, PCA, Ridge regression")
            return max_vif <= 4 and len(high_corr_pairs) == 0

        else:
            # Ridge/Lasso interpretation
            print(f"   ℹ️  INFO: {self.model_type.upper()} regression analysis:")
            if self.model_type == "ridge":
                print("   • Ridge handles multicollinearity by shrinking coefficients")
                print("   • High VIF/correlations are less problematic")
                if hasattr(self.model, "alpha_"):
                    print(f"   • Regularization strength (α): {self.model.alpha_:.4f}")
            elif self.model_type == "lasso":
                print("   • Lasso performs automatic feature selection")
                print("   • Correlated features may be automatically removed")
                if hasattr(self.model, "alpha_"):
                    print(f"   • Regularization strength (α): {self.model.alpha_:.4f}")

                # Show which features have non-zero coefficients
                if hasattr(self.model, "coef_"):
                    non_zero_features = np.sum(np.abs(self.model.coef_) > 1e-10)
                    print(f"   • Features selected: {non_zero_features}/{len(self.feature_names)}")

            if max_vif > 10 or len(high_corr_pairs) > 0:
                print("   ⚠️  WARNING: High multicollinearity detected")
                print("   • Monitor model stability and coefficient interpretation")
            else:
                print("   ✅ ACCEPTABLE: Multicollinearity within reasonable bounds")

            # For regularized models, we're more lenient
            return True

    def check_homoscedasticity(self, plot=True):
        """
        Check Assumption 4: Homoscedasticity (Equal Variance)
        Using Breusch-Pagan test and residual plots
        """
        print("\n" + "=" * 60)
        print("ASSUMPTION 4: HOMOSCEDASTICITY (EQUAL VARIANCE)")
        print("=" * 60)

        # Breusch-Pagan Test
        bp_test = het_breuschpagan(self.residuals, self.X)
        bp_stat, bp_p_value = bp_test[0], bp_test[1]

        # White Test
        white_test = het_white(self.residuals, self.X)
        white_stat, white_p_value = white_test[0], white_test[1]

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            # Residuals vs Fitted (Homoscedasticity check)
            axes[0].scatter(self.y_pred, self.residuals, alpha=0.7, color="purple")
            axes[0].axhline(y=0, color="red", linestyle="--")
            axes[0].set_xlabel("Fitted Values")
            axes[0].set_ylabel("Residuals")
            axes[0].set_title("Residuals vs Fitted\n(Check for Homoscedasticity)")

            # Scale-Location Plot (Square root of standardized residuals)
            sqrt_std_resid = np.sqrt(np.abs(self.std_residuals))
            axes[1].scatter(self.y_pred, sqrt_std_resid, alpha=0.7, color="orange")
            axes[1].set_xlabel("Fitted Values")
            axes[1].set_ylabel("√|Standardized Residuals|")
            axes[1].set_title("Scale-Location Plot")

            # Add smooth line
            z = np.polyfit(self.y_pred, sqrt_std_resid, 1)
            p = np.poly1d(z)
            x_smooth = np.linspace(self.y_pred.min(), self.y_pred.max(), 100)
            axes[1].plot(x_smooth, p(x_smooth), color="red", linewidth=2)

            plt.tight_layout()
            plt.show()

        print("📊 Statistical Tests:")
        print("   • Breusch-Pagan Test:")
        print(f"     - Statistic: {bp_stat:.4f}")
        print(f"     - P-value: {bp_p_value:.6f}")
        print("   • White Test:")
        print(f"     - Statistic: {white_stat:.4f}")
        print(f"     - P-value: {white_p_value:.6f}")

        alpha = 0.05
        if bp_p_value > alpha and white_p_value > alpha:
            print("   ✅ PASSED: Homoscedasticity assumption satisfied")
            print("   • Equal variance across all fitted values")
        else:
            print("   ❌ VIOLATED: Heteroscedasticity detected")
            print("   • Solutions: log(Y), √Y transformation, or weighted least squares")

        return bp_p_value > alpha and white_p_value > alpha

    def check_normality(self, plot=True):
        """
        Check Assumption 5: Normality of residuals
        Using Q-Q plot and statistical tests
        """
        print("\n" + "=" * 60)
        print("ASSUMPTION 5: NORMALITY OF RESIDUALS")
        print("=" * 60)

        # Statistical Tests
        shapiro_stat, shapiro_p = shapiro(self.residuals)
        jb_stat, jb_p = jarque_bera(self.residuals)
        anderson_stat = anderson(self.residuals, dist="norm")

        if plot:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Q-Q Plot
            stats.probplot(self.residuals, dist="norm", plot=axes[0, 0])
            axes[0, 0].set_title("Q-Q Plot\n(Check for Normality)")
            axes[0, 0].grid(True)

            # Histogram of residuals
            axes[0, 1].hist(self.residuals, bins=30, density=True, alpha=0.7, color="lightblue")
            mu, sigma = np.mean(self.residuals), np.std(self.residuals)
            x = np.linspace(self.residuals.min(), self.residuals.max(), 100)
            axes[0, 1].plot(x, stats.norm.pdf(x, mu, sigma), "r-", linewidth=2, label="Normal")
            axes[0, 1].set_xlabel("Residuals")
            axes[0, 1].set_ylabel("Density")
            axes[0, 1].set_title("Histogram of Residuals")
            axes[0, 1].legend()

            # Box plot
            axes[1, 0].boxplot(self.residuals, vert=True)
            axes[1, 0].set_ylabel("Residuals")
            axes[1, 0].set_title("Box Plot of Residuals")

            # Residuals vs Normal Quantiles
            axes[1, 1].scatter(range(len(self.residuals)), sorted(self.residuals), alpha=0.7)
            axes[1, 1].set_xlabel("Index")
            axes[1, 1].set_ylabel("Sorted Residuals")
            axes[1, 1].set_title("Sorted Residuals Plot")

            plt.tight_layout()
            plt.show()

        print("📊 Normality Tests:")
        print("   • Shapiro-Wilk Test:")
        print(f"     - Statistic: {shapiro_stat:.4f}")
        print(f"     - P-value: {shapiro_p:.6f}")
        print("   • Jarque-Bera Test:")
        print(f"     - Statistic: {jb_stat:.4f}")
        print(f"     - P-value: {jb_p:.6f}")
        print("   • Anderson-Darling Test:")
        print(f"     - Statistic: {anderson_stat.statistic:.4f}")

        alpha = 0.05
        normality_tests = [shapiro_p > alpha, jb_p > alpha]

        if all(normality_tests):
            print("   ✅ PASSED: Residuals are normally distributed")
        else:
            print("   ❌ VIOLATED: Residuals are not normally distributed")
            print("   • Solutions: Transform Y or X variables, remove outliers")

        return all(normality_tests)

    def check_outliers_influence(self, plot=True):
        """
        Check Assumption 6: No influential outliers
        Using Cook's distance and leverage analysis
        """
        print("\n" + "=" * 60)
        print("ASSUMPTION 6: NO INFLUENTIAL OUTLIERS")
        print("=" * 60)

        # Calculate Cook's distance (simplified calculation)
        n = len(self.residuals)
        p = self.X.shape[1] + 1  # number of parameters

        # Leverage values (hat matrix diagonal)
        X_design = sm.add_constant(self.X)
        hat_matrix = X_design @ np.linalg.inv(X_design.T @ X_design) @ X_design.T
        leverage = np.diag(hat_matrix)

        # Cook's distance approximation
        cooks_d = (self.std_residuals**2 / p) * (leverage / (1 - leverage) ** 2)

        # Thresholds
        cook_threshold = 4 / n
        leverage_threshold = 2 * p / n

        if plot:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Cook's Distance Plot
            axes[0, 0].bar(range(n), cooks_d, alpha=0.7, color="red")
            axes[0, 0].axhline(
                y=cook_threshold,
                color="black",
                linestyle="--",
                label=f"Threshold: {cook_threshold:.4f}",
            )
            axes[0, 0].set_xlabel("Observation Index")
            axes[0, 0].set_ylabel("Cook's Distance")
            axes[0, 0].set_title("Cook's Distance Plot")
            axes[0, 0].legend()

            # Residuals vs Leverage
            axes[0, 1].scatter(leverage, self.std_residuals, alpha=0.7, color="blue")
            axes[0, 1].axhline(y=0, color="red", linestyle="--")
            axes[0, 1].axhline(y=2, color="orange", linestyle="--", label="|Std Residual| = 2")
            axes[0, 1].axhline(y=-2, color="orange", linestyle="--")
            axes[0, 1].axvline(
                x=leverage_threshold,
                color="green",
                linestyle="--",
                label=f"Leverage threshold: {leverage_threshold:.4f}",
            )
            axes[0, 1].set_xlabel("Leverage")
            axes[0, 1].set_ylabel("Standardized Residuals")
            axes[0, 1].set_title("Residuals vs Leverage")
            axes[0, 1].legend()

            # Standardized Residuals Plot
            axes[1, 0].scatter(range(n), self.std_residuals, alpha=0.7, color="green")
            axes[1, 0].axhline(y=0, color="red", linestyle="--")
            axes[1, 0].axhline(y=2, color="orange", linestyle="--", alpha=0.7)
            axes[1, 0].axhline(y=-2, color="orange", linestyle="--", alpha=0.7)
            axes[1, 0].set_xlabel("Observation Index")
            axes[1, 0].set_ylabel("Standardized Residuals")
            axes[1, 0].set_title("Standardized Residuals vs Index")

            # Influence Plot (Cook's D vs Leverage)
            axes[1, 1].scatter(leverage, cooks_d, alpha=0.7, color="purple")
            axes[1, 1].axhline(y=cook_threshold, color="red", linestyle="--")
            axes[1, 1].axvline(x=leverage_threshold, color="green", linestyle="--")
            axes[1, 1].set_xlabel("Leverage")
            axes[1, 1].set_ylabel("Cook's Distance")
            axes[1, 1].set_title("Influence Plot")

            plt.tight_layout()
            plt.show()

        # Identify influential points
        high_cook = np.where(cooks_d > cook_threshold)[0]
        high_leverage = np.where(leverage > leverage_threshold)[0]
        outlier_residuals = np.where(np.abs(self.std_residuals) > 2)[0]

        print("📊 Outlier Analysis:")
        print(f"   • Cook's Distance threshold: {cook_threshold:.4f}")
        print(f"   • Leverage threshold: {leverage_threshold:.4f}")
        print(f"   • High Cook's Distance points: {len(high_cook)} observations")
        print(f"   • High Leverage points: {len(high_leverage)} observations")
        print(f"   • Outlier residuals (|z| > 2): {len(outlier_residuals)} observations")

        if len(high_cook) > 0:
            print(
                f"   • Influential observations (high Cook's D): {high_cook[:10]}..."
            )  # Show first 10

        total_outliers = len(set(list(high_cook) + list(high_leverage) + list(outlier_residuals)))
        outlier_percentage = (total_outliers / n) * 100

        if outlier_percentage < 5:
            print("   ✅ PASSED: No significant influential outliers detected")
        else:
            print("   ❌ ATTENTION: Potential influential outliers detected")
            print(f"   • {outlier_percentage:.1f}% of observations may be influential")
            print("   • Solutions: Investigate outliers, robust regression, or remove if justified")

        return outlier_percentage < 5

    def run_full_diagnostic(self):
        """
        Run all assumption checks and provide summary report
        """
        print("🔍 LINEAR REGRESSION ASSUMPTIONS DIAGNOSTIC REPORT")
        print(f"🎯 Model Type: {self.model_type.upper()}")
        print("📖 Based on Analytics Vidhya methodology")
        print(
            "📅 Reference: https://www.analyticsvidhya.com/blog/2016/07/deeper-regression-analysis-assumptions-plots-solutions/"
        )
        print("\n")

        # Run all checks
        results = {
            "Linearity": self.check_linearity(),
            "Independence": self.check_independence(),
            "No Multicollinearity": self.check_multicollinearity(),
            "Homoscedasticity": self.check_homoscedasticity(),
            "Normality": self.check_normality(),
            "No Influential Outliers": self.check_outliers_influence(),
        }

        # Summary Report
        print("\n" + "=" * 60)
        print("📋 SUMMARY REPORT")
        print("=" * 60)

        passed = sum(results.values())
        total = len(results)

        print(f"✅ Assumptions Satisfied: {passed}/{total}")
        print(f"📊 Model Reliability Score: {(passed/total)*100:.1f}%")
        print("\n📝 Detailed Results:")

        for assumption, passed in results.items():
            status = "✅ PASSED" if passed else "❌ VIOLATED"
            print(f"   • {assumption}: {status}")

        # Model-specific interpretation
        if self.model_type in ["ridge", "lasso"]:
            print(f"\n🎯 {self.model_type.upper()} Regression Notes:")
            if self.model_type == "ridge":
                print("   • Multicollinearity is handled by coefficient shrinkage")
                print("   • More robust to outliers than OLS")
            elif self.model_type == "lasso":
                print("   • Automatic feature selection reduces multicollinearity")
                print("   • Some features may have zero coefficients")

        if passed == total:
            print("\n🎉 EXCELLENT! All assumptions satisfied.")
            print(f"   Your {self.model_type} regression model is statistically sound.")
        elif passed >= total * 0.75:
            print("\n👍 GOOD! Most assumptions satisfied.")
            print("   Minor violations may not significantly impact model performance.")
        elif passed >= total * 0.5:
            print("\n⚠️  MODERATE! Several assumptions violated.")
            print("   Consider model improvements or alternative approaches.")
        else:
            print("\n❌ POOR! Major assumption violations detected.")
            if self.model_type == "ols":
                print(
                    "   Consider: Ridge/Lasso regression, non-linear models, or data transformations."
                )
            else:
                print("   Consider: non-linear models, data transformations, or ensemble methods.")

        return results


# Key differences for Ridge/Lasso:
def ridge_lasso_notes():
    """Print a cheat-sheet on which OLS assumptions still apply to Ridge/Lasso.

    Returns
    -------
    None
        Prints to stdout.
    """
    print("\\n🎯 KEY DIFFERENCES FOR RIDGE/LASSO:")
    print("=" * 50)
    print("""
✅ STILL IMPORTANT:
• Linearity - Ridge/Lasso are still linear models
• Independence - Autocorrelation still problematic
• Homoscedasticity - Equal variance still needed
• Normality - For confidence intervals
• Outliers - Still affect model (Ridge more robust)

⚠️ MODIFIED INTERPRETATION:
• Multicollinearity - Less critical for Ridge/Lasso:
  - Ridge: Shrinks correlated coefficients
  - Lasso: Automatically selects features
  - VIF analysis informative but not decisive

🔧 REGULARIZATION BENEFITS:
• Ridge: Handles multicollinearity naturally
• Lasso: Performs feature selection automatically
• Both: More robust than OLS to assumption violations
    """)
