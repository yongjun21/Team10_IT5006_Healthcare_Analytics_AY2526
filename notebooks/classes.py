import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.class_weight import compute_class_weight
from scipy.optimize import minimize
from scipy import stats

class CustomLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Custom Logistic Regression with L1/L2 regularization, callback support, and class weights
    """
    
    def __init__(self, alpha=0.0, l1_ratio=0.0, max_iter=100, tol=1e-4, 
                 random_state=None, callback=None, class_weight=None):
        self.alpha = alpha
        self.l1_ratio = l1_ratio  # 0 = L2, 1 = L1, 0.5 = Elastic Net
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.callback = callback
        self.class_weight = class_weight
        
        # Training history
        self.loss_history = []
        self.gradient_history = []
        self.iteration_history = []
        
    def _sigmoid(self, z):
        """Sigmoid function with numerical stability"""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _get_class_weights(self, y):
        """Calculate class weights for given labels"""
        if self.class_weight is None:
            return None
        
        if isinstance(self.class_weight, dict):
            # Dictionary format: {class_label: weight}
            weights = np.array([self.class_weight.get(label, 1.0) for label in y])
        elif isinstance(self.class_weight, str) and self.class_weight == 'balanced':
            # Balanced weights: n_samples / (n_classes * np.bincount(y))
            classes = np.unique(y)
            class_weights = compute_class_weight('balanced', classes=classes, y=y)
            weight_dict = dict(zip(classes, class_weights))
            weights = np.array([weight_dict[label] for label in y])
        else:
            # List/array format
            weights = np.array(self.class_weight)
        
        return weights
    
    def _log_likelihood(self, X, y, params):
        """Calculate weighted log-likelihood"""
        z = np.dot(X, params)
        probs = self._sigmoid(z)
        probs = np.clip(probs, 1e-15, 1-1e-15)
        
        # Calculate basic log-likelihood
        log_likelihood = y * np.log(probs) + (1 - y) * np.log(1 - probs)
        
        # Apply class weights if specified
        weights = self._get_class_weights(y)
        if weights is not None:
            log_likelihood = weights * log_likelihood
        
        return np.sum(log_likelihood)
    
    def _regularization_penalty(self, params):
        """Calculate regularization penalty"""
        if self.alpha == 0:
            return 0
        
        # L1 penalty
        l1_penalty = self.l1_ratio * np.sum(np.abs(params))
        
        # L2 penalty  
        l2_penalty = (1 - self.l1_ratio) * np.sum(params ** 2)
        
        return self.alpha * (l1_penalty + l2_penalty)
    
    def _objective(self, params, X, y):
        """Objective function to minimize"""
        # Negative log-likelihood
        nll = -self._log_likelihood(X, y, params)
        
        # Add regularization
        penalty = self._regularization_penalty(params)
        
        total_loss = nll + penalty
        
        # Store for history
        self.loss_history.append(total_loss)
        self.iteration_history.append(len(self.loss_history))
        
        # Call callback if provided
        if self.callback is not None:
            self.callback(params, total_loss, len(self.loss_history))
        
        return total_loss
    
    def _gradient(self, params, X, y):
        """Calculate gradient of the objective function with class weights"""
        z = np.dot(X, params)
        probs = self._sigmoid(z)
        
        # Calculate basic gradient of negative log-likelihood
        grad_nll = -np.dot(X.T, y - probs)
        
        # Apply class weights if specified
        weights = self._get_class_weights(y)
        if weights is not None:
            # Apply weights to gradient
            # For each sample, multiply the gradient contribution by its weight
            weighted_residuals = weights * (y - probs)
            grad_nll = -np.dot(X.T, weighted_residuals)
        
        # Gradient of regularization
        grad_penalty = np.zeros_like(params)
        if self.alpha > 0:
            # L1 gradient (subgradient)
            if self.l1_ratio > 0:
                grad_penalty += self.alpha * self.l1_ratio * np.sign(params)
            
            # L2 gradient
            if self.l1_ratio < 1:
                grad_penalty += self.alpha * (1 - self.l1_ratio) * 2 * params
        
        total_grad = grad_nll + grad_penalty
        self.gradient_history.append(np.linalg.norm(total_grad))
        
        return total_grad
    
    def fit(self, X, y):
        """Fit the model"""
        X, y = check_X_y(X, y)
        
        # Add intercept column
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        # Initialize parameters
        np.random.seed(self.random_state)
        n_features = X_with_intercept.shape[1]
        params = np.random.normal(0, 0.01, n_features)
        
        # Clear history
        self.loss_history = []
        self.gradient_history = []
        self.iteration_history = []
        
        # Optimize
        result = minimize(
            fun=self._objective,
            x0=params,
            args=(X_with_intercept, y),
            method='L-BFGS-B',
            jac=self._gradient,
            options={'maxiter': self.max_iter, 'ftol': self.tol}
        )
        
        # Store results
        self.coef_ = result.x[1:]  # Exclude intercept
        self.intercept_ = result.x[0]
        self.n_iter_ = result.nit
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        X = check_array(X)
        z = np.dot(X, self.coef_) + self.intercept_
        probs = self._sigmoid(z)
        return np.column_stack([1 - probs, probs])
    
    def predict(self, X):
        """Predict class labels"""
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)
    
    def _hessian(self, X, y):
        """Calculate Hessian matrix (second derivatives of log-likelihood)"""
        # Add intercept
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        # Get current parameters
        params = np.concatenate([[self.intercept_], self.coef_])
        
        # Calculate probabilities
        z = np.dot(X_with_intercept, params)
        probs = self._sigmoid(z)
        
        # Calculate diagonal weights: W[i,i] = p_i * (1 - p_i)
        W_diag = probs * (1 - probs)
        
        # Apply class weights if specified
        weights = self._get_class_weights(y)
        if weights is not None:
            # Apply weights to diagonal elements
            W_diag = W_diag * weights
        
        # Calculate Hessian efficiently: H = X^T * diag(W) * X
        # Instead of creating the full diagonal matrix, we can compute this as:
        # H = X^T * (W_diag * X) where W_diag is broadcasted
        # This is equivalent to: H[i,j] = sum_k(X[k,i] * W_diag[k] * X[k,j])
        hessian = np.dot(X_with_intercept.T, W_diag[:, np.newaxis] * X_with_intercept)
        
        return hessian
    
    def get_standard_errors(self, X, y, ridge_alpha=0):
        """Calculate standard errors of coefficients with tiny ridge regularization"""
        try:
            hessian = self._hessian(X, y)
            
            # Add tiny ridge regularization to improve numerical stability
            # This adds a small diagonal term to make the matrix more invertible
            n_params = hessian.shape[0]
            ridge_term = ridge_alpha * np.eye(n_params)
            regularized_hessian = hessian + ridge_term
            
            # Calculate covariance matrix
            # Cov = inv(Hessian) for maximum likelihood estimates
            try:
                cov_matrix = np.linalg.inv(regularized_hessian)
            except np.linalg.LinAlgError:
                # If still singular, use pseudo-inverse
                cov_matrix = np.linalg.pinv(regularized_hessian)
            
            # Standard errors are square root of diagonal elements
            standard_errors = np.sqrt(np.diag(cov_matrix))
            
            # Separate intercept and coefficient standard errors
            intercept_se = standard_errors[0]
            coef_se = standard_errors[1:]
            
            return intercept_se, coef_se
            
        except Exception as e:
            print(f"Error calculating standard errors: {e}")
            return None, None
    
    def get_t_values(self, X, y, ridge_alpha=0):
        """Calculate t-values for coefficients"""
        intercept_se, coef_se = self.get_standard_errors(X, y, ridge_alpha)
        
        if intercept_se is None or coef_se is None:
            return None, None
        
        # Calculate t-values
        intercept_t = self.intercept_ / intercept_se
        coef_t = self.coef_ / coef_se
        
        return intercept_t, coef_t
    
    def get_p_values(self, X, y, ridge_alpha=0):
        """Calculate p-values for coefficients"""
        intercept_t, coef_t = self.get_t_values(X, y, ridge_alpha)
        
        if intercept_t is None or coef_t is None:
            return None, None
        
        # Calculate degrees of freedom (n_samples - n_features - 1)
        n_samples = X.shape[0]
        n_features = X.shape[1]
        df = n_samples - n_features - 1
        
        # Calculate p-values (two-tailed test)
        intercept_p = 2 * (1 - stats.t.cdf(np.abs(intercept_t), df))
        coef_p = 2 * (1 - stats.t.cdf(np.abs(coef_t), df))
        
        return intercept_p, coef_p
    
    def get_confidence_intervals(self, X, y, alpha=0.05, ridge_alpha=0):
        """Calculate confidence intervals for coefficients"""
        intercept_se, coef_se = self.get_standard_errors(X, y, ridge_alpha)
        
        if intercept_se is None or coef_se is None:
            return None, None
        
        # Calculate degrees of freedom
        n_samples = X.shape[0]
        n_features = X.shape[1]
        df = n_samples - n_features - 1
        
        # Calculate critical t-value
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        # Calculate confidence intervals
        intercept_ci = (
            self.intercept_ - t_critical * intercept_se,
            self.intercept_ + t_critical * intercept_se
        )
        
        coef_ci = np.column_stack([
            self.coef_ - t_critical * coef_se,
            self.coef_ + t_critical * coef_se
        ])
        
        return intercept_ci, coef_ci


class MixedNaiveBayes:
    def __init__(self, class_weight='balanced'):
        self.binary_model = None
        self.continuous_model = None
        self.binary_features = None
        self.continuous_features = None
        self.class_weight = class_weight
        
    def fit(self, X, y):
        # Identify feature types
        self.binary_features = []
        self.continuous_features = []
        
        for i in range(X.shape[1]):
            unique_vals = np.unique(X[:, i])
            if len(unique_vals) == 2 and set(unique_vals) == {0, 1}:
                self.binary_features.append(i)
            else:
                self.continuous_features.append(i)
        
        # Calculate class weights if needed
        if self.class_weight == 'balanced':
            classes = np.unique(y)
            class_weights = compute_class_weight('balanced', classes=classes, y=y)
            self.class_prior = dict(zip(classes, class_weights))
        else:
            self.class_prior = None
        
        # Train separate models
        if self.binary_features:
            X_binary = X[:, self.binary_features]
            self.binary_model = BernoulliNB()
            if self.class_prior is not None:
                # Apply class balancing by adjusting class priors
                self.binary_model.fit(X_binary, y)
                # Manually adjust class priors
                self.binary_model.class_prior_ = np.array([self.class_prior[0], self.class_prior[1]])
            else:
                self.binary_model.fit(X_binary, y)
            
        if self.continuous_features:
            X_continuous = X[:, self.continuous_features]
            self.continuous_model = GaussianNB()
            if self.class_prior is not None:
                # Apply class balancing by adjusting class priors
                self.continuous_model.fit(X_continuous, y)
                # Manually adjust class priors
                self.continuous_model.class_prior_ = np.array([self.class_prior[0], self.class_prior[1]])
            else:
                self.continuous_model.fit(X_continuous, y)
            
        return self
    
    def predict_proba(self, X):
        log_probs = np.zeros((X.shape[0], 2))
        
        if self.binary_model is not None:
            X_binary = X[:, self.binary_features]
            binary_log_probs = self.binary_model.predict_log_proba(X_binary)
            log_probs += binary_log_probs
            
        if self.continuous_model is not None:
            X_continuous = X[:, self.continuous_features]
            continuous_log_probs = self.continuous_model.predict_log_proba(X_continuous)
            log_probs += continuous_log_probs
            
        # Convert back to probabilities
        exp_log_probs = np.exp(log_probs - np.max(log_probs, axis=1, keepdims=True))
        return exp_log_probs / np.sum(exp_log_probs, axis=1, keepdims=True)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def get_log_likelihood(self, X):
        """
        Get log likelihood as a single 2D array with shape (n_features, n_classes).
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input samples
            
        Returns:
        --------
        log_likelihood : array
            Array of shape (n_features, n_classes) with log likelihoods
        """
        n_features = X.shape[1]
        n_classes = 2  # Assuming binary classification
        
        # Initialize the result array
        log_likelihood = np.zeros((n_features, n_classes))
        
        if self.binary_model is not None:
            X_binary = X[:, self.binary_features]
            binary_log_probs = self.binary_model.predict_log_proba(X_binary)
            
            # For each binary feature, calculate its individual contribution
            for i, feature_idx in enumerate(self.binary_features):
                feature_values = X[:, feature_idx]
                for class_idx in range(binary_log_probs.shape[1]):
                    # Calculate log likelihood for this feature
                    if hasattr(self.binary_model, 'feature_log_prob_'):
                        # Use the feature log probabilities from the trained model
                        feature_log_prob = self.binary_model.feature_log_prob_[class_idx, i]
                        # For binary features, this is the log probability of the feature being 1
                        # We need to adjust based on the actual feature values
                        log_likelihood_contrib = np.where(feature_values == 1, 
                                                         feature_log_prob, 
                                                         1 - np.exp(feature_log_prob))
                        log_likelihood[feature_idx, class_idx] = np.mean(log_likelihood_contrib)
                    else:
                        # Fallback: use the overall log probabilities divided by number of features
                        log_likelihood[feature_idx, class_idx] = np.mean(binary_log_probs[:, class_idx]) / len(self.binary_features)
            
        if self.continuous_model is not None:
            X_continuous = X[:, self.continuous_features]
            continuous_log_probs = self.continuous_model.predict_log_proba(X_continuous)
            
            # For each continuous feature, calculate its individual contribution
            for i, feature_idx in enumerate(self.continuous_features):
                feature_values = X[:, feature_idx]
                for class_idx in range(continuous_log_probs.shape[1]):
                    # Get the parameters for this feature and class
                    if hasattr(self.continuous_model, 'theta_') and hasattr(self.continuous_model, 'sigma_'):
                        # Use the mean and variance from the trained model
                        mean = self.continuous_model.theta_[class_idx, i]
                        var = self.continuous_model.sigma_[class_idx, i]
                        
                        # Calculate log likelihood for this feature using Gaussian distribution
                        log_likelihood_contrib = -0.5 * (np.log(2 * np.pi * var) + 
                                                        ((feature_values - mean) ** 2) / var)
                        log_likelihood[feature_idx, class_idx] = np.mean(log_likelihood_contrib)
                    else:
                        # Fallback: use the overall log probabilities divided by number of features
                        log_likelihood[feature_idx, class_idx] = np.mean(continuous_log_probs[:, class_idx]) / len(self.continuous_features)
        
        return log_likelihood


class LossTracker:
    """
    Advanced loss tracker with callback support for custom GLM
    """
    
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # Training history
        self.train_loss_history = []
        self.test_loss_history = []
        self.train_nll_history = []
        self.test_nll_history = []
        self.penalty_history = []
        self.iteration_history = []
        self.gradient_norm_history = []
        
        # Current model state
        self.current_params = None
        self.current_alpha = None
        self.current_l1_ratio = None
        
    def _calculate_log_likelihood(self, X, y, params):
        """Calculate pure log-likelihood (no regularization)"""
        try:
            # Add intercept
            X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
            
            # Calculate probabilities
            z = np.dot(X_with_intercept, params)
            z = np.clip(z, -500, 500)
            probs = 1 / (1 + np.exp(-z))
            probs = np.clip(probs, 1e-15, 1-1e-15)
            
            # Calculate log-likelihood
            log_likelihood = np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))
            return -log_likelihood  # Return negative log-likelihood
            
        except Exception as e:
            return np.inf
    
    def _calculate_regularization_penalty(self, params, alpha, l1_ratio):
        """Calculate regularization penalty"""
        if alpha == 0:
            return 0
        
        # L1 penalty
        l1_penalty = l1_ratio * np.sum(np.abs(params))
        
        # L2 penalty
        l2_penalty = (1 - l1_ratio) * np.sum(params ** 2)
        
        return alpha * (l1_penalty + l2_penalty)
    
    def create_callback(self, alpha, l1_ratio):
        """Create callback function for optimization"""
        self.current_alpha = alpha
        self.current_l1_ratio = l1_ratio
        
        def callback(params, loss, iteration):
            # Store current parameters
            self.current_params = params
            
            # Calculate pure log-likelihood (no regularization)
            train_nll = self._calculate_log_likelihood(self.X_train, self.y_train, params)
            test_nll = self._calculate_log_likelihood(self.X_test, self.y_test, params)
            
            # Calculate regularization penalty
            penalty = self._calculate_regularization_penalty(params, alpha, l1_ratio)
            
            # Store in history
            self.train_loss_history.append(loss)
            self.test_loss_history.append(test_nll)
            self.train_nll_history.append(train_nll)
            self.test_nll_history.append(test_nll)
            self.penalty_history.append(penalty)
            self.iteration_history.append(iteration)
            
            # Calculate gradient norm (approximate)
            if len(self.train_loss_history) > 1:
                grad_norm = abs(self.train_loss_history[-1] - self.train_loss_history[-2])
                self.gradient_norm_history.append(grad_norm)
            else:
                self.gradient_norm_history.append(0)
        
        return callback
    
    def get_per_sample_losses(self):
        """Get per-sample losses for fair comparison"""
        if not self.train_loss_history:
            return None
        
        n_train = len(self.y_train)
        n_test = len(self.y_test)
        
        return {
            'train_loss_per_sample': [loss / n_train for loss in self.train_loss_history],
            'test_loss_per_sample': [loss / n_test for loss in self.test_loss_history],
            'train_nll_per_sample': [nll / n_train for nll in self.train_nll_history],
            'test_nll_per_sample': [nll / n_test for nll in self.test_nll_history],
            'penalty_per_sample': [penalty / n_train for penalty in self.penalty_history],
            'iterations': self.iteration_history
        }
    
    def get_overfitting_metrics(self):
        """Get proper overfitting metrics using per-sample losses"""
        if not self.train_loss_history:
            return None
        
        n_train = len(self.y_train)
        n_test = len(self.y_test)
        
        final_train_loss_per_sample = self.train_loss_history[-1] / n_train
        final_test_loss_per_sample = self.test_loss_history[-1] / n_test
        
        overfitting_gap = final_test_loss_per_sample - final_train_loss_per_sample
        overfitting_ratio = (overfitting_gap / final_test_loss_per_sample * 100) if final_test_loss_per_sample > 0 else 0
        
        return {
            'overfitting_gap': overfitting_gap,
            'overfitting_ratio_percent': overfitting_ratio,
            'is_overfitting': final_test_loss_per_sample > final_train_loss_per_sample,
            'final_train_loss_per_sample': final_train_loss_per_sample,
            'final_test_loss_per_sample': final_test_loss_per_sample
        }
    
    def get_summary(self):
        """Get training summary"""
        if not self.train_loss_history:
            return None
        
        return {
            'final_train_loss': self.train_loss_history[-1],
            'final_test_loss': self.test_loss_history[-1],
            'final_train_nll': self.train_nll_history[-1],
            'final_test_nll': self.test_nll_history[-1],
            'final_penalty': self.penalty_history[-1],
            'n_iterations': len(self.train_loss_history),
            'convergence': self.gradient_norm_history[-1] if self.gradient_norm_history else 0
        }