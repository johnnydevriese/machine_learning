import numpy as np
import util

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'

def main_LogReg(train_path, valid_path, save_path):
    """Problem (1b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # Train a logistic regression classifier
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    # Plot decision boundary on top of validation set set
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    plot_path = save_path.replace('.txt', '.png')
    util.plot(x_eval, y_eval, clf.theta, plot_path)

    # Use np.savetxt to save predictions on eval set to save_path
    p_eval = clf.predict(x_eval)
    yhat = p_eval > 0.5
    print('LR Accuracy: %.2f' % np.mean( (yhat == 1) == (y_eval == 1)))
    np.savetxt(save_path, p_eval)

class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def gradient(self,x, y):
        n_examples, dim = x.shape
        logits = self.sigmoid(x)
        # grad of logit function
        gradient = 1 / n_examples * x.T @ (logits - y)
        return gradient

    def hessian(self, x, y):
        n_examples, dim = x.shape
        # sigmoid = lambda x: 1 / 1 + np.exp(- x @ self.theta)
        logits = self.sigmoid(x)

        # probs = self._sigmoid(x.dot(self.theta))
        # diag = np.diag(logits * (1. - logits))
        # hess = 1 / n_examples * x.T.dot(diag).dot(x)
        # return hess

        # main diag is just second derivative wrt to itself. e.g. f_xx and f_yy
        main_diagonal = np.diag(logits * (1 - logits))
        hessian = 1 / n_examples * x.T @ main_diagonal @ x 
        return hessian 

    def loss(self, x, y):
        # https://developers.google.com/machine-learning/crash-course/logistic-regression/model-training
        # also in p.16 in Supervised Learning notes 
        n_examples, dim = x.shape
        # sigmoid = lambda x: 1 / 1 + np.exp(- x @ self.theta)
        logits = self.sigmoid(x)

        loss = -np.mean(y * np.log(logits) + (1 + y) * np.log(1 - logits))
        return loss


    def sigmoid(self, x):
        # return 1 / (1 + np.exp(-x.dot(self.theta)))
        return 1 / (1 + np.exp(- x @ self.theta))


    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # NOTE: look at p.18 in notes 
        # we need to calculate theta with Newton and then maximize the 
        # logistic regression log likelihood function l(theta) 
        # prev_theta = theta # store for comparison 

        # m = rows = number of examples 
        # n = columns = number of features 

        # breakpoint()
        # NOTE: it looks like they prepend the '1' at the beginning of the x array! 
        n_examples, dim = x.shape
        if self.theta is None: 
            self.theta = np.zeros(dim)
        
        # just need to init for first time. 
        # theta_prev = np.ones(dim)
        # # print(np.sum(np.abs(theta_prev - self.theta)) < self.eps)

        # current_iteration = 0
        # theta_difference = np.sum(np.abs(theta_prev - self.theta))
        # while theta_difference > self.eps and current_iteration < self.max_iter:
        for i in range(self.max_iter):
            # current_iteration += 1

            gradient = self.gradient(x, y)
            hessian = self.hessian(x, y)

            # theta_prev = self.step(gradient, hessian)
            theta_prev = np.copy(self.theta)
            # theta_prev = self.step()
            # theta_prev = self.theta
            # self.theta = self.theta - self.step_size * np.linalg.inv(hessian) @ gradient
            self.theta -= self.step_size * np.linalg.inv(hessian).dot(gradient)

            if np.sum(np.abs(theta_prev - self.theta)) < self.eps:
                break

        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        # breakpoint()
        # sigmoid = lambda x: 1 / 1 + np.exp(- x @ self.theta)
        prediction = self.sigmoid(x)
        return prediction 
        # *** END CODE HERE ***

def main_GDA(train_path, valid_path, save_path):
    """Problem (1e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # Train a GDA classifier
    clf = GDA()
    clf.fit(x_train, y_train)

    # Plot decision boundary on validation set
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=False)
    plot_path = save_path.replace('.txt', '.png')
    util.plot(x_eval, y_eval, clf.theta, plot_path)
    x_eval = util.add_intercept(x_eval)

    # Use np.savetxt to save outputs from validation set to save_path
    p_eval = clf.predict(x_eval)
    yhat = p_eval > 0.5
    print('GDA Accuracy: %.2f' % np.mean( (yhat == 1) == (y_eval == 1)))
    np.savetxt(save_path, p_eval)

class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def sigmoid(self, x):
        # return 1 / (1 + np.exp(-x.dot(self.theta)))
        return 1 / (1 + np.exp(- x @ self.theta))


    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n_examples, dim = x.shape

        # Find phi, mu_0, mu_1, and sigma
        phi = 1 / n_examples * np.sum(y == 1)
        mu_0 = (y == 0).dot(x) / np.sum(y == 0)
        mu_1 = (y == 1).dot(x) / np.sum(y == 1)
        mu_yi = np.where(np.expand_dims(y == 0, -1),
                         np.expand_dims(mu_0, 0),
                         np.expand_dims(mu_1, 0))
        sigma = 1 / n_examples * (x - mu_yi).T.dot(x - mu_yi)

        # Write theta in terms of the parameters
        self.theta = np.zeros(dim + 1)
        sigma_inv = np.linalg.inv(sigma)
        mu_diff = mu_0.T.dot(sigma_inv).dot(mu_0) - mu_1.T.dot(sigma_inv).dot(mu_1)
        self.theta[0] = 1 / 2 * mu_diff - np.log((1 - phi) / phi)
        self.theta[1:] = -sigma_inv.dot(mu_0 - mu_1)

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        prediction = self.sigmoid(x)
        return prediction 
        # *** END CODE HERE

def main_posonly(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    
    NOTE: You need to complete logreg implementation first (see class above)!!!
    """
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')

    plot_path = save_path.replace('.txt', '.png')
    plot_path_true = plot_path.replace(WILDCARD, 'true')
    plot_path_naive = plot_path.replace(WILDCARD, 'naive')
    plot_path_adjusted = plot_path.replace(WILDCARD, 'adjusted')

    # Problem (2a): Train and test on true labels (t)
    full_predictions = fully_observed_predictions(train_path, test_path, output_path_true, plot_path_true)

    # Problem (2b): Train on y-labels and test on true labels
    naive_predictions, clf = naive_partial_labels_predictions(train_path, test_path, output_path_naive, plot_path_naive)

    # Problem (2f): Apply correction factor using validation set and test on true labels
    alpha = find_alpha_and_plot_correction(clf, valid_path, test_path, output_path_adjusted, plot_path_adjusted, naive_predictions)

    return

def fully_observed_predictions(train_path, test_path, output_path_true, plot_path_true):
    """
    Problem (2a): Fully Observable Binary Classification Helper Function

    Args:
        train_path: Path to CSV file containing dataset for training.
        test_path: Path to CSV file containing dataset for testing.
        output_path_true: Path to save observed predictions
        plot_path_true: Path to save the plot using plot_posonly util function
    Return:
        full_predictions: tensor of predictions returned from applied LogReg classifier prediction
    """
    full_predictions = None
    # Problem (2a): Train and test on true labels (t)
    # Make sure to save predicted probabilities to output_path_true using np.savetxt()
    # *** START CODE HERE ***
    x_train, t_train = util.load_dataset(train_path, label_col='t',
                                         add_intercept=True)
    clf = LogisticRegression()
    clf.fit(x_train, t_train)

    x_test, t_test = util.load_dataset(test_path, label_col='t',
                                       add_intercept=True)

    full_predictions = clf.predict(x_test)
    np.savetxt(output_path_true, full_predictions)
    util.plot(x_test, t_test, clf.theta, plot_path_true)
    # *** END CODE HERE ***
    return full_predictions

def naive_partial_labels_predictions(train_path, test_path, output_path_naive, plot_path_naive):
    """
    Problem (2b): Naive Partial Labels Binary Classification Helper Function

    Args:
        train_path: Path to CSV file containing dataset for training.
        test_path: Path to CSV file containing dataset for testing.
        output_path_naive: Path to save observed predictions
        plot_path_naive: Path to save the plot using plot_posonly util function
    Return:
        naive_predictions: tensor of predictions returned from applied LogReg prediction
        clf: Logistic Regression classifier (will be reused for 2f)
    """
    naive_predictions = None
    clf = None
    # Problem (2b): Train on y-labels and test on true labels
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    # *** START CODE HERE ***
    x_train, y_train = util.load_dataset(train_path, label_col='y',
                                         add_intercept=True)
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    x_test, t_test = util.load_dataset(test_path, label_col='t',
                                       add_intercept=True)
    naive_predictions = clf.predict(x_test)
    np.savetxt(output_path_naive, naive_predictions)
    util.plot(x_test, t_test, clf.theta, plot_path_naive)
    # *** END CODE HERE ***
    return naive_predictions, clf

def find_alpha_and_plot_correction(clf, valid_path, test_path, output_path_adjusted, plot_path_adjusted, naive_predictions):
    """
    Problem (2f): Alpha Correction Binary Classification Helper Function

    Args:
        clf: Logistic regression classifier from part 2b
        valid_path: Path to CSV file containing dataset for validation.
        test_path: Path to CSV file containing dataset for testing.
        output_path_adjusted: Path to save observed predictions
        plot_path_adjusted: Path to save the plot using plot_posonly util function
        naive_predictions: tensor of predictions returned from applied LogReg prediction from 2b
    Return:
        alpha: corrected alpha value
    """
    alpha = None
    # Problem (2f): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to output_path_adjusted
    # *** START CODE HERE ***
    x_valid, y_valid = util.load_dataset(valid_path, label_col='y')
    x_valid = x_valid[y_valid == 1, :]  # Restrict to just the labeled examples
    x_valid = util.add_intercept(x_valid)
    y_pred = clf.predict(x_valid)
    alpha = np.mean(y_pred)
    print('Found alpha = {}'.format(alpha))
    x_test, t_test = util.load_dataset(test_path, label_col='t',
                                       add_intercept=True)

    # Plot and use np.savetxt to save outputs to output_path_adjusted
    np.savetxt(output_path_adjusted, naive_predictions / alpha)
    util.plot(x_test, t_test, clf.theta, plot_path_adjusted, correction=alpha)
    # *** END CODE HERE ***
    return alpha

if __name__ == '__main__':
    '''
    Start of Problem 1: Linear Classifiers
    '''
    # 1b
    main_LogReg(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')
    main_LogReg(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
    # 1e
    main_GDA(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')
    main_GDA(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
    
    '''
    Start of Problem 2: Incomplete, Positive-Only Labels
    '''
    main_posonly(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred.txt')
