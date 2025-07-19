import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.1, n_epochs=100):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Training loop
        for _ in range(self.n_epochs):
            for idx, x_i in enumerate(X):
                # Calculate prediction
                output = np.dot(x_i, self.weights) + self.bias
                y_pred = self._step_function(output)

                # Update weights and bias
                updated_weights = self.lr * (y[idx] - y_pred)
                self.weights += updated_weights * x_i
                self.bias += updated_weights

    def _step_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, X):
        output = np.dot(X, self.weights) + self.bias
        y_pred = [self._step_function(x) for x in  output]
        return np.array(y_pred)


# (OR gate problem)
if __name__ == "__main__":
    # Training data (OR gate)
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 1, 1, 1])

    # Create and train perceptron
    p = Perceptron()
    p.fit(X, y)

    # Take input from the user
    print("Enter two binary inputs (0 or 1) for the OR gate:")
    input_1 = int(input("Input 1: "))
    input_2 = int(input("Input 2: "))

    # Validation of input
    if input_1 not in [0, 1] or input_2 not in [0, 1]:
        print("Invalid input! Please enter 0 or 1.")
    else:
        # Prediction
        user_input = np.array([[input_1, input_2]])
        prediction = p.predict(user_input)
        print(f"\nPrediction for input [{input_1}, {input_2}]: {prediction[0]}")


