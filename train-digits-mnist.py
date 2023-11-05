# train-digits-mnist.py
# Version : 1.1.1
# Author : Sam Kwok
# License : MIT
# github.com/skwk

from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
import json
import matplotlib.pyplot as plt

# Use true to debug/fine-tune the neural net
use_test_set = True

# Download MNIST
mnist = fetch_openml(
    "mnist_784", version=1, return_X_y=False, as_frame=False, parser="pandas"
)

# Print keys
print('Show keys:', mnist.keys())

# Normalize data
X, y = mnist.data / 255., mnist.target

# Explore data
print('X shape:', X.shape)
print('y shape:', y.shape)

model = MLPClassifier(hidden_layer_sizes=(200,), max_iter=50, alpha=1e-4,
                      solver='sgd', verbose=10, tol=1e-4, random_state=1,
                      learning_rate_init=.1)

print('Training the net')

if use_test_set:
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    model.fit(X_train, y_train)

    print('Training set score: %f' % model.score(X_train, y_train))
    print('Test set score: %f' % model.score(X_test, y_test))

    # Predict using the trained model
    sample_index = 0  # Index of the sample to predict
    sample_to_predict = X_test[sample_index].reshape(1, -1)
    predicted_class = model.predict(sample_to_predict)
    print('Predicted class for sample %d: %d' % (sample_index, predicted_class))

    # Display the sample
    sample_image = X_test[sample_index].reshape(28, 28)
    plt.imshow(sample_image, cmap='gray')
    plt.show()
else:
    model.fit(X, y)
    print('Training set score: %f' % model.score(X, y))

    model_data = {
        "weights": [item.tolist() for item in model.coefs_],
        "biases": [item.tolist() for item in model.intercepts_]
    }

    with open('weights.json', 'w') as json_file:
        json.dump(model_data, json_file, indent=4)

    print('Weights file saved successfully (weights.json)')