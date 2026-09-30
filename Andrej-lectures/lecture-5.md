# Training Tips

- **Caffeine model zoo** is a place where we get all pre-trained models
- You should have zero centered values for inputs throughout the network
- ReLU has dead neuron problem when its input is negative. This happens due to improper learning rate
- Try Leaky ReLU (may be it will give better results)
- Xavier initialization is good option BUT only for Tanh. It does not work for ReLU
- perform expirements for choosing better hyperparameter. Also do cross validation test
- Plot accuracy while training and testing. Because accuracy is interpretable. meaning, if accuracy for training is high but for testing is low, then its overfitting
- Track ratio of weights (magnitude) and update weights as per the ratio of magnitude. Meaning, initial weight magnitude should be appropriate with its update ratio
