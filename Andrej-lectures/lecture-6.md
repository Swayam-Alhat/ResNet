# Neural networks and Intro to CNNs

- Basic SGD are slow. SO, use momentum SGD
- Loss surface have multiple local minima and 1 global minima. But this is not issue. Because when we have large networks, there is no good or bad minima. Difference between both shrinks down.
- Adagrad is good option. It shrinks learning rate fast. So, RMSprop solves this. It solves the problem
- Adam is also good
- Use learning rate decay while training
- Andrej use Adam optimizer (his personal choice)
- Before training begins, overfit small training data to our model
