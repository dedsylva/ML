# Linear Regression 

## Description
Linear Regression is the goto for tackling the problem you encounter for the first time. As a physics major, I found that to be true for nature as well. If you don't know how/where to begin, this is the place for you. Try a Linear Regression on your dataset (or, if you're solving a Physics problem, expand your function in first order Taylor Series).

## How To Use
```python
#LEARNING RATE, SIZE, FEATURES, TARGETS, ITERATIONS, OPTIMIZER
# This program tries to predict the equation Y = 3x + 2

./main.py 0.005 100 2 1 10000 Adam
```

## TODO
* Vanilla Regression (Gradient Descent) [x]
* Metrics for valuating model [x]
* Nice plotting [x]
* Adam Optimizer, aka changing learning rate and doing the moving average on the mean and variance of the distribution (whatever that means) [x]

### Questions
- Does the implementation of Adam Optimizer makes the regression better or we should just use the vanilla approach (aka Gradient Descent)? Let's find out!

- <b>Answer</b>: It appears to not change that much. Maybe that has something to do with having only one layer (if we're comparing the regression with Neural Networks).
