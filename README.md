# Workout

### What is Workout?

Workout is an API to import and use `OpenAI-Gym`'s environment with `PyTorch` effortlessly

### Why it is required?

`PyTorch`: Flexible framework to implement deep neural networks and has better GPU integration
`OpenAI-Gym`: Provides extensive and varied Reinforcement Learning environments to use readily

![workout](docs/workout.jpg)

However, the integration between two is not very extensive. Many works have been done to implement
deep network based Reinforcement Learning algorithms using `PyTorch` seperately, then transfer the whole control
to `Gym`'s environment to estimate reward function, state of the system, possible actions for next step, etc.,
and pass it again to `PyTorch`'s model. Therefore, to avoid such complications, `Workout` provides a higher level of abstraction to the `Gym`'s environment, providing an interface to make it more `PyTorch` oriented. By doing so,
the users shall effortlessly use `Gym`'s environment without affecting `PyTorch`'s syntactic sugar. Also, the
translation to `PyTorch` codebase would improve the uniformity of the underlying kernel and helps heavily in
parallelization using GPUs.

### How it is done?

`Workout` provides several classes that acts as an interface between `Gym` and `Pytorch`. The package is centered towards Q-Learning. So it will allow users to define their own Policies, Models, preprocessing the inputs to and outputs from the model and define training loops or use the default ones.
