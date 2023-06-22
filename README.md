# Reinforcement learning methods applied to a portfolio management problem

In relation to a Master Project, we apply reinforcement learning methods such as Monte-Carlo estimation, Q-Learning and Deep Q-Networks to a portfolio management problem over two stocks. 

The agent seek to invest at each time-step a fraction of the portfolio in the first stock, thus the action space corresponds to: {0.0,0.1,...,1.0}.

State space will corresponds to the returns which can be discretized to exploit Q-Learning or Monte-Carlo estimation.

We assume zero market impact and zero slippage implying that transitions aren't influenced by the actions of the agent. Furthermore, we suppose that the the agent presents no risk aversion, thus it seeks to maximize its expected cumulative returns. Consequently, as the reward function must induce equivalence to the accomplishment of the task, the reward function is simply the realized log-return of the portfolio at the next time-step.

Binomial processes, Markov chains and autoregressive models are used to simulate stock's returns on which we will apply Q-Learning and deep Q-networks to derive an optimal policy. Furthermore, we analyze the impact of some learning parameters on the convergence of the different methods.
