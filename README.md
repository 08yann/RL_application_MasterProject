# Reinforcement learning methods applied to a portfolio management problem

Firstly, we introduce the portfolio management problem as a reinforcement learning problem, i.e. defining reward function, state and action spaces. However, before this, we need to specify some constraints on the financial level to ensure that our later specifications of a MDP make sense. We assume the following two hypotheses:

\textbf{Zero market impact}: Buying or selling a stock has no effect on the price process. 

\textbf{Zero slippage}: All trades can be done without any delay and at the price specified. 

The zero market impact assumption is essential to analyze performance of models on past financial data. Anyway, in the real-world, the amount traded is very small compared to the total capital in the market and thus it really has a negligible effect. The zero slippage hypothesis is far less verified in reality. It will be assumed for now but transactions costs/losses due to transaction delays can be considered by a transformation of the reward function as adding some term modelling those losses. In addition, no short-selling is allowed, meaning all weights should be positive and sum to one. This will reduce the number of admissible actions for the agent and, thus, induce faster convergence of the different methods. It also makes sense as in the real world there exist some strict constraints to short-selling.

Now, translating this management portfolio problem into a reinforcement task requires the specification of a Markov decision process, especially, the definition of state and action spaces. We will consider only two simulated stock's returns and, thus, the action space consists simply in choosing the percentage invested in the first stock at the next step. To take advantage of models exploiting the optimal Bellman equations, we consider a finite action space as tens of percent:
\begin{equation*}
    \mathcal{A}=\{0.0,0.1,\ldots, 0.9,1.0\} \quad \text{ and } \quad \lvert \mathcal{A}\rvert = 11
\end{equation*}
On the other hand, the state space is a little bit more complex. It should include previous returns of both stocks. The number of past returns for the state is denoted by window size. A window of size $1$ corresponds to only looking at the returns of the last time-step. Hence, given a window $w$, the state at time $t$ consists in:
\begin{equation*}
    s_t = \begin{pmatrix}
        r_{1,t-w+1}\  \ldots \ r_{1,t}\\
        r_{2,t-w+1}\ \ldots \ r_{2,t}
    \end{pmatrix} \in \mathbb{R}^{2\times w}
\end{equation*}

In the real world the state space is continuous, however in practice we will sometimes use discrete stochastic processes to simulate returns inducing finite state space. Otherwise, if we want to apply methods from the tabular case such as Q-Learning, we will discretize the real values returns. Hence, we introduce the following discretization routine, $d$, which rounds to the closest two decimals number if it lies between some minimal and maximal bounds $x_{min}$, $x_{max}$.
\begin{equation*}
    d(x) = \begin{cases}
        x_{max}, &\text{ if } x \geq x_{max}\\
        x_{min}, &\text{ if } x\leq x_{min}\\
        \lfloor x\cdot 100 + 0.5\rfloor/100, &\text{ otherwise}
    \end{cases}, \qquad \qquad \forall x\in\mathbb{R}
\end{equation*}
Usual values for the bounds are chosen as $x_{min}=0.96$ and $x_{max} = 1.04$. Thus, the state space becomes for window size $w$:
\begin{equation*}
    s_t = \begin{pmatrix}
        d(r_{1,t-w+1})\  \ldots \ d(r_{1,t})\\
        d(r_{2,t-w+1})\ \ldots \ d(r_{2,t})
    \end{pmatrix}  \quad \text{ and } \quad\lvert \mathcal{S}\rvert \leq 9^{2\cdot w}
\end{equation*}

By the hypothesis of zero market impact, actions don't influence the next returns and, thus, transitions depend only on previous states. This will let us consider simpler simulation processes. We still need to define the discount factor, $\gamma$, and the reward function representing the goal of the reinforcement learning task. The goal consists in maximizing financial expected returns over time. Here, we assume that the actor presents no risk aversion, meaning it is only interested in finding a policy maximizing expected financial returns and not caring about volatility of such investment strategy. This goal's definition is also motivated by the fact that, as we introduced it, an agent in reinforcement learning seeks to maximize cumulative expected rewards. Consequently, it isn't able to model higher order terms such as standard deviation of the returns, i.e. the states.

Hence, the agent looks to derive an optimal policy as:
\begin{equation*}
    \pi^\star = \sup_{\pi} \ \mathbb{E}_{\pi}\Big[ \sum_{t=0}^\infty \gamma^t \cdot  r(S_t,A_t,S_{t+1})\Big]
\end{equation*}
\begin{rem}
    This specifies a continuous task as there is no end to the simulated returns.
\end{rem}
Financial returns over a time interval is the multiplication of the returns from each time step in this interval, hence to get equivalence with the RL formulation using a sum and not a product, we define the reward function as:
\begin{equation*}
    r(s_t,a_t,s_{t+1}) = \ln \biggl(\begin{pmatrix}
        a_t\\ 1-a_t
    \end{pmatrix}^T\cdot \begin{pmatrix}
        s_{t+1, (1,w)}\\
         s_{t+1, (2,w)}
    \end{pmatrix}\biggl) =  \ln\Big(
        a_t \cdot r_{1,t+1} +   (1-a_t)\cdot r_{2,t+1}\Big)
\end{equation*}
For notation's simplicity, we define:
\begin{equation*}
    \bm{w}_t = \begin{pmatrix}
        a_t\\ 1-a_t
    \end{pmatrix}
\end{equation*}
and thus the reward function can be directly written as
\begin{equation*}
    r(s_t,a_t,s_{t+1}) = \ln \big( \bm{w}_t^T \cdot \bm{r}_{t+1} \big) 
\end{equation*}
\begin{rem}
    If transaction costs appear in the reward function, as they depend on the change of weights invested, we need to add $a_{t-1}$ to the state $s_t$ to remain consistent with our definition of MDPs.
\end{rem}
Since we are interested to maximize total expected returns, the discount factor should be close to $1$. However, this problem has infinite horizon and, thus, it requires $\gamma < 1$ to ensure that value functions and cumulative rewards are well defined. Hence, we set arbitrarily $\gamma=0.98$.
