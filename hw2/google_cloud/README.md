## Homework - Policy Gradient Implementation
This code implements the foundational Policy Gradient (PG) algorithm, using  variance reduction techniques such as reward-to-go and advantage normalisation, to train a Reinforcement Learning agent to play simple games like CartPole-v0. My changes are mainly for adjusting parameters and completing the TODO sections in run_hw2.py, pg_agent.py, utils.py, and policies.py.

### cs285/scripts/run_hw2.py modifications
This code guides the overall training loop, including environment interaction and calling the agent's update method. The run_training_loop function was updated to correctly sample trajectories and pass them to the agent for training/updating.
```python
# TODO: sample `args.batch_size` transitions using utils.sample_trajectories
# make sure to use `max_ep_len`
trajs, envsteps_this_batch = utils.sample_trajectories(env, agent.actor, args.batch_size, max_ep_len)
total_envsteps += envsteps_this_batch

# ... (unchanged)

# TODO: train the agent using the sampled trajectories and the agent's update function
train_info: dict = agent.update(
    trajs_dict["observation"],
    trajs_dict["action"],
    trajs_dict["reward"],
    trajs_dict["terminal"]
)
```
### pg_agent.py modifications:
##### 1: Flattening Trajectories: 
The update method now flattens the lists of NumPy arrays for observations, actions, rewards, and terminals into single concatenated arrays.
```python
obs = np.concatenate(obs)
actions = np.concatenate(actions)
rewards = np.concatenate(rewards)
terminals = np.concatenate(terminals)
q_values = np.concatenate(q_values)
```
##### 2: _calculate_q_vals:
This method now correctly implements both the full trajectory discounted return (Case 1) and the reward-to-go discounted return (Case 2) by calling helper functions.
```python
def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
    """Monte Carlo estimation of the Q function."""

    if not self.use_reward_to_go:
        # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
        # trajectory at each point.
        # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
        # TODO: use the helper function self._discounted_return to calculate the Q-values
        q_values = [self._discounted_return(reward) for reward in rewards]
    else:
        # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
        # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        # TODO: use the helper function self._discounted_reward_to_go to calculate the Q-values
        q_values = [self._discounted_reward_to_go(reward) for reward in rewards]

    return q_values
```
##### 3: _estimate_advantage:
This method calculates advantages. For this assignment, **it defaults to q_values if no baseline is used**. Advantage normalisation is also implemented.
```python
def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        if self.critic is None:
            # TODO: if no baseline, then what are the advantages?
            advantages = q_values.copy()
        # ... (sections for baseline and GAE are not active for the initial experiments)

        # TODO: normalize the advantages to have a mean of zero and a standard deviation of one within the batch
        if self.normalize_advantages:
            mean = np.mean(advantages)
            std = np.std(advantages)
            if std > 1e-8:
                advantages = (advantages - mean) / std
            else:
                advantages = (advantages - mean)
        return advantages
```
#### 4: _discounted_return
This helper function calculates the discounted return for an entire trajectory.
```python
def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!
        """
        discounted_sum = 0
        for i, r in enumerate(rewards):
            discounted_sum += (self.gamma ** i) * r
        return np.array([discounted_sum] * len(rewards))
```
#### 5: _discounted_reward_to_go:
This helper function calculates the discounted reward-to-go for each timestep.
```python
def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        """
        rtg = np.zeros_like(rewards, dtype=np.float32)
        cur = 0.0
        for i in reversed(range(len(rewards))):
            cur = rewards[i] + self.gamma * cur
            rtg[i] = cur
        return rtg
```
### utils.py changes:
The sample_trajectory function was updated to allow the policy to select actions and interact with the environment.
```python
# TODO use the most recent ob and the policy to decide what to do
ac: np.ndarray = policy.get_action(ob)

# TODO: use that action to take a step in the environment
next_ob, rew, done, _ = env.step(ac)

# TODO rollout can end due to done, or due to max_length
steps += 1
rollout_done: bool = done or (steps >= max_length)
```

### policies.py changes:
#### 1: MLPPolicy.get_action: 
This method now uses the policy's forward pass to get an action distribution and samples an action from it.

```python
@torch.no_grad()
def get_action(self, obs: np.ndarray) -> np.ndarray:
    """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
    # TODO: implement get_action
    obs_tensor = ptu.from_numpy(obs[None])
    action_distribution = self.forward(obs_tensor)
    action = action_distribution.sample()
    return ptu.to_numpy(action[0])
```
#### 2: MLPPolicy.forward: 
This method defines the forward pass for both discrete and continuous action spaces (for this CartPole problem, we have a discrete action space), returning a torch.distributions.Distribution object.
```python
def forward(self, obs: torch.FloatTensor):
"""
This function defines the forward pass of the network.  You can return anything you want, but you should be
able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
"""
if self.discrete:
    logits = self.logits_net(obs)
    action_distribution = distributions.Categorical(logits=logits)
    return action_distribution
else:
    mean = self.mean_net(obs)
    std = torch.exp(self.logstd) # std = exp(logstd)
    action_distribution = distributions.Normal(loc=mean, scale=std)
    return action_distribution
```
#### 3: MLPPolicyPG.update: 
This method implements the core policy gradient actor update, calculating the loss and performing backpropagation.
```python
class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO: implement the policy gradient actor update.
        self.optimizer.zero_grad()
        dist = self.forward(obs)
        logp  = dist.log_prob(actions)
        loss  = -(logp * advantages).mean()
        loss.backward()
        self.optimizer.step()
        return {
            "Actor Loss": ptu.to_numpy(loss),
        }
```

Command Line Configurations:
````
Small Batch Experiments (batch_size=1000):

python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name cartpole_rtg
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -na --exp_name cartpole_na
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -na --exp_name cartpole_rtg_na
Large Batch Experiments (batch_size=4000):

python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 --exp_name cartpole_lb
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -rtg --exp_name cartpole_lb_rtg
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -na --exp_name cartpole_lb_na
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -rtg -na --exp_name cartpole_lb_rtg_na
```
