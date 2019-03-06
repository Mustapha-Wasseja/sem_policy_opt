import pandas as pd

def run_env(my_market, pricing_agent, n_times=1):
    """
    Runs the market n_times using pricing agent as the protagonist agent

    pricing_agent can be a simple prediction function, or a baseline style RL agent
    If pricing_agent is something more complex, it is wrapped into a function internally by
    wrap_pricing_agent()
    """
    all_data = []
    all_rewards = []

    for i in range(n_times):
        obs = my_market.reset()
        my_market.seed(i)
        episode_reward = 0.

        while not my_market.episode_over:
            my_price = pricing_agent(*obs)
            obs, reward, _, detailed_data = my_market.step(my_price)
            episode_reward += reward
            all_data.append(detailed_data)
        all_rewards.append(episode_reward)
    return pd.Series(all_rewards), pd.DataFrame(all_data)