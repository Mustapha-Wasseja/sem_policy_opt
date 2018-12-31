import pandas as pd
import stable_baselines

def run_env(my_market, pricing_agent, n_times=1):

    pricing_fn = wrap_pricing_agent(pricing_agent)
    all_data = []
    all_rewards = []

    for _ in range(n_times):
        obs = my_market.reset()
        episode_over = False
        episode_reward = 0

        while not episode_over:
            my_price = pricing_fn(obs)
            obs, reward, episode_over, detailed_data = my_market.step(my_price)
            episode_reward += reward
            all_data.append(detailed_data)
            
        all_rewards.append(episode_reward)
    return pd.Series(all_rewards), pd.DataFrame(all_data)


def wrap_pricing_agent(raw_agent):
    if callable(raw_agent):
        # raw functions are assumed to take individual args as parameters rather than a single numpy array
        def pricing_fn(arg_array):
            return raw_agent(*arg_array)
        return pricing_fn
    elif isinstance(raw_agent, stable_baselines.common.base_class.BaseRLModel):
        def pricing_fn(arg_array):
            return raw_agent.predict(arg_array)[0][0]   # first [0] gets the action and removes LSTM specific stuff. Second [0] converts to scalar
        return pricing_fn
    assert NotImplementedError
        