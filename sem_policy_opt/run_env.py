import pandas as pd
# import stable_baselines    # disabled until I install it in kernels

def run_env(my_market, pricing_agent, n_times=1):
    """
    Runs the market n_times using pricing agent as the protagonist agent

    pricing_agent can be a simple prediction function, or a baseline style RL agent
    If pricing_agent is something more complex, it is wrapped into a function internally by
    wrap_pricing_agent()
    """
    pricing_fn = wrap_pricing_agent(pricing_agent)
    all_data = []
    all_rewards = []

    for _ in range(n_times):
        obs = my_market.reset()
        episode_reward = 0

        while not my_market.episode_over:
            my_price = pricing_fn(obs)
            obs, reward, _, detailed_data = my_market.step(my_price)
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
    # TODO: Uncomment after fixing stable_baselines in kernels
    #elif isinstance(raw_agent, stable_baselines.common.base_class.BaseRLModel):
    #    def pricing_fn(arg_array):
    #        return raw_agent.predict(arg_array)[0][0]   # first [0] gets the action and removes LSTM specific stuff. Second [0] converts to scalar
    #    return pricing_fn
    assert NotImplementedError
