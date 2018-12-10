import numpy as np


def get_true_qty_demanded_fn(potential_customers_per_day, customer_level_randomness):
    def true_calc_q(jetblue_price, delta_price, demand_level, jetblue_seats_avail, delta_seats_avail):
        jetblue_customer_level_demand = demand_level + np.random.randn(potential_customers_per_day) * customer_level_randomness
        jetblue_consumer_surplus = jetblue_customer_level_demand - jetblue_price

        delta_customer_level_demand = demand_level + np.random.randn(potential_customers_per_day) * customer_level_randomness
        delta_consumer_surplus = delta_customer_level_demand - delta_price

        jetblue_seats_demanded = ((jetblue_consumer_surplus > delta_consumer_surplus) * (jetblue_consumer_surplus > 0)).sum()
        delta_seats_demanded = ((delta_consumer_surplus > jetblue_consumer_surplus) * (delta_consumer_surplus > 0)).sum()
        return jetblue_seats_demanded, delta_seats_demanded
    return true_calc_q

