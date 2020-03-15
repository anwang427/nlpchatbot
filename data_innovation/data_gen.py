"""
Characteristics of a SMS message:
    Topic (Chikoo, wheat, ...) -> (t_1, t_2, ..., t_T)
    Query (Fertilizer, location to buy, ...) -> (q_1, q_2, ..., q_Q)
    Time (continuous) -> (n_1, n_2, ..., n_N)
    Region (Area 1, Area 2, ...) -> (r_1, r_2, ..., r_R)
    Weather (Clear skies, rainy, ...)
    Season (Early summer , mid summer, ...)

Each message will be encoded as a bag-of-words. Number of elements will be equal to the sum of all the total of possibilities.
For example:
    Caring for chikoo tree. What fertilizer should I use?
        Topic: Chikoo
        Query: Fertilizer
        Time: 955pm
        Region: Toronto, Canada
        Weather: Clear skies
        Season: Winter

Assumptions made in this simulation:
    SMS elements are independent from one another (which is not true!)
    More below

^So in reality, there will be more complex mechanisms in play.
""" 

import numpy as np


class SmsElements:
    queries = { # Frequencies out of 175
        "new crop production": 124,
        "seeds availability": 130,
        "insecticide availability": 109,
        "fertilizer availability": 60,
        "water management": 113,
        "weather information": 41,
        "new agricultural equipments": 31
    }
    time = { # Key is the upperbound in hours. Value is the estimated probability. Hours in integers
        6: 0.05,
        12: 0.35,
        18: 0.45,
        23: 0.15
    }
    district = 36 # Number of districts in Maharashtra. Assuming equal probability for any of the 36
    season = {
        'winter': 0.17, #2/12,
        'summer': 0.25, #3/12,
        'monsoon': 0.33, #4/12,
        'post-monsoon': 0.25 # 3/12
    }

    word_vector_length = len(queries) + 24 + district + len(season)


def create_texts(num_texts):
    # Devising query probabilities
    query_probs = np.array(list(SmsElements.queries.values())) / 175.
    query_probs_normalized = query_probs / sum(query_probs)

    # Devising time probabilities
    time_probs = np.zeros(24)
    for hour in range(24):
        for upper_bound in np.sort(list(SmsElements.time.keys())):
            if hour <= upper_bound:
                time_probs[hour] = SmsElements.time[upper_bound]
                break 
    time_probs = time_probs / sum(time_probs)

    # Creating word vectors
    word_vectors = []
    for i in range(num_texts):
        # 1. Forming query
        #(option 1)
        # for j, word_prob in enumerate(query_probs):
        #     if np.random.uniform(0, 1) < word_prob:
        #         new_word = 1

        #(option 2)
        num_queries = len(SmsElements.queries)
        query_vector = np.eye(num_queries)[np.random.choice(num_queries, p=query_probs_normalized)]
        # print("q", query_vector)

        # 2. Forming time
        num_time = len(time_probs)
        time_vector = np.eye(num_time)[np.random.choice(num_time, p=time_probs)]
        # print("t", time_vector)

        # 3. Forming district
        district_vector = np.eye(SmsElements.district)[np.random.choice(SmsElements.district)]
        # print("d", district_vector)

        #4. Forming season
        num_season = len(SmsElements.season)
        season_vector = np.eye(num_season)[np.random.choice(num_season, p=list(SmsElements.season.values()))]
        # print("s", season_vector)

        # 5. Combining vectors to form one word vector
        word_vectors.append(np.hstack([query_vector, time_vector, district_vector, season_vector]))
    
    print("Successfully generated {n} word vectors".format(n=num_texts))
    return word_vectors

if __name__ == "__main__":
    print(create_texts(1))