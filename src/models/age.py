from typing import List, Dict

import numpy as np


def sample_age(age_ranges: Dict[int, List[int]], age_probs: Dict[int, float]):
    age_range_key = np.random.choice(list(age_ranges.keys()), p=list(age_probs.values()))

    return (age_range_key,
            np.random.randint(low=age_ranges[age_range_key][0],
                              high=age_ranges[age_range_key][1]))
