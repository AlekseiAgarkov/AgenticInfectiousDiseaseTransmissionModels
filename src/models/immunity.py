import math


def pascals_snail_day_mapping(day):
    """
    Map day number (0-365) to Pascal's Snail (limaçon) function value.

    Parameters:
    - day: Integer from 0 to 365 representing day of the year

    Returns:
    - Value sampled from Pascal's Snail, with maximum around day 182-183
    """
    # Convert day to angle (0 to 2π)
    # Shift by π/2 to make maximum occur at day ~182 (middle of year)
    theta = (2 * math.pi * day / 365) - (math.pi / 2)

    # Pascal's Snail equation: r = a + b*sin(theta), a and b are taken for the desired shape.
    # Sin rotates the Snail so the lowest values occurs around beginning of the year and highest value occurs mid-year.
    r = 2 + 1 * math.sin(theta)

    return r


def immunity_by_year_day(day, low=0.0, high=1.0):
    """
    Map day number to Pascal's Snail value scaled to arbitrary range.

    Parameters:
    - day: Integer from 0 to 365
    - output_low: Desired minimum output value
    - output_high: Desired maximum output value

    Returns:
    - Scaled value between output_low and output_high
    """
    # Get raw snail value (ranges from 1 to 3)
    raw_value = pascals_snail_day_mapping(day)

    # Rescale from [1, 3] to [output_low, output_high]
    input_min, input_max = 1.0, 3.0
    scaled_value = low + (raw_value - input_min) * (high - low) / (input_max - input_min)

    return scaled_value
