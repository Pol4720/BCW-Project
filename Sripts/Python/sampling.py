import pandas as pd
import numpy as np
from scipy import stats

def sample_statistics(data_path, variable, sample_size):
    """
    Calculate and print sample statistics: mean, variance, and standard deviation.

    Parameters:
    data_path (str): Path to the dataset.
    variable (str): Name of the variable to sample.
    sample_size (int): Number of samples to draw.

    Returns:
    None
    """
    data = pd.read_csv(data_path)
    sample = data[variable].sample(n=sample_size)

    mean = sample.mean()
    variance = sample.var()
    std_dev = sample.std()

    print(f"Mean: {mean}")
    print(f"Variance: {variance}")
    print(f"Standard Deviation: {std_dev}")

def hypothesis_test(sample_size, test_type, tails, tail_orientation, limit_value, significance_level):
    """
    Perform a hypothesis test on the mean or variance assuming a normal population.

    Parameters:
    sample_size (int): Size of the sample.
    test_type (str): Type of test ('mean' or 'variance').
    tails (int): Number of tails (1 or 2).
    tail_orientation (str): Orientation of the tail if one-tailed ('left' or 'right').
    limit_value (float): The value to compare the sample against.
    significance_level (float): The significance level of the test.

    Returns:
    None
    """
    # Generate a random sample from a normal distribution
    sample = np.random.normal(loc=0, scale=1, size=sample_size)

    if test_type == 'mean':
        sample_mean = sample.mean()
        sample_std = sample.std(ddof=1)
        t_statistic = (sample_mean - limit_value) / (sample_std / np.sqrt(sample_size))

        if tails == 2:
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), df=sample_size-1))
        elif tails == 1:
            if tail_orientation == 'left':
                p_value = stats.t.cdf(t_statistic, df=sample_size-1)
            elif tail_orientation == 'right':
                p_value = 1 - stats.t.cdf(t_statistic, df=sample_size-1)

    elif test_type == 'variance':
        sample_variance = sample.var(ddof=1)
        chi_square_statistic = (sample_size - 1) * sample_variance / limit_value

        if tails == 2:
            p_value = 2 * min(stats.chi2.cdf(chi_square_statistic, df=sample_size-1), 1 - stats.chi2.cdf(chi_square_statistic, df=sample_size-1))
        elif tails == 1:
            if tail_orientation == 'left':
                p_value = stats.chi2.cdf(chi_square_statistic, df=sample_size-1)
            elif tail_orientation == 'right':
                p_value = 1 - stats.chi2.cdf(chi_square_statistic, df=sample_size-1)

    print(f"Test Statistic: {t_statistic if test_type == 'mean' else chi_square_statistic}")
    print(f"P-value: {p_value}")
    print(f"Significance Level: {significance_level}")
    print(f"Reject Null Hypothesis: {p_value < significance_level}")

if __name__ == "__main__":
    import sys

    print("Choose a function to call:")
    print("1. sample_statistics")
    print("2. hypothesis_test")
    choice = input("Enter the number of your choice: ")

    if choice == "1":
        data_path = input("Enter the data path: ")
        variable = input("Enter the variable name: ")
        sample_size = int(input("Enter the sample size: "))
        sample_statistics(data_path, variable, sample_size)
    elif choice == "2":
        sample_size = int(input("Enter the sample size: "))
        test_type = input("Enter the test type ('mean' or 'variance'): ")
        tails = int(input("Enter the number of tails (1 or 2): "))
        tail_orientation = input("Enter the tail orientation ('left' or 'right'): ")
        limit_value = float(input("Enter the limit value: "))
        significance_level = float(input("Enter the significance level: "))
        hypothesis_test(sample_size, test_type, tails, tail_orientation, limit_value, significance_level)
    else:
        print("Invalid choice")