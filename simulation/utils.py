import numpy as np
import pandas as pd

def calculate_gini(wealth_array):
    """
    Calculate the Gini coefficient from an array of wealth values.
    
    Args:
        wealth_array: Array of wealth values
        
    Returns:
        Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    """
    # Sort wealth in ascending order
    sorted_wealth = np.sort(wealth_array)
    n = len(sorted_wealth)
    
    # Edge case: if all wealth is 0 or array is empty
    if n == 0 or np.sum(sorted_wealth) == 0:
        return 0
    
    # Calculate Gini coefficient
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * sorted_wealth)) / (n * np.sum(sorted_wealth))

def calculate_wealth_percentiles(wealth_array):
    """
    Calculate key wealth percentiles.
    
    Args:
        wealth_array: Array of wealth values
        
    Returns:
        Dictionary with percentile values
    """
    # Convert to numpy array if it's not already
    wealth = np.array(wealth_array)
    
    # Calculate percentiles
    p10 = np.percentile(wealth, 10)
    p25 = np.percentile(wealth, 25)
    p50 = np.percentile(wealth, 50)
    p75 = np.percentile(wealth, 75)
    p90 = np.percentile(wealth, 90)
    p99 = np.percentile(wealth, 99)
    
    return {
        'p10': p10,
        'p25': p25,
        'p50': p50,
        'p75': p75,
        'p90': p90,
        'p99': p99
    }

def calculate_wealth_shares(wealth_array):
    """
    Calculate the share of total wealth held by different groups.
    
    Args:
        wealth_array: Array of wealth values
        
    Returns:
        Dictionary with wealth shares by group
    """
    # Convert to numpy array and ensure it's sorted
    wealth = np.array(wealth_array)
    sorted_wealth = np.sort(wealth)
    total_wealth = np.sum(sorted_wealth)
    
    # Edge case: if total wealth is 0
    if total_wealth == 0:
        return {'top1': 0, 'top10': 0, 'bottom50': 0}
    
    n = len(sorted_wealth)
    
    # Calculate wealth shares
    top1_idx = int(n * 0.99)  # Index for top 1%
    top10_idx = int(n * 0.9)   # Index for top 10%
    mid_idx = int(n * 0.5)     # Index for bottom 50%
    
    top1_wealth = np.sum(sorted_wealth[top1_idx:])
    top10_wealth = np.sum(sorted_wealth[top10_idx:])
    bottom50_wealth = np.sum(sorted_wealth[:mid_idx])
    
    # Calculate shares as percentages
    top1_share = (top1_wealth / total_wealth) * 100
    top10_share = (top10_wealth / total_wealth) * 100
    bottom50_share = (bottom50_wealth / total_wealth) * 100
    
    return {
        'top1': top1_share,
        'top10': top10_share,
        'bottom50': bottom50_share
    }

def calculate_economic_mobility(wealth_history, years=10):
    """
    Calculate economic mobility metrics.
    
    Args:
        wealth_history: List of arrays containing wealth values for each year
        years: Number of years to consider for mobility calculations
        
    Returns:
        Dictionary with mobility metrics
    """
    if len(wealth_history) < years:
        return None
    
    # Get initial and final wealth for the period
    initial_wealth = np.array(wealth_history[0])
    final_wealth = np.array(wealth_history[years-1])
    
    n = len(initial_wealth)
    
    # Calculate initial quintiles
    quintile_size = n // 5
    initial_sorted_indices = np.argsort(initial_wealth)
    initial_quintiles = np.zeros(n, dtype=int)
    
    for i in range(5):
        start_idx = i * quintile_size
        end_idx = (i + 1) * quintile_size if i < 4 else n
        initial_quintiles[initial_sorted_indices[start_idx:end_idx]] = i
    
    # Calculate final quintiles
    final_sorted_indices = np.argsort(final_wealth)
    final_quintiles = np.zeros(n, dtype=int)
    
    for i in range(5):
        start_idx = i * quintile_size
        end_idx = (i + 1) * quintile_size if i < 4 else n
        final_quintiles[final_sorted_indices[start_idx:end_idx]] = i
    
    # Calculate transition matrix (rows = initial quintile, columns = final quintile)
    transition_matrix = np.zeros((5, 5))
    
    for i in range(5):
        initial_i = np.where(initial_quintiles == i)[0]
        if len(initial_i) == 0:
            continue
            
        for j in range(5):
            count = np.sum(final_quintiles[initial_i] == j)
            transition_matrix[i, j] = count / len(initial_i)
    
    # Calculate mobility metrics
    # Upward mobility: probability of moving up at least one quintile
    upward_mobility = np.sum([
        transition_matrix[i, j] * (len(np.where(initial_quintiles == i)[0]) / n)
        for i in range(4) for j in range(i+1, 5)
    ])
    
    # Downward mobility: probability of moving down at least one quintile
    downward_mobility = np.sum([
        transition_matrix[i, j] * (len(np.where(initial_quintiles == i)[0]) / n)
        for i in range(1, 5) for j in range(i)
    ])
    
    # Absolute mobility: average absolute change in wealth
    wealth_changes = final_wealth - initial_wealth
    absolute_mobility = np.mean(np.abs(wealth_changes)) / max(1, np.mean(initial_wealth))
    
    return {
        'upward_mobility': upward_mobility,
        'downward_mobility': downward_mobility,
        'absolute_mobility': absolute_mobility,
        'transition_matrix': transition_matrix
    }
