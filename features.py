import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt


def information_coeff(data: pd.DataFrame, param1: str, param2: str):
    clean_data = data.dropna(subset=[param1, param2])
    ic, p_value = stats.spearmanr(clean_data[param1], clean_data[param2])

    pearson_ic, pearson_p = stats.pearsonr(clean_data[param1], clean_data[param2])
    
    print(f"--- Research Results ---")
    print(f"Spearman IC (Rank): {ic:.4f}")
    print(f"P-Value: {p_value:.4f}")
    print(f"-" * 24)
    print(f"Pearson IC (Linear):{pearson_ic:.4f}")
    
    return ic, p_value

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_decile_analysis(data, feature_col='trend_5', target_col='next_day_returns'):
    # 1. Create a copy to avoid SettingWithCopy warnings
    df_plot = data.copy().dropna(subset=[feature_col, target_col])
    
    # 2. Create Deciles (10 Bins)
    # qcut divides the data into equal-sized buckets
    df_plot['decile'] = pd.qcut(df_plot[feature_col], 20, labels=False, duplicates='drop')
    
    # 3. Calculate Mean Return per Decile
    # We group by the bin and take the average of the NEXT day's return
    decile_means = df_plot.groupby('decile')[target_col].mean()
    
    # 4. Plotting
    plt.figure(figsize=(10, 6))
    
    # We use a bar plot to visualize the "staircase"
    barplot = sns.barplot(x=decile_means.index, y=decile_means.values, color='steelblue')
    
    # Add a zero line for reference (Crucial!)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    
    plt.title(f"Decile Analysis: {feature_col} vs Mean {target_col}", fontsize=14)
    plt.xlabel("Trend Decile (0 = Lowest Trend, 9 = Highest Trend)", fontsize=12)
    plt.ylabel("Average Next Day Return", fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Labeling the bars for clarity
    for i, v in enumerate(decile_means.values):
        barplot.text(i, v, f'{v:.4f}', color='black', ha="center", va="bottom" if v > 0 else "top", fontsize=9)
        
    plt.show()


def plot_yearly_analysis(data: pd.DataFrame, param1: str, param2: str):
    df = data.copy()
    
    df = df.dropna(subset=[param1, param2])
    df['bin'] = pd.qcut(df[param1], 20, labels=False)

    extremes = df[df['bin'].isin([0,19])]

    yearly_perf = extremes.groupby(['year', 'bin'])[param2].mean().unstack()
    
    yearly_perf['strat_edge'] = yearly_perf[0] - yearly_perf[19]

    plt.figure(figsize=(12,6))

    colors= ['green' if x > 0 else 'red' for x in yearly_perf['strat_edge']]

    sns.barplot(x=yearly_perf.index, y=yearly_perf['strat_edge'], palette=colors)

    plt.axhline(0, color='black', linewidth=1)
    plt.title("Year-over-Year Strategy Stability (Long Bin 0 vs Short Bin 19)")
    plt.ylabel("Average Daily Edge (Spread)")
    plt.xlabel("Year")
    plt.grid(axis='y', alpha=0.3)
    
    plt.show()
    
    # Print the raw numbers for verification
    print("--- Annual Strategy Edge ---")
    print(yearly_perf[['strat_edge']])
