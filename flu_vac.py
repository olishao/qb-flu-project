# FLU VACCINATION PROJECT CODE
# SUMMER 2019
# OLIVIA SHAO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar

WANING_RATE = 0.05
N_SEASONS = 22

# 1997-2015 flu surveillance data
df9715 = pd.read_csv('WHO_NREVSS_Combined_prior_to_2015_16.csv', header=1)
# 2015-2019 flu surveillance data (clinical labs)
df1519c = pd.read_csv('WHO_NREVSS_Clinical_Labs.csv', header=1)
# 2015-2019 flu surveillance data (public health labs)
df1519ph = pd.read_csv('WHO_NREVSS_Public_Health_Labs.csv', header=1)
# 1997-2015 influenza like illness data
df_ili = pd.read_csv('ILINet.csv', header=1)

# clean data
df9715 = df9715.drop(['REGION TYPE', 'REGION', 'A (2009 H1N1)', 'A (H1)', \
    'A (H3)', 'A (Subtyping not Performed)', 'A (Unable to Subtype)', \
    'B', 'H3N2v'], axis=1)
df1519c = df1519c.drop(['REGION TYPE', 'REGION', 'PERCENT A', 'PERCENT B'], \
    axis=1)
df1519ph = df1519ph.drop(['REGION TYPE', 'REGION'], axis=1)
df_ili = df_ili.drop(['REGION TYPE', 'REGION', 'AGE 0-4', 'AGE 25-49', \
    'AGE 25-64', 'AGE 5-24', 'AGE 50-64', 'AGE 65', 'NUM. OF PROVIDERS'], \
    axis=1)
df_ili['% WEIGHTED ILI'] = df_ili['% WEIGHTED ILI']/100
df_ili['% UNWEIGHTED ILI'] = df_ili['% UNWEIGHTED ILI']/100
# combine 2015-2019 clinical and public health lab data
df1519c.rename(columns={'TOTAL SPECIMENS':'TOTAL C'}, inplace=True)
df1519ph.rename(columns={'TOTAL SPECIMENS':'TOTAL PH'}, inplace=True)
df1519c['TOTAL POS C'] = df1519c.loc[:, 'TOTAL A':'TOTAL B'].sum(axis=1)
df1519ph['TOTAL POS PH'] = df1519ph.iloc[:, -7:].sum(axis=1)
df1519c = df1519c.drop(['TOTAL A', 'TOTAL B', 'PERCENT POSITIVE'], axis=1)
df1519ph = df1519ph.drop(['A (2009 H1N1)', 'A (H3)', \
    'A (Subtyping not Performed)', 'B', 'BVic', 'BYam', 'H3N2v'], axis=1)
df1519 = pd.merge(df1519c, df1519ph, how='left', on=['YEAR','WEEK'])
df1519['TOTAL POSITIVE'] = df1519['TOTAL POS C'] + df1519['TOTAL POS PH']
df1519['TOTAL'] = df1519['TOTAL C'] + df1519['TOTAL PH']
df1519['PCT POSITIVE'] = df1519['TOTAL POSITIVE']/df1519['TOTAL']
df1519 = df1519.drop(['TOTAL C', 'TOTAL POS C', 'TOTAL PH', 'TOTAL POS PH', \
    'TOTAL POSITIVE'], axis=1)
# combine 1997-2015 and 2015-2019 data
df9715.rename(columns={'TOTAL SPECIMENS':'TOTAL'}, inplace=True)
df9715['PERCENT POSITIVE'] = df9715['PERCENT POSITIVE']/100
df9715.rename(columns={'PERCENT POSITIVE':'PCT POSITIVE'}, inplace=True)
df = pd.concat([df9715, df1519]).reset_index()
# define flu seasons in flu and ili dataframes (for graphing purposes)
def seasons(df):
    '''
    adds columns in df for the flu season (numbered 1-22) and week in season
    beginning with 1
    input:
        df: dataframe
    '''
    season_lb, season_ub = 0, 53
    for n in range(N_SEASONS):
        df.loc[season_lb:season_ub, 'SEASON'] = n + 1
        if n == 0: # seasons/years with 53 weeks
            season_lb += 53
            season_ub += 51
        elif n == 5 or n == 10 or n == 16:
            season_lb += 52
            season_ub += 53
        elif n == 6 or n == 11 or n == 17:
            season_lb += 53
            season_ub += 52
        else: # normal seasons/years
            season_lb += 52
            season_ub += 52
    # flu season start = week 40 (≈ Oct 1)
    df.loc[df['WEEK'] >= 40, 'WEEK IN SEASON'] = \
        df.loc[df['WEEK'] >= 40, 'WEEK'] - 39
    df.loc[(df['WEEK'] < 40) & (df['SEASON'].isin([1, 7, 12, 18])), \
        'WEEK IN SEASON'] = df.loc[df['WEEK'] < 40, 'WEEK'] + 14
    df.loc[(df['WEEK'] < 40) & (~df['SEASON'].isin([1, 7, 12, 18])), \
        'WEEK IN SEASON'] = df.loc[df['WEEK'] < 40, 'WEEK'] + 13
seasons(df)
seasons(df_ili)

# some exploratory/descriptive plotting and analysis
pct_pos = df['PCT POSITIVE']
plt.plot(range(len(df)), pct_pos)
pct_ili = df_ili['% WEIGHTED ILI']
plt.plot(range(len(df_ili)), pct_ili)
plt.show()
def avg_foi_dist(df, var):
    '''
    gives average/aggregate distribution for force of infection based on
    variable var
    inputs:
        df: dataframe
        var: variable in df representing force of infection
    '''
    mean_foi = []
    for week in range(52):
        mean_foi.append(np.mean(df.loc[df['WEEK IN SEASON'] == week + 1, \
            var]))
    return mean_foi
## average/aggregate graphs for % positive flu tests and % ili visits
mean_pct_pos = avg_foi_dist(df, 'PCT POSITIVE')
plt.plot(range(52), mean_pct_pos)
plt.show()
mean_pct_ili = avg_foi_dist(df_ili, '% WEIGHTED ILI')
plt.plot(range(52), mean_pct_ili)
plt.show()
## stats for peak of flu season and peak ili
def get_stats(df, var):
    '''
    inputs:
        df: dataframe
        var (str): variable name
    '''
    peak_var = []
    peak_week = []
    for n in range(N_SEASONS):
        var_max = max(df.loc[df['SEASON'] == n + 1, var])
        peak_var.append(var_max)
        peak_week.append(int(df.loc[df[var] == var_max, 'WEEK IN SEASON']))
    print("mean peak {} is {}, with standard deviation {}".format(var, \
        np.mean(peak_var), np.std(peak_var)))
    print("mean peak week is {}, with standard deviation {}".format( \
        np.mean(peak_week), np.std(peak_week)))
get_stats(df, 'PCT POSITIVE')
get_stats(df_ili, '% WEIGHTED ILI')
## graph % positive by influenza season beginning week 40
# season 1 starts in 1997; season 22 ends in 2019
df_new = df.set_index('WEEK IN SEASON')
df_new.groupby('SEASON')['PCT POSITIVE'].plot(legend=True)
plt.show()
## graph % ili visits by influenza season 
df_ili_new = df_ili.set_index('WEEK IN SEASON')
df_ili_new.groupby('SEASON')['% WEIGHTED ILI'].plot(legend=True)
plt.show()

def immunity(t, t_vac, waning_rate):
    '''
    gives simulated distribution for immunity to flu after vaccination
    inputs:
        t (array of integers): represents weeks
        t_vac (int): day of flu vaccination
    '''
    t = t.astype(float)
    # piecewise function needs to be fixed
    # right now hard-coded for waning rate of 0.05
    dist = np.piecewise(t, [t < 2, t >= 2], [lambda t: np.exp(0.346575*t - 1), \
       lambda t: np.exp(-waning_rate*t) + 0.0952])
    # dist = np.exp(-waning_rate*t)
    unvac_weeks = np.array([0]*t_vac)
    dist = np.concatenate([unvac_weeks, dist])
    dist = dist[0:len(t)]
    return dist

def flu_foi(t):
    '''
    gives simulated distribution for a flu force of infection curve
    input:
        t (array of integers): represents days/weeks
    '''
    a = 1/2
    k = 1/30
    p = np.pi
    b = 0.5
    return a*np.cos(k*t + p) + b

def season_foi(df, season, var):
    '''
    inputs:
        df: dataframe
        var (str): variable in df representing force of infection
        season (int): flu season of interest
    returns foi distribution
    '''
    foi_dist = []
    season += 1 # account for difference in indexing
    weeks = (df['SEASON'] == season).sum() # weeks in season
    for week in range(weeks):
        foi_dist.append(df.loc[(df['WEEK IN SEASON'] == week + 1) & \
            (df['SEASON'] == season), var].item())
    return foi_dist

def red_dist_foi(foi, waning_rate):
    '''
    returns x and y to graph distribution for flu reduction
    inputs: 
        foi: force of infection distribution
    '''
    weeks_tot = np.arange(0, len(foi))
    pct_flu_red = []
    for week in weeks_tot:
        imm = immunity(weeks_tot, week, waning_rate)
        pct_flu_red.append(sum(imm*foi))
    return np.arange(1, len(foi) + 1), pct_flu_red
    # return plt.plot(np.arange(1, len(foi) + 1), pct_flu_red)

def red_dist_df(df, var, waning_rate):
    '''
    returns groupby object to graph time series for flu reduction
    inputs:
        df: dataframe
        var (str): variable in df representing force of infection
    '''
    for season in range(N_SEASONS):
        foi = season_foi(df, season, var)
        weeks_tot = np.arange(0, len(foi))
        for week in weeks_tot:
            imm = immunity(weeks_tot, week, waning_rate)
            df.loc[(df['WEEK IN SEASON'] == week + 1) & (df['SEASON'] \
                == season + 1), 'FLU REDUCTION'] = sum(imm*foi)
    new_df = df.set_index('WEEK IN SEASON')
    return new_df.groupby('SEASON')['FLU REDUCTION']
    # return new_df.groupby('SEASON')['FLU REDUCTION'].plot(alpha=0.4, \
        # legend=True)

def get_distributions(df, var, waning_rate):
    '''
    finds distributions for force of infection and immunity
    inputs:
        df: dataframe
        var (str): variable in df representing force of infection
    '''
    foi_distributions = []
    for season in range(N_SEASONS):
        foi = season_foi(df, season, var)
        weeks_tot = np.arange(0, len(foi))
        foi_distributions.append(foi)
    weeks_tot = np.arange(0, 52)
    imm_distributions = []
    for week in weeks_tot:
        imm_distributions.append(immunity(weeks_tot, week, waning_rate))
    return foi_distributions, imm_distributions

# three-panel plots
def plot_all(df, var, waning_rate):
    '''
    plots vaccine effectiveness, force of infection, and flu reduction 
    on one plot
    inputs:
        df: dataframe
        var (str): variable in df representing force of infection
    '''
    avg_foi = avg_foi_dist(df, var)
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    foi_distributions, imm_distributions = get_distributions(df, var, waning_rate)
    # for dist in imm_distributions:
        # ax1.plot(np.arange(1, 53), dist)
    ax1.plot(np.arange(1, 53), imm_distributions[0], color='black')
    ax1.set_ylabel('Vaccine Effectiveness')
    for dist in foi_distributions:
        ax2.plot(np.arange(1, len(dist) + 1), dist, color='gray', linewidth=0.3)
    ax2.plot(np.arange(1, 53), avg_foi, color='black')
    ax2.set_ylabel('Force of Infection')
    grouped = red_dist_df(df, var, waning_rate)
    for g in grouped:
        ax3.plot(g[1], color='gray', linewidth=0.3)
    x3, y3 = red_dist_foi(avg_foi, waning_rate)
    ax3.plot(x3, y3, color='black')
    ax3.set_ylabel('Flu Reduction')
    ax3.set_xlabel('Time')
    ax3.set_xticks(np.arange(1, 48.67, step=4.33))
    labels = calendar.month_abbr[10:13] + calendar.month_abbr[1:10]
    ax3.set_xticklabels(labels, rotation=90, fontsize=8)
    for ax in ([ax1, ax2]):
        ax.set_xticks(np.arange(1, 48.67, step=4.33))
        ax.set_xticklabels([])
    # following code to set font size inspired by ryggyr on Stack Overflow
    # https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
    for item in ([ax1.yaxis.label, ax2.yaxis.label, ax3.yaxis.label, \
        ax3.xaxis.label]):
        item.set_fontsize(8)
    plt.show()
plot_all(df, 'PCT POSITIVE', WANING_RATE)
plot_all(df_ili, '% WEIGHTED ILI', WANING_RATE)

# individual plots
x_mean_flu, y_mean_flu = red_dist_foi(foi=mean_pct_pos, WANING_RATE)
plt.plot(x_mean_flu, y_mean_flu)
plt.show()

red_dist_df(df=df, var='PCT POSITIVE', WANING_RATE).plot(legend=True)
plt.show()

x_mean_ili, y_mean_ili = red_dist_foi(foi=mean_pct_ili, WANING_RATE)
plt.plot(x_mean_ili, y_mean_ili)
plt.show()

red_dist_df(df=df_ili, var='% WEIGHTED ILI', WANING_RATE).plot(legend=True)
plt.show()

# plot relationship between waning rate and optimate time to vaccinate
def waningrate_plot(df, var):
    '''
    plots vaccination week of max protection for range of waning rates
    inputs:
        df: dataframe
        var (str): variable in df representing force of infection
    '''
    week_max_red = []
    waning_rates = np.linspace(0, 0.2)
    for wr in waning_rates:
        flu_red_dist = red_dist_foi(avg_foi_dist(df, var), wr)[1]
        week_max_red.append(flu_red_dist.index(max(flu_red_dist)) + 1)
    plt.plot(week_max_red, waning_rates, color='black')
    ax = plt.axes()
    ax.set_xticks(np.arange(1, 14, step=4.33))
    labels = calendar.month_abbr[10:13] + calendar.month_abbr[1:2]
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    plt.xlabel('Optimal Week to Vaccinate')
    plt.ylabel('Waning Rate')
    plt.show()

waningrate_plot(df, 'PCT POSITIVE')
waningrate_plot(df_ili, '% WEIGHTED ILI')
