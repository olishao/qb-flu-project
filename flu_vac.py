# FLU VACCINATION PROJECT CODE
# SUMMER 2019
# OLIVIA SHAO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams["font.family"] = "Tex Gyre Pagella"
WANING_RATE = 0.025
N_SEASONS = 22
SEASON_START_WEEK = 28 # mid-july, with week indices beginning at 0 \
# (actually 29th week of the year)

# 1997-2015 flu surveillance data
df9715 = pd.read_csv('WHO_NREVSS_Combined_prior_to_2015_16.csv', header=1)
# 2015-2019 flu surveillance data (clinical labs)
df1519c = pd.read_csv('WHO_NREVSS_Clinical_Labs.csv', header=1)
# 2015-2019 flu surveillance data (public health labs)
df1519ph = pd.read_csv('WHO_NREVSS_Public_Health_Labs.csv', header=1)
# 1997-2015 influenza like illness data
df_ili = pd.read_csv('ILINet.csv', header=1)

# clean data
df9715['TOTAL POSITIVE'] = df9715.iloc[:, -7:].sum(axis=1)
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
df1519 = df1519.drop(['TOTAL C', 'TOTAL POS C', 'TOTAL PH', 'TOTAL POS PH'], \
    axis=1)
# combine 1997-2015 and 2015-2019 data
df9715.rename(columns={'TOTAL SPECIMENS':'TOTAL'}, inplace=True)
df9715['PERCENT POSITIVE'] = df9715['PERCENT POSITIVE']/100
df9715.rename(columns={'PERCENT POSITIVE':'PCT POSITIVE'}, inplace=True)
df = pd.concat([df9715, df1519], sort=False).reset_index()
# define flu seasons in flu and ili dataframes (for analysis/graphing purposes)
def seasons(df, start_week):
    '''
    adds columns in df for the flu season (numbered 1-22) and week in season
    beginning with 0 (the start of the season being week 29, mid-july)
    input:
        df: dataframe
    '''
    season_lb, season_ub = 0, 52 - (40 - start_week)
    for n in range(N_SEASONS):
        df.loc[season_lb:season_ub, 'SEASON'] = n + 1
        if n == 0: # seasons/years with 53 weeks
            season_lb += 53 - (40 - start_week)
            season_ub += 52 
        elif n == 5 or n == 10 or n == 16:
            season_lb += 52
            season_ub += 53
        elif n == 6 or n == 11 or n == 17:
            season_lb += 53
            season_ub += 52
        else: # normal seasons/years
            season_lb += 52
            season_ub += 52
    df.loc[df['WEEK'] >= start_week, 'WEEK IN SEASON'] = \
        df.loc[df['WEEK'] >= start_week, 'WEEK'] - start_week
    df.loc[(df['WEEK'] < start_week) & (df['SEASON'].isin([1, 7, 12, 18])), \
        'WEEK IN SEASON'] = df.loc[df['WEEK'] < start_week, 'WEEK'] + 53 \
        - start_week
    df.loc[(df['WEEK'] < start_week) & (~df['SEASON'].isin([1, 7, 12, 18])), \
        'WEEK IN SEASON'] = df.loc[df['WEEK'] < start_week, 'WEEK'] + 52 \
        - start_week
seasons(df, SEASON_START_WEEK)
seasons(df_ili, SEASON_START_WEEK)

def scale_foi(df, var):
    '''
    inputs:

    '''
    for n in range(N_SEASONS):
        season_foi = df.loc[df['SEASON'] == n + 1, var]
        if len(season_foi) < 52:
            pass
        else:
            df.loc[df['SEASON'] == n + 1, 'SCALED FOI'] = \
                season_foi/season_foi.sum(axis=0)
scale_foi(df, 'TOTAL POSITIVE')
scale_foi(df_ili, 'ILITOTAL')

# some exploratory/descriptive plotting and analysis
pct_pos = df['PCT POSITIVE']
plt.plot(range(len(df)), pct_pos)
pct_ili = df_ili['% WEIGHTED ILI']
plt.plot(range(len(df_ili)), pct_ili)
plt.show()
def avg_dist(df, var):
    '''
    gives average/aggregate distribution for force of infection based on
    variable var
    inputs:
        df: dataframe
        var: variable in df representing force of infection
    '''
    mean_dist = []
    for week in range(52):
        mean_dist.append(np.mean(df.loc[df['WEEK IN SEASON'] == week, \
            var]))
    return mean_dist
## average/aggregate graphs for % positive flu tests and % ili visits
mean_pct_pos = avg_dist(df, 'PCT POSITIVE')
plt.plot(range(52), mean_pct_pos)
plt.show()
mean_pct_ili = avg_dist(df_ili, '% WEIGHTED ILI')
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
    mean_peak_week = np.mean(peak_week)
    print("mean peak {} is {}, with standard deviation {}".format(var, \
        np.mean(peak_var), np.std(peak_var)))
    if mean_peak_week > 52.143 - (SEASON_START_WEEK + 1):
        print("mean peak week is {}, with standard deviation {}".format( \
            SEASON_START_WEEK + 1 + mean_peak_week - 52.143, np.std(peak_week)))
    else:
        print("mean peak week is {}, with standard deviation {}".format( \
            SEASON_START_WEEK + 1 + mean_peak_week, np.std(peak_week)))
    return peak_week
get_stats(df, 'PCT POSITIVE')
get_stats(df_ili, '% WEIGHTED ILI')
## graph % positive by influenza season beginning week 29
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
        waning_rate (float): waning rate
    '''
    t = t.astype(float)
    dist = np.piecewise(t, [t < 2, t >= 2], [lambda t: np.exp((np.log(2)/2)*t) \
        - 1, lambda t: np.exp(-waning_rate*(t - 2))])
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
        if season == 1:
            week += 40 - SEASON_START_WEEK
        foi_dist.append(df.loc[(df['WEEK IN SEASON'] == week) & \
            (df['SEASON'] == season), var].item())
    return foi_dist

def red_dist_foi(foi, waning_rate):
    '''
    returns x and y to graph distribution for flu reduction
    inputs: 
        foi: force of infection distribution
        waning_rate (float): waning rate
    '''
    weeks_tot = np.arange(0, len(foi))
    pct_flu_red = []
    for week in weeks_tot:
        imm = immunity(weeks_tot, week, waning_rate)
        pct_flu_red.append(sum(imm*foi))
    return np.arange(0, len(foi)), pct_flu_red

def red_dist_df(df, var, waning_rate):
    '''
    returns groupby object to graph time series for flu reduction/protection
    inputs:
        df: dataframe
        var (str): variable in df representing force of infection
        waning_rate (float): waning rate
    '''
    for season in range(N_SEASONS):
        foi = season_foi(df, season, var)
        weeks_tot = np.arange(0, len(foi))
        for week in weeks_tot:
            imm = immunity(weeks_tot, week, waning_rate)
            df.loc[(df['WEEK IN SEASON'] == week) & (df['SEASON'] \
                == season + 1), 'FLU REDUCTION'] = sum(imm*foi)
    new_df = df.set_index('WEEK IN SEASON')
    return new_df.groupby('SEASON')['FLU REDUCTION']

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
def plot_all(df, var, waning_rate, color=False, scale_foi=False):
    '''
    plots vaccine effectiveness, force of infection, and flu reduction 
    inputs:
        df: dataframe
        var (str): variable in df representing force of infection
        waning_rate (float): waning rate
    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    if scale_foi:
        avg_foi = avg_dist(df, 'SCALED FOI')
        foi_distributions, imm_distributions = get_distributions(df, \
            'SCALED FOI', waning_rate)
        grouped = red_dist_df(df, 'SCALED FOI', waning_rate)
    else: 
        avg_foi = avg_dist(df, var)
        foi_distributions, imm_distributions = get_distributions(df, \
            var, waning_rate)
        grouped = red_dist_df(df, var, waning_rate)
    # for dist in imm_distributions:
        # ax1.plot(np.arange(0, 52), dist, color='gray', linewidth=0.3)
    ax1.set_ylim(-0.05, 1.1)
    ax1.set_ylabel('Relative VE')
    ax1.set_xlabel('Time')
    ax2.set_ylabel(' Force of \n Infection')
    ax2.set_xlabel('Time')
    if color:
        for dist in foi_distributions:
            ax2.plot(np.arange(0, len(dist)), dist, linewidth=0.3)
        for g in grouped:
            ax3.plot(g[1], linewidth=0.3)
    else:   
        for dist in foi_distributions:
            ax2.plot(np.arange(0, len(dist)), dist, color='gray', linewidth=0.3)
        for g in grouped:
            ax3.plot(g[1], color='gray', linewidth=0.3)
    ax1.plot(np.arange(0, 52), imm_distributions[0], color='black')
    ax2.plot(np.arange(0, 52), avg_foi, color='black')
    x3, y3 = red_dist_foi(avg_foi, waning_rate)
    ax3.plot(x3, y3, color='black')
    ax3.set_ylabel('Protection')
    ax3.set_xlabel('Time of Vaccination')
    for ax in ([ax1, ax2, ax3]):
        # 4.345 weeks on average in a month
        ax.set_xticks(np.arange(4.345/2, 47.8 + 4.345/2, step=4.345))
        ax.set_xticks(np.arange(0, 53, step=4.345), minor=True)
        ax.set_xticklabels([])
    labels = calendar.month_abbr[7:13] + calendar.month_abbr[1:8]
    ax2.set_xticklabels(labels, fontsize=12, minor=True)
    ax3.set_xticklabels(labels, fontsize=12, minor=True)
    # following code to set font size inspired by ryggyr on Stack Overflow
    # https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
    for item in ([ax1.yaxis.label, ax2.yaxis.label, ax3.yaxis.label, \
        ax1.xaxis.label, ax2.xaxis.label, ax3.xaxis.label]):
        item.set_fontsize(16)
    plt.tight_layout()
    plt.show()
plot_all(df, 'PCT POSITIVE', WANING_RATE, scale_foi=True)
plot_all(df_ili, '% WEIGHTED ILI', WANING_RATE, scale_foi=True)

# individual plots
x_mean_flu, y_mean_flu = red_dist_foi(mean_pct_pos, WANING_RATE)
plt.plot(x_mean_flu, y_mean_flu)
plt.show()

red_dist_df(df, 'PCT POSITIVE', WANING_RATE).plot(legend=True)
plt.show()

x_mean_ili, y_mean_ili = red_dist_foi(mean_pct_ili, WANING_RATE)
plt.plot(x_mean_ili, y_mean_ili)
plt.show()

red_dist_df(df_ili, '% WEIGHTED ILI', WANING_RATE).plot(legend=True)
plt.show()

def plot_flu_red(df, var):
    '''
    plots mean flu reduction/protection curve for varying waning rates
    inputs:
        df: dataframe
        var (str): variable in df representing force of infection
    '''
    distributions = []
    waning_rates = np.linspace(0, 0.1, 6)
    avg_foi = avg_dist(df, var)
    x = np.arange(0, 52)
    a = 0.2
    for wr in waning_rates:
        dist = red_dist_foi(avg_foi, wr)[1]
        distributions.append(dist)
    lines = LineCollection([np.column_stack([x, dist]) for dist in \
        distributions], cmap='viridis_r')
    lines.set_array(waning_rates)
    fig, ax = plt.subplots()
    ax.set_ylim(0, max(distributions[0]))
    ax.set_xlim(0, 52)
    ax.set_ylabel('Protection')
    ax.set_xlabel('Time of Vaccination')
    ax.tick_params(axis="y", labelsize=16)
    ax.set_xticks(np.arange(4.345/2, 47.8 + 4.345/2, step=4.345))
    ax.set_xticks(np.arange(0, 53, step=4.345), minor=True)
    ax.set_xticklabels([])
    labels = calendar.month_abbr[7:13] + calendar.month_abbr[1:8]
    ax.set_xticklabels(labels, fontsize=16, minor=True)
    ax.add_collection(lines)
    ax.yaxis.label.set_fontsize(24)
    ax.xaxis.label.set_fontsize(24)
    axins = inset_axes(ax, width="5%", height="50%", loc='upper left', \
        bbox_to_anchor=(0.75, -0.05, 1, 1), bbox_transform=ax.transAxes,)
    cb = fig.colorbar(lines, cax=axins)
    cb.set_label('Waning Rate', fontsize=24, labelpad=10)
    cb.ax.tick_params(labelsize=14)
    plt.tight_layout()
    plt.show()
plot_flu_red(df, 'PCT POSITIVE')
plot_flu_red(df_ili, '% WEIGHTED ILI')

# get stats for peak reduction
get_stats(df, 'FLU REDUCTION')
get_stats(df_ili, 'FLU REDUCTION')

# plot relationship between waning rate and optimal time to vaccinate
def waningrate_plot(df, var, label, fmt=None):
    '''
    plots vaccination week of max protection for range of waning rates
    inputs:
        df: dataframe
        var (str): variable in df representing force of infection
    '''
    week_max_red = []
    waning_rates = np.linspace(0, 0.1)
    avg_foi = avg_dist(df, var)
    for wr in waning_rates:
        flu_red_dist = red_dist_foi(avg_foi, wr)[1]
        week_max_red.append(flu_red_dist.index(max(flu_red_dist)))
    if fmt:
        plt.plot(waning_rates, week_max_red, fmt, label=label)
    else:
        plt.plot(waning_rates, week_max_red, color='black', label=label)
    plt.ylim([-1, 21.75])
    plt.xlim([-0.0025, 0.1025])
    ax = plt.axes()
    ax.set_yticks(np.arange(4.345/2, 21.75 + 4.345/2, step=4.345))
    ax.set_yticks(np.arange(0, 21.75, step=4.345), minor=True)
    ax.set_yticklabels([])
    labels = calendar.month_abbr[7:13]
    ax.set_yticklabels(labels, rotation=90, fontsize=16, minor=True, va='center')
    ax.set_xticks(np.arange(0, 0.11, step=0.01))
    ax.tick_params(axis="y", length=7,  which='major')
    ax.tick_params(axis="x", labelsize=16)
    plt.ylabel('Optimal Week to Vaccinate')
    plt.xlabel('Weekly Waning Rate')
    ax.yaxis.label.set_fontsize(24)
    ax.xaxis.label.set_fontsize(24)
    plt.legend(loc='best', fontsize=16, borderpad=0.5, borderaxespad=1)
    plt.tight_layout()
waningrate_plot(df, 'PCT POSITIVE', label='% positive flu tests for FOI')
waningrate_plot(df_ili, '% WEIGHTED ILI', label='% ILI visits for FOI', fmt='--k')
plt.show()

import lin_reg
from sklearn.metrics import r2_score

def plot_linreg(dfx, xvar, dfy, yvar):
    '''
    '''
    x = np.array(get_stats(dfx, xvar)).reshape(22, 1)
    y = np.array(get_stats(dfy, yvar)).reshape(22, 1)
    coeffs = lin_reg.model(x, y)
    lin_reg.print_reg_eqn(coeffs)
    print('R^2 =', r2_score(y, lin_reg.fit(x, coeffs)))
    plt.scatter(x, y)
    lin_reg.graph(x, y)
    plt.show()

plot_linreg(df, 'PCT POSITIVE', df, 'FLU REDUCTION')
plot_linreg(df_ili, '% WEIGHTED ILI', df_ili, 'FLU REDUCTION')

plot_linreg(df, 'PCT POSITIVE', df_ili, '% WEIGHTED ILI')
plot_linreg(df, 'FLU REDUCTION', df_ili, 'FLU REDUCTION')