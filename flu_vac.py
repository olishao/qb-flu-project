# FLU VACCINATION PROJECT CODE
# SUMMER 2019
# OLIVIA SHAO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

WANING_RATE = 0.022
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
    '''
    n_seasons = int(np.ceil(len(df)/52))
    season_lb, season_ub = 0, 53
    for n in range(n_seasons):
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
## average/aggregate graph for % positive flu tests
mean_pct_pos = []
for week in range(52):
    mean_pct_pos.append(np.mean(df.loc[df['WEEK IN SEASON'] == week + 1, \
        'PCT POSITIVE']))
plt.plot(range(52), mean_pct_pos)
plt.show()
print("peak of flu season for aggregate data: {} weeks after start of season" \
    .format(mean_pct_pos.index(max(mean_pct_pos)) + 1))
## average/aggregate graph for ili
mean_pct_ili = []
for week in range(52):
    mean_pct_ili.append(np.mean(df_ili.loc[df_ili['WEEK IN SEASON'] == \
        week + 1, '% WEIGHTED ILI']))
plt.plot(range(52), mean_pct_ili)
plt.show()
print("peak of ili visits: {} weeks after start of season" \
    .format(mean_pct_ili.index(max(mean_pct_ili)) + 1))
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
## season 1 starts in 1997; season 22 ends in 2019
df_new = df.set_index('WEEK IN SEASON')
df_new.groupby('SEASON')['PCT POSITIVE'].plot(legend=True)
plt.show()
## graph ili by year 
df_ili_new = df_ili.set_index('WEEK IN SEASON')
df_ili_new.groupby('SEASON')['% WEIGHTED ILI'].plot(legend=True)
plt.show()

# def immunity(t, t_vac=0):
#     '''
#     gives distribution for immunity of flu vaccine 
#     inputs:
#         t (array of integers): represents days
#         t_vac (int): day of flu vaccination
#     '''
#     dist = np.exp(-WANING_RATE*t)
#     unvac_days = np.array([0]*t_vac)
#     dist = np.concatenate([unvac_days, dist])
#     dist = dist[0:len(t)]
#     return dist

def immunity(t, t_vac=0):
    '''
    gives distribution for immunity of flu vaccine 
    inputs:
        t (array of integers): represents weeks
        t_vac (int): day of flu vaccination
    '''
    dist = np.exp(-WANING_RATE*t)
    unvac_weeks = np.array([0]*t_vac)
    dist = np.concatenate([unvac_weeks, dist])
    dist = dist[0:len(t)]
    return dist

def flu_foi(t):
    '''
    gives distribution for force of infection of flu
    input:
        t (array of integers): represents days
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
        var (str): variable name for foi
        season: flu season of interest
    returns foi distribution
    '''
    foi_dist = []
    season += 1 # account for difference in indexing
    weeks = (df['SEASON'] == season).sum() # weeks in season
    for week in range(weeks):
        foi_dist.append(df.loc[(df['WEEK IN SEASON'] == week + 1) & \
            (df['SEASON'] == season), var].item())
    return foi_dist

# days_tot = np.arange(0, 180)
# pct_flu_red = []
# for day in days_tot:
#     t_vac = day
#     imm = immunity(days_tot, t_vac)
#     foi = flu_foi(days_tot)
#     pct_flu_red.append(sum(imm*foi))

def flu_red_dist(foi=None, df=None, var=None):
    '''
    graphs distribution for percentage of flu reduction
    inputs: 
        foi: force of infection distribution
    '''
    if foi and not df: 
        weeks_tot = np.arange(0, len(foi))
        pct_flu_red = []
        for week in weeks_tot:
            imm = immunity(weeks_tot, week)
            pct_flu_red.append(sum(imm*foi))
        plt.plot(weeks_tot, pct_flu_red)
        plt.show()
    elif not foi and not df.empty:
        for season in range(N_SEASONS):
            foi = season_foi(df, season, var)
            weeks_tot = np.arange(0, len(foi))
            for week in weeks_tot:
                imm = immunity(weeks_tot, week)
                # print(imm)
                # print(foi)
                df.loc[(df['WEEK IN SEASON'] == week + 1) & (df['SEASON'] \
                    == season + 1), 'PCT FLU REDUCTION'] = sum(imm*foi)
        new_df = df.set_index('WEEK IN SEASON')
        new_df.groupby('SEASON')['PCT FLU REDUCTION'].plot(legend=True)
        plt.show()

flu_red_dist(foi=mean_pct_pos)
flu_red_dist(foi=mean_pct_ili)
flu_red_dist(df=df, var='PCT POSITIVE')
flu_red_dist(df=df_ili, var='% WEIGHTED ILI')