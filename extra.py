# ADDITIONAL CODE

from flu_vac import *

pct_pos = df['PCT POSITIVE']
plt.plot(range(len(df)), pct_pos, label='percent positive')
pct_ili = df_ili['% WEIGHTED ILI']
plt.plot(range(len(df_ili)), pct_ili, label='percent ILI visits')
plt.legend(loc='best')
plt.show()
scaled_flu = df['SCALED TOTAL POSITIVE']
plt.plot(range(len(df)), scaled_flu, label='positive tests scaled')
scaled_ili = df_ili['SCALED ILITOTAL']
plt.plot(range(len(df_ili)), scaled_ili, label='ILI visits scaled')
plt.legend(loc='best')
plt.show()

## average/aggregate graphs for % positive flu tests and % ili visits
mean_pct_pos = avg_dist(df, 'PCT POSITIVE')
plt.plot(range(52), mean_pct_pos, label='percent positive tests')
mean_pct_ili = avg_dist(df_ili, '% WEIGHTED ILI')
plt.plot(range(52), mean_pct_ili, label='percent ILI visits')
plt.legend(loc='best')
plt.show()
## average/aggregate graphs for scaled incidence variables
mean_pos_flu = avg_dist(df, 'SCALED TOTAL POSITIVE')
plt.plot(range(52), mean_pos_flu, label='positive tests scaled')
mean_ili_vis = avg_dist(df_ili, 'SCALED ILITOTAL')
plt.plot(range(52), mean_ili_vis, label='ILI visits scaled')
plt.legend(loc='best')
plt.show()

get_stats(df, 'PCT POSITIVE')
get_stats(df, 'TOTAL POSITIVE')
get_stats(df_ili, '% WEIGHTED ILI')
get_stats(df_ili, 'ILITOTAL')

## graph % positive by influenza season beginning week 29
### season 1 starts in 1997; season 22 ends in 2019
df_new = df.set_index('WEEK IN SEASON')
df_new.groupby('SEASON')['PCT POSITIVE'].plot(legend=True)
plt.show()
### graph % ili visits by influenza season 
df_ili_new = df_ili.set_index('WEEK IN SEASON')
df_ili_new.groupby('SEASON')['% WEIGHTED ILI'].plot(legend=True)
plt.show()

# plot protection distribution using pct positive for incidence
mean_pct_pos = avg_dist(df, 'PCT POSITIVE')
x_mean_flu, y_mean_flu = red_dist_inc(mean_pct_pos, waning_rate=WANING_RATE)
plt.plot(x_mean_flu, y_mean_flu)
plt.show()
# plot protection distribution using pct weighted ili for incidence
mean_pct_ili = avg_dist(df_ili, '% WEIGHTED ILI')
x_mean_ili, y_mean_ili = red_dist_inc(mean_pct_ili, waning_rate=WANING_RATE)
plt.plot(x_mean_ili, y_mean_ili)
plt.show()

get_iqr(df, 'FLU REDUCTION')