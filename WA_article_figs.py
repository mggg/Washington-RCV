import pandas as pd
import matplotlib.pyplot as plt
import os
import geopandas as gpd
import matplotlib
import numpy as np


figs_dir = "./figs/"
os.makedirs(os.path.dirname(figs_dir), exist_ok=True)

#demog map
polity = 'WA'
run_name = polity + '_demog_'
demog_list = ['hvap','avap','wvap','pocvap','hcvap','acvap','wcvap','poccvap']
demog_names = {'hvap':'HVAP','avap':'AVAP','wvap':'WVAP','pocvap':'POC_VAP','hcvap':'HCVAP','acvap':'ACVAP','wcvap':'WCVAP','poccvap':'POC_CVAP'}
inset_bounds = [-13675000, 5950000,-13550000, 6125000]

polity_shapefile_path = './WA_bg_w_cvap_data/WA_bg_w_cvap_data.shp'  #for shapefile
gdf = gpd.read_file(polity_shapefile_path)
gdf = gdf.rename(columns = {'CVAP19':'CVAP', 'AIANCVAP19':'AIANCVAP', 'ACVAP19':'ACVAP', 'BCVAP19':'BCVAP','NHPICVAP19':'NHPICVAP', 'WCVAP19':'WCVAP', 'HCVAP19':'HCVAP', 'OCVAP19':'OCVAP'})
gdf[['CVAP', 'AIANCVAP', 'ACVAP', 'BCVAP','NHPICVAP', 'WCVAP', 'HCVAP', 'OCVAP']] = gdf[['CVAP', 'AIANCVAP', 'ACVAP', 'BCVAP','NHPICVAP', 'WCVAP', 'HCVAP', 'OCVAP']].fillna(0)
polity_block_data = pd.DataFrame(gdf.drop('geometry', axis=1))
gdf['hvap_pct'] = gdf['HVAP']/gdf['TOTVAP']
gdf['avap_pct'] = gdf['NH_AVAP']/gdf['TOTVAP']
gdf['wvap_pct'] = gdf['NH_WVAP']/gdf['TOTVAP']
gdf['pocvap_pct'] = 1-gdf['wvap_pct']
gdf['hcvap_pct'] = gdf['HCVAP']/gdf['CVAP']
gdf['acvap_pct'] = gdf['ACVAP']/gdf['CVAP']
gdf['wcvap_pct'] = gdf['WCVAP']/gdf['CVAP']
gdf['poccvap_pct'] = 1-gdf['wcvap_pct']
gdf = gdf.to_crs('EPSG:3857')
for demog in demog_list:
    fig,ax = plt.subplots()
    gdf.plot(column=demog+'_pct', cmap = 'Spectral_r',ax = ax, vmin = 0, vmax = 1, legend=True, legend_kwds = {'label': "Percent "+demog_names[demog],'shrink': 0.6}, missing_kwds={'color': 'grey'}, edgecolor='black', linewidth = .05)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(figs_dir+run_name+'_'+demog+'_pct_map.png', dpi = 600)
    plt.close("all")
    fig,ax = plt.subplots()
    gdf.plot(column=demog+'_pct', cmap = 'Spectral_r',ax = ax, vmin = 0, vmax = 1, legend=True, legend_kwds = {'label': "Percent "+demog_names[demog]}, missing_kwds={'color': 'grey'}, edgecolor='black', linewidth = .05)
    ax.set_axis_off()
    ax.set_xlim(inset_bounds[0], inset_bounds[2])
    ax.set_ylim(inset_bounds[1], inset_bounds[3])
    plt.tight_layout()
    plt.savefig(figs_dir+run_name+'_'+demog+'_pct_map_inset.png', dpi = 600)
    plt.close("all")


#boxplots of ensemble demogs with enacted
#read in ensemble
polity = 'WA'
plans_path = './Outputs/'
plans_run_name = 'WA_49_False_False'
plans_time = '0'
plans_store_steps = [50000*i for i in range(1,11)]
demog = 'poc_cvap' #'superdit_poc_cvap' 'poc_cvap' 
run_type = 'neutral'
dists_file_path = './WA_dists_w_cvap/'
poc_rep_file = './WA_poc_reps.csv'
run_name = polity+'_'+demog+'_dists_'+run_type+'w_enacted'
polity_poc_cvaps = {'WA':.227,'OR':.161}
polity_poc_pops = {'WA':.275,'OR':.215}

dists_df = gpd.read_file(dists_file_path)
poc_reps = pd.read_csv(poc_rep_file)
dists_df = dists_df.merge(poc_reps, how = 'left',left_on = 'OBJECTID', right_on ='dist')
dists_df['POC_CVAP'] = 1-dists_df['WCVAP19']/dists_df['CVAP19']
poc_cvap_vals = [(list(dists_df['POC_CVAP'])[i],list(dists_df['poc_reps'])[i]/3) for i in range(len(dists_df))]
poc_cvap_vals.sort()

plan_dfs = []
for store_step in plans_store_steps:
    df = pd.read_csv(plans_path+'Outputs_'+plans_run_name+'_'+plans_time+'/'+demog+str(store_step)+'_'+plans_run_name+'.csv', header = None, names = range(len(dists_df)))
    plan_dfs.append(df)
plans = pd.concat(plan_dfs)

c = 'k'
plt.figure()
plt.boxplot(
    [np.array(plans[col]) for col in plans.columns],
    whis=[1, 99],
    showfliers=False,
    patch_artist=True,
    boxprops=dict(facecolor='gray', color=c),
    capprops=dict(color=c),
    whiskerprops=dict(color=c),
    flierprops=dict(color=c, markeredgecolor=c),
    medianprops=dict(color=c),
    zorder = 0,
    widths = .7, 
)
plt.plot([0.5, len(dists_df)], [0.5, 0.5], color='green', label='50%', zorder = 1)
plt.scatter(list(range(1,len(dists_df)+1)), [a for a,b in poc_cvap_vals], color='red', label='Enacted POC_CVAP %', alpha = .5,zorder = 2, s=100)
plt.scatter(list(range(1,len(dists_df)+1)), [b for a,b in poc_cvap_vals], color='blue', label='% POC Legislators', alpha = .5,zorder = 3, s=100)
plt.xlabel('Sorted Districts', fontsize = 16)
plt.ylabel('POC_CVAP %', fontsize = 16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.legend(fontsize = 16)
fig = plt.gcf()
fig.set_size_inches((20, 10), forward=False)
fig.savefig(figs_dir+run_name+'_demog_boxplot.png', dpi = 300)
plt.close('all')




#WA POC seats chart
polity = 'WA'
# fig_name = 'system_compare_all'
# alternatives = ["System 0 Senate","System 0 House","System 0 Total","System 1 Senate","System 1 House","System 1 Total","System 2 Senate","System 2 House","System 2 Total","System 3 Senate","System 3 House","System 3 Total", "System 4","System 5"]
# sizes = [49,98,147,48,96,144,33,99,132,49,98,147,150,150]
# pol1_rcv_seats = [(1,2),(2,4),(2,6),(8,11),(20,26),(28,36),(0,0),(15,23),(15,23),(10,14),(21,27),(32,41),(5,9),(31,40)]
# pol2_rcv_seats = [(2,6),(4,11),(6,17),(13,18),(28,36),(41,54),(1,4),(27,37),(28,39),(14,18),(29,36),(44,54),(11,20),(42,57)]
# pol3_rcv_seats = [(0,2),(1,5),(1,7),(12,16),(26,33),(38,49),(0,1),(24,33),(24,35),(13,17),(27,33),(40,50),(4,11),(37,52)]
# pol4_rcv_seats = [(3,24),(5,47),(8,71),(15,22),(36,42),(51,63),(2,16),(30,45),(32,60),(18,21),(36,41),(54,62),(10,72),(53,67)]
# current_seats = [None,None,None,None,None,None,8,20,28,None,None]
# scales = {"System 1 Senate":list(range(0,148,21)),"System 1 House":list(range(0,295,21)),"System 1 Total":list(range(0,442,21)),"System 2 Senate":list(range(0,34,3)),"System 2 House":list(range(0,100,9)),"System 2 Total":list(range(0,133,12)),"System 3 Senate":list(range(0,50,7)),"System 3 House":list(range(0,99,14)),"System 3 Total":list(range(0,148,21)), "System 4":list(range(0,151,10)),"System 5":list(range(0,201,10))}

fig_name = 'system_compare_pared'
alternatives = ["System 0 Senate","System 0 House", "System 1 Senate","System 1 House","System 2 Senate","System 2 House","System 3 Senate","System 3 House", "System 4","System 5"]
sizes = [49,98,48,96,33,99,49,98,150,150]
pol1_rcv_seats = [(1,2),(2,4),(8,11),(20,26),(0,0),(15,23),(10,14),(21,27),(5,9),(31,40)]
pol2_rcv_seats = [(2,6),(4,11),(13,18),(28,36),(1,4),(27,37),(14,18),(29,36),(11,20),(42,57)]
pol3_rcv_seats = [(0,2),(1,5),(12,16),(26,33),(0,1),(24,33),(13,17),(27,33),(4,11),(37,52)]
pol4_rcv_seats = [(3,24),(5,47),(15,22),(36,42),(2,16),(30,45),(18,21),(36,41),(10,72),(53,67)]
current_seats = [8,20,None,None,None,None,8,20,None,None]
scales = {"System 0 Senate":list(range(0,50,7)),"System 0 House":list(range(0,99,14)),"System 0 Total":list(range(0,148,21)),"System 1 Senate":list(range(0,49,6)),"System 1 House":list(range(0,97,12)),"System 1 Total":list(range(0,145,12)),"System 2 Senate":list(range(0,34,3)),"System 2 House":list(range(0,100,9)),"System 2 Total":list(range(0,133,12)),"System 3 Senate":list(range(0,50,7)),"System 3 House":list(range(0,99,14)),"System 3 Total":list(range(0,148,21)), "System 4":list(range(0,151,10)),"System 5":list(range(0,151,10))}

POC_CVAP =  [.227]*len(sizes)
POC_POP =  [.275]*len(sizes)
alternatives_inds = np.array(range(len(alternatives)))

x_buffer = .1
x_pol_buffer = .03
fig, axs = plt.subplots(1, len(alternatives),figsize=(4+2*len(alternatives),5))
matplotlib.rcParams.update({'font.size': 24})
for i in range(len(alternatives)):
    y_buffer = 10/max(sizes)*sizes[i]
    axs[i].scatter([x_buffer], [POC_CVAP[i]*sizes[i]], s = 300, c = 'lightblue', zorder = 4, edgecolors='black',label = 'Proportional (POC CVAP)', marker = '<')
    axs[i].scatter([x_buffer], [POC_POP[i]*sizes[i]], s = 300, c = 'tab:blue', zorder = 3, edgecolors='black',label = 'Proportional (POC Population)', marker = '<', linewidths=1)
    axs[i].scatter([x_buffer], [current_seats[i]], s = 300, c = 'gray', zorder = 5, edgecolors='black',label = 'Current POC Seats (where applicable)', marker = '<', linewidths=1)
    axs[i].errorbar([0], [sizes[i]/2], [sizes[i]/2], capsize=15, elinewidth=2,markeredgewidth=2, zorder = 0, color = 'black', alpha = .5,ls='none')
    axs[i].plot([0,0], [pol1_rcv_seats[i][0]-.01,pol1_rcv_seats[i][1]+.01], solid_capstyle = 'round', lw = 8,  color = 'tab:green', alpha = .6, zorder = 1, label = 'Expected RCV POC Seat Range: Polarization Category 1')
    axs[i].plot([0,0], [pol2_rcv_seats[i][0],pol2_rcv_seats[i][1]], solid_capstyle = 'round', lw = 8,  color = 'tab:red', alpha = .6, zorder = 1, label = 'Expected RCV POC Seat Range: Polarization Category 2')
    axs[i].plot([0-x_pol_buffer,0-x_pol_buffer], [pol3_rcv_seats[i][0],pol3_rcv_seats[i][1]], solid_capstyle = 'round', lw = 8,  color = 'orange', alpha = .6, zorder = 1, label = 'Expected RCV POC Seat Range: Polarization Category 3')
    axs[i].plot([0+x_pol_buffer,0+x_pol_buffer], [pol4_rcv_seats[i][0],pol4_rcv_seats[i][1]], solid_capstyle = 'round', lw = 8,  color = 'tab:purple', alpha = .6, zorder = 1, label = 'Expected RCV POC Seat Range: Polarization Category 4')
    axs[i].plot([0,0],[0,sizes[i]], linewidth = 1, color = 'black', zorder = 0)
    axs[i].set_xlabel(alternatives[i], fontsize = 12)
    axs[i].set_xticklabels([None])
    axs[i].locator_params(axis="y", integer=True, tight=True)
    axs[i].set_ylim(0-y_buffer,sizes[i]+y_buffer)
    axs[i].set_yticks(scales[alternatives[i]]) 
    axs[i].set_yticklabels(scales[alternatives[i]],fontsize=12) 
    axs[i].set_ylim(0-y_buffer,sizes[i]+y_buffer)
    axs[i].set_xlim(-.1,.15)
axs[0].set_ylabel('Estimated POC Seats',fontsize=14)
matplotlib.rcParams.update({'font.size': 14})
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
plt.tight_layout()
plt.savefig(figs_dir+polity+'_poc_seats_chart'+fig_name+'.png', dpi = 300)
