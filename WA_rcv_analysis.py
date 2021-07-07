import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
import seaborn as sns
from matplotlib import colors
from tqdm import tqdm


figs_dir = "./figs/"
os.makedirs(os.path.dirname(figs_dir), exist_ok=True)

data_dir = "./output_data/"
os.makedirs(os.path.dirname(data_dir), exist_ok=True)

num_draws = 10000

polity = 'WA'

voting_systems_dists = {('0','House'):49,('0','Senate'):49,('1','House'):16,('1','Senate'):16,('2','House'):33,('2','Senate'):33,('3','House'):14,('3','Senate'):7,('4','Unicameral'):150,('5','Unicameral'):30}
voting_systems_seats = {('0','House'):1,('0','Senate'):1,('1','House'):6,('1','Senate'):3,('2','House'):3,('2','Senate'):1,('3','House'):7,('3','Senate'):7,('4','Unicameral'):1,('5','Unicameral'):5}
nested_voting_systems = [('3','House'),('3','Senate')]
multi_dist_elecs = {('0','House'):2}
plans_time_dict = {('WA',7):'0',('WA',14):'0',('WA',16):'0',('WA',30):'0',('WA',33):'0',('WA',49):'0',('WA',150):'0'}
plans_store_steps = [50000*i for i in range(1,11)]
voting_systems_to_plot = [vs for vs in voting_systems_dists.keys()]
voting_systems_to_plot = [('0', 'House')]

for voting_system in voting_systems_to_plot:
    print('Voting System: ',voting_system)
    #ensemble variables
    plans_path = './Outputs/'
    num_dists = voting_systems_dists[voting_system]
    chamber = voting_system[1]
    plans_run_name = polity+'_'+str(num_dists if (voting_system not in nested_voting_systems or chamber == 'Senate') else int(num_dists/2))+('_False_False' if voting_system not in nested_voting_systems else '_True_False')
    plans_time = plans_time_dict[(polity,num_dists)]
    demog = 'subdit_poc_cvap' if voting_system in nested_voting_systems and chamber == 'House' else 'poc_cvap'
    seat_size = voting_systems_seats[voting_system]
    dist_elecs = multi_dist_elecs[voting_system] if voting_system in multi_dist_elecs.keys() else 1
    run_name = polity+'_'+voting_system[0]+'_'+chamber+'_'+demog+'_'+str(num_dists)+'_dists_'+str(seat_size)+'_seats'

    #current seat share
    polity_poc_cvaps = {'WA':.227}
    polity_poc_pops = {'WA':.275}
    current_poc_reps = {('WA','House'):.204,('WA','Senate'):.163,('WA','Unicameral'):.190}

    #rcv file variables
    max_demog_dict = {('WA',1):85, ('WA',2):65, ('WA',3):70,('WA',5):65, ('WA',6):70, ('WA',7):54}
    max_demog = max_demog_dict[(polity,seat_size)]
    prefix = polity+'_'+str(seat_size)+'_seats'
    seats_cands_dict = {1:[(1,1,2),(1,2,2)],2:[(2,1,2),(2,2,2)], 3:[(3,2,3),(3,3,3)],5:[(5,3,5),(5,5,5)],6:[(6,3,6),(6,6,6)],7:[(7,3,7),(7,7,7)],9:[(9,3,9),(9,9,9)]}
    path = './'+polity+'_RCV_outputs/'

    #read in ensemble
    plan_dfs = []
    for store_step in plans_store_steps:
        df = pd.read_csv(plans_path+'Outputs_'+plans_run_name+'_'+plans_time+'/'+demog+str(store_step)+'_'+plans_run_name+'.csv', header = None, names = range(num_dists))
        plan_dfs.append(df)

    plans = pd.concat(plan_dfs)
    plans = plans[~plans[0].isna()]
    pol_colors = ['tab:green','tab:red','orange','tab:purple']


    #read in RCV data
    demog = 'POC_CVAP'
    polarization_levels = ['high','high_alt','medium','low']
    pol_lev_dict = {'high_alt':(.9,.2),'high':(.95,.05),'medium':(.75,.2),'low':(.6,.4)}
    demog_range = [i/100 for i in range(0,max_demog)]
    seats_cands = seats_cands_dict[seat_size]
    model_list = ['PL','BT','AC','CS']
    scenario_list = ['A','B','C','D','E']


    df_list = []
    for pol_lev in polarization_levels:
        for seats in seats_cands:
            for demog_lev in demog_range:
                df = pd.read_csv(path+','.join([prefix,demog,pol_lev,str(demog_lev),str(pol_lev_dict[pol_lev][0]),str(pol_lev_dict[pol_lev][1])]+[str(i) for i in list(seats)])+'.csv', header = None, names = scenario_list)
                df['model'] = model_list
                df['demog'] = demog
                df['demog_lev'] = demog_lev
                df['seats'] = seats[0]
                df['POC_cands'] = seats[1]
                df['W_cands'] = seats[2]
                df['pol_level'] = pol_lev
                df['POC_for_POC'] = pol_lev_dict[pol_lev][0]
                df['W_for_POC'] = pol_lev_dict[pol_lev][1]
                df_list.append(df)

    df_rcv = pd.concat(df_list)
    df_rcv = df_rcv.melt(id_vars = ['model', 'demog', 'demog_lev', 'seats','POC_cands', 'W_cands', 'pol_level', 'POC_for_POC', 'W_for_POC'], var_name = 'scenario',value_name = 'est_POC_seats')



    # Draws for parameter combinations
    new_df_cols = []
    for model in model_list:
        new_df_cols.append(model+'_POC_seats_sum')
        for dist in range(num_dists):
            new_df_cols.append('d'+str(dist)+'_'+model+'_POC_seats')
        
    plans_sample = pd.DataFrame(0,index=np.arange(num_draws),columns = new_df_cols)
    for i in range(dist_elecs):
        plans_sample_i = plans.sample(n = num_draws, replace = True)
        plans_sample_i = plans_sample_i.round(decimals=2)
        for model in model_list:
            plans_sample_i[model+'_POC_seats_sum'] = 0
        for dist in tqdm(range(num_dists)):
            plans_sample_i['scenario_draw'] = np.random.choice(scenario_list, len(plans_sample_i))
            plans_sample_i['poc_cands_draw'] = np.random.choice([b for a,b,c in seats_cands_dict[seat_size]], len(plans_sample_i))
            plans_sample_i['pol_draw'] = np.random.choice([a for a,b in pol_lev_dict.values()], len(plans_sample_i))
            for model in model_list:
                plans_sample_i['d'+str(dist)+'_'+model+'_POC_seats'] = list(plans_sample_i.merge(df_rcv[df_rcv['model']==model][['scenario','POC_cands','POC_for_POC','demog_lev','est_POC_seats']],how = 'left', left_on = ['scenario_draw','poc_cands_draw','pol_draw',dist], right_on = ['scenario','POC_cands','POC_for_POC','demog_lev'])['est_POC_seats'])
                plans_sample_i[model+'_POC_seats_sum'] = plans_sample_i[model+'_POC_seats_sum'] + plans_sample_i['d'+str(dist)+'_'+model+'_POC_seats']
        
        for model in model_list:
            plans_sample[model+'_POC_seats_sum'] = plans_sample[model+'_POC_seats_sum'].add(list(plans_sample_i[model+'_POC_seats_sum']))
            for dist in range(num_dists):
                plans_sample['d'+str(dist)+'_'+model+'_POC_seats'] = plans_sample['d'+str(dist)+'_'+model+'_POC_seats'].add(list(plans_sample_i['d'+str(dist)+'_'+model+'_POC_seats']))


    # *********  FIGURES  *********

    #histogram of POC seats
    fig, axs = plt.subplots(len(model_list),1, figsize = (7,len(model_list)*2))
    x_min = .95* min([plans_sample[model_list[i]+'_POC_seats_sum'].min()/(seat_size*num_dists*dist_elecs) for i in range(len(model_list))]+[polity_poc_cvaps[polity],polity_poc_pops[polity],current_poc_reps[(polity,chamber)]])
    x_max = 1.05* max([plans_sample[model_list[i]+'_POC_seats_sum'].max()/(seat_size*num_dists*dist_elecs) for i in range(len(model_list))]+[polity_poc_cvaps[polity],polity_poc_pops[polity],current_poc_reps[(polity,chamber)]])
    for i in range(len(model_list)):
        axs[i].hist(plans_sample[model_list[i]+'_POC_seats_sum']/(seat_size*num_dists*dist_elecs), color = 'slategray', bins = [0.02*i for i in range (51)])
        axs[i].axvline(x = polity_poc_cvaps[polity],label = polity + ' POC CVAP' if i == 0 else '', color = 'lightblue', lw = 4, alpha = .7)
        axs[i].axvline(x = polity_poc_pops[polity],label = polity + ' POC POP' if i == 0 else '', color = 'mediumblue', lw = 4, alpha = .4)
        axs[i].axvline(x = current_poc_reps[(polity,chamber)],label = polity + ' current seats' if i == 0 else '', color = 'gray', lw = 4, alpha = .5)
        axs[i].set_title('Model: '+model_list[i])
        axs[i].set_xlabel('POC Seats')
        axs[i].set_ylabel('Frequency')
        axs[i].set_xlim(x_min,x_max)
    fig.legend(loc=7)
    plt.tight_layout()
    fig.subplots_adjust(right=0.7)  
    plt.savefig(figs_dir+run_name+'_POC_seat_sum_hist.png')
    plt.close('all')

    #histogram of POC seats for specific district
    dist = int(round(num_dists/2))
    bin_num = 20
    fig, axs = plt.subplots(len(model_list),1, figsize = (7,len(model_list)*2))
    x_min = .9* min([plans_sample['d'+str(dist)+'_'+model_list[i]+'_POC_seats'].min() for i in range(len(model_list))])
    x_max = 1.1* max([plans_sample['d'+str(dist)+'_'+model_list[i]+'_POC_seats'].max() for i in range(len(model_list))])
    for i in range(len(model_list)):
        axs[i].hist(plans_sample['d'+str(dist)+'_'+model_list[i]+'_POC_seats'], color = 'slategray', bins = [x_min+i*(x_max-x_min)/bin_num for i in range(bin_num+1)],align = "left")
        axs[i].set_title('Model: '+model_list[i])
        axs[i].set_xlabel('POC Seats')
        axs[i].set_ylabel('Frequency')
        axs[i].set_xlim(x_min,x_max)
    fig.legend(loc=7)
    plt.tight_layout()
    fig.subplots_adjust(right=0.7)  
    plt.savefig(figs_dir+run_name+'_POC_seat_d_'+str(dist)+'_hist.png')
    plt.close('all')

    #boxplots of POC seats
    c = 'k'
    fig, axs = plt.subplots(len(model_list),1, figsize = (10,len(model_list)*2))
    for i in range(len(model_list)):
        axs[i].boxplot(
        [np.array(plans_sample['d'+str(dist)+'_'+model_list[i]+'_POC_seats']) for dist in range(num_dists)],
        whis=[1, 99],
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor='None', color=c),
        capprops=dict(color=c),
        whiskerprops=dict(color=c),
        flierprops=dict(color=c, markeredgecolor=c),
        medianprops=dict(color=c),
        )
        axs[i].set_title('Model: '+model_list[i])
        axs[i].set_xlabel('Districts (sorted by POC CVAP)')
        axs[i].set_ylabel('POC Seats')
        axs[i].set_ylim(0,seat_size)
    plt.tight_layout()
    plt.savefig(figs_dir+run_name+'_POC_seat_boxplot.png')
    plt.close('all')

    #boxplots of ensemble demogs
    c = 'k'
    plt.figure()
    plt.boxplot(
        [np.array(plans[col]) for col in plans.columns],
        whis=[1, 99],
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor='None', color=c),
        capprops=dict(color=c),
        whiskerprops=dict(color=c),
        flierprops=dict(color=c, markeredgecolor=c),
        medianprops=dict(color=c),
    )
    plt.plot([0.5, num_dists], [0.5, 0.5], color='green', label='50%')
    plt.xlabel('Sorted Districts')
    plt.ylabel('POC_CVAP %')
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches((20, 10), forward=False)
    fig.savefig(figs_dir+run_name+'_demog_boxplot.png')
    plt.close('all')



    # # ***************************************************************
    # # *********  Breakdown by Polarization  *********

    # Draws for parameter combinations

    new_df_cols_by_pol = []
    for model in model_list:
        for pol_lev in polarization_levels:
            new_df_cols_by_pol.append(model+'_'+pol_lev+'_POC_seats_sum')
            for dist in range(num_dists):
                new_df_cols_by_pol.append('d'+str(dist)+'_'+model+'_'+pol_lev+'_POC_seats')

    plans_sample_by_pol = pd.DataFrame(0,index=np.arange(num_draws),columns = new_df_cols_by_pol)        
    
    for i in range(dist_elecs):
        plans_sample_by_pol_i = plans.sample(n = num_draws, replace = True)
        plans_sample_by_pol_i = plans_sample_by_pol_i.round(decimals=2)
        for model in model_list:
            for pol_lev in polarization_levels:
                plans_sample_by_pol_i[model+'_'+pol_lev+'_POC_seats_sum'] = 0
        for dist in tqdm(range(num_dists)):
            plans_sample_by_pol_i['scenario_draw'] = np.random.choice(scenario_list, len(plans_sample_by_pol_i))
            plans_sample_by_pol_i['poc_cands_draw'] = np.random.choice([b for a,b,c in seats_cands_dict[seat_size]], len(plans_sample_by_pol_i))
            for model in model_list:
                for pol_lev in polarization_levels:
                    plans_sample_by_pol_i['d'+str(dist)+'_'+model+'_'+pol_lev+'_POC_seats'] = list(plans_sample_by_pol_i.merge(df_rcv[(df_rcv['model']==model)&(df_rcv['pol_level']==pol_lev)][['scenario','POC_cands','demog_lev','est_POC_seats']],how = 'left', left_on = ['scenario_draw','poc_cands_draw',dist], right_on = ['scenario','POC_cands','demog_lev'])['est_POC_seats'])
                    plans_sample_by_pol_i[model+'_'+pol_lev+'_POC_seats_sum'] = plans_sample_by_pol_i[model+'_'+pol_lev+'_POC_seats_sum'] + plans_sample_by_pol_i['d'+str(dist)+'_'+model+'_'+pol_lev+'_POC_seats']

        for model in model_list:
            for pol_lev in polarization_levels:
                plans_sample_by_pol[model+'_'+pol_lev+'_POC_seats_sum'] = plans_sample_by_pol[model+'_'+pol_lev+'_POC_seats_sum'].add(list(plans_sample_by_pol_i[model+'_'+pol_lev+'_POC_seats_sum']))
                for dist in range(num_dists):
                    plans_sample_by_pol['d'+str(dist)+'_'+model+'_'+pol_lev+'_POC_seats'] = plans_sample_by_pol['d'+str(dist)+'_'+model+'_'+pol_lev+'_POC_seats'].add(list(plans_sample_by_pol_i['d'+str(dist)+'_'+model+'_'+pol_lev+'_POC_seats']))
                

    # Data
    plans_sample_by_pol['num_seats'] = seat_size*num_dists*dist_elecs
    col_list = ['num_seats']
    for i in range(len(model_list)):
        for j in range(len(polarization_levels)):
            col_list.append(model_list[i]+'_'+polarization_levels[j]+'_POC_seats_sum')
    plans_sample_by_pol[col_list].to_csv(data_dir+run_name+'_sums.csv')



    # *********  FIGURES  *********

    #histogram of POC seats by Polarization 
    fig, axs = plt.subplots(len(model_list),1, figsize = (7,len(model_list)*2))
    x_min = .95* min([plans_sample_by_pol[model_list[i]+'_'+polarization_levels[j]+'_POC_seats_sum'].min()/(seat_size*num_dists*dist_elecs) for i in range(len(model_list)) for j in range(len(polarization_levels))]+[polity_poc_cvaps[polity],polity_poc_pops[polity],current_poc_reps[(polity,chamber)]])
    x_max = 1.05* max([plans_sample_by_pol[model_list[i]+'_'+polarization_levels[j]+'_POC_seats_sum'].max()/(seat_size*num_dists*dist_elecs) for i in range(len(model_list)) for j in range(len(polarization_levels))]+[polity_poc_cvaps[polity],polity_poc_pops[polity],current_poc_reps[(polity,chamber)]])
    leg = []
    for i in range(len(model_list)):
        for j in range(len(polarization_levels)):
            sns.kdeplot(list(plans_sample_by_pol[model_list[i]+'_'+polarization_levels[j]+'_POC_seats_sum']/(seat_size*num_dists*dist_elecs)) +[plans_sample_by_pol[model_list[i]+'_'+polarization_levels[j]+'_POC_seats_sum'].mean()/(seat_size*num_dists*dist_elecs)+.00001],ax = axs[i],shade=True, legend =False, color = pol_colors[j], bw = .005, clip = (0,float('inf')))
            leg_marker = mlines.Line2D([], [],  markeredgecolor=pol_colors[j], markerfacecolor=colors.to_rgba(pol_colors[j])[:3]+tuple([.3]), marker='s', linestyle='None',markersize=10, label='Polarization Category '+str(j+1))
            if i == 0:
                leg.append(leg_marker)
        axs[i].axvline(x = polity_poc_cvaps[polity],label = polity + ' POC CVAP' if i == 0 else '', color = 'lightblue', lw = 4, alpha = .7)
        axs[i].axvline(x = polity_poc_pops[polity],label = polity + ' POC POP' if i == 0 else '', color = 'mediumblue', lw = 4, alpha = .4)
        axs[i].axvline(x = current_poc_reps[(polity,chamber)],label = polity + ' current seats' if i == 0 else '', color = 'gray', lw = 4, alpha = .5)
        axs[i].set_title('Model: '+model_list[i])
        axs[i].set_xlabel('POC Seats')
        axs[i].set_ylabel('Frequency')
        axs[i].set_xlim(x_min,x_max)
    l1 = mlines.Line2D([], [],label = polity + ' POC CVAP' , color = 'lightblue', lw = 4, alpha = .7)
    l2 = mlines.Line2D([], [],label = polity + ' POC POP' , color = 'mediumblue', lw = 4, alpha = .4)
    l3 = mlines.Line2D([], [],label = polity + ' current seats' , color = 'gray', lw = 4, alpha = .5)
    fig.legend(handles=leg+[l1,l2,l3], loc=7)
    plt.tight_layout()
    fig.subplots_adjust(right=0.65)  
    plt.savefig(figs_dir+run_name+'_POC_seat_sum_hist_by_pol_kde.png')
    plt.close('all')

    #histogram of POC seats by Polarization 
    fig, axs = plt.subplots(len(model_list),1, figsize = (7,len(model_list)*2))
    x_min = .95* min([plans_sample_by_pol[model_list[i]+'_'+polarization_levels[j]+'_POC_seats_sum'].min()/(seat_size*num_dists*dist_elecs) for i in range(len(model_list)) for j in range(len(polarization_levels))]+[polity_poc_cvaps[polity],polity_poc_pops[polity],current_poc_reps[(polity,chamber)]])
    x_max = 1.05* max([plans_sample_by_pol[model_list[i]+'_'+polarization_levels[j]+'_POC_seats_sum'].max()/(seat_size*num_dists*dist_elecs) for i in range(len(model_list)) for j in range(len(polarization_levels))]+[polity_poc_cvaps[polity],polity_poc_pops[polity],current_poc_reps[(polity,chamber)]])
    for i in range(len(model_list)):
        for j in range(len(polarization_levels)):
            kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, ec="k", bins = [0.02*i for i in range (51)])
            axs[i].hist(plans_sample_by_pol[model_list[i]+'_'+polarization_levels[j]+'_POC_seats_sum']/(seat_size*num_dists*dist_elecs), **kwargs, label = 'Polarization Category '+str(j+1) if i==0 else '', color = pol_colors[j])
        axs[i].axvline(x = polity_poc_cvaps[polity],label = polity + ' POC CVAP' if i==0 else '', color = 'lightblue', lw = 4, alpha = .7)
        axs[i].axvline(x = polity_poc_pops[polity],label = polity + ' POC POP' if i==0 else '', color = 'mediumblue', lw = 4, alpha = .4)
        axs[i].axvline(x = current_poc_reps[(polity,chamber)],label = polity + ' current POC seats' if i==0 else '', color = 'gray', lw = 4, alpha = .5)
        axs[i].set_title('Model: '+model_list[i])
        axs[i].set_xlabel('POC Seats')
        axs[i].set_ylabel('Frequency')
        axs[i].set_xlim(x_min,x_max)
    fig.legend(loc=7)
    plt.tight_layout()
    fig.subplots_adjust(right=0.65)  
    plt.savefig(figs_dir+run_name+'_POC_seat_sum_hist_by_pol.png')
    plt.close('all')

    #histogram of POC seats for specific district by Polarization 
    dist = int(round(num_dists/2))
    bin_num = 20
    fig, axs = plt.subplots(len(model_list),1, figsize = (7,len(model_list)*2))
    x_min = .9* min([plans_sample_by_pol['d'+str(dist)+'_'+model_list[i]+'_'+polarization_levels[j]+'_POC_seats'].min() for i in range(len(model_list)) for j in range(len(polarization_levels))])
    x_max = 1.1* max([plans_sample_by_pol['d'+str(dist)+'_'+model_list[i]+'_'+polarization_levels[j]+'_POC_seats'].max() for i in range(len(model_list)) for j in range(len(polarization_levels))])
    leg = []
    for i in range(len(model_list)):
        for j in range(len(polarization_levels)):
            sns.kdeplot(list(plans_sample_by_pol['d'+str(dist)+'_'+model_list[i]+'_'+polarization_levels[j]+'_POC_seats'])+[plans_sample_by_pol['d'+str(dist)+'_'+model_list[i]+'_'+polarization_levels[j]+'_POC_seats'].mean()+.000001],ax = axs[i],shade=True, color = pol_colors[j],legend =False, bw = .03*x_max, clip = (0,float('inf')))
            leg_marker = mlines.Line2D([], [],  markeredgecolor=pol_colors[j], markerfacecolor=colors.to_rgba(pol_colors[j])[:3]+tuple([.3]), marker='s', linestyle='None',markersize=10, label='Polarization Category '+str(j+1))
            if i == 0:
                leg.append(leg_marker)
        axs[i].set_title('Model: '+model_list[i])
        axs[i].set_xlabel('POC Seats')
        axs[i].set_ylabel('Frequency')
        axs[i].set_xlim(x_min,x_max)
    fig.legend(handles=leg,loc=7)
    plt.tight_layout()
    fig.subplots_adjust(right=0.65)  
    plt.savefig(figs_dir+run_name+'_POC_seat_d_'+str(dist)+'_hist_by_pol_kde.png')
    plt.close('all')


    #boxplots of POC seats by Polarization 
    color_list = ['royalblue','orange','green','tab:red','mediumpurple','mediumturquoise','tab:pink','limegreen','saddlebrown','darkblue']
    fig, axs = plt.subplots(len(model_list),1, figsize = (15,len(model_list)*2))
    for i in range(len(model_list)):
        for j in range(len(polarization_levels)):
            c=pol_colors[j]
            axs[i].boxplot(
                [np.array(plans_sample_by_pol['d'+str(dist)+'_'+model_list[i]+'_'+polarization_levels[j]+'_POC_seats']) for dist in range(num_dists)],
                positions = [d-.1*(len(polarization_levels)-1)+j*.2 for d in range(num_dists)],
                whis = (1,99), sym = 'grey', showfliers = False, widths = .15, 
                patch_artist = True,
                boxprops=dict(facecolor=c, color=c,linewidth=.5),
                capprops=dict(color='k'),
                whiskerprops=dict(color='k'),
                flierprops=dict(color=c, markeredgecolor=c),
                medianprops=dict(color='k'),
            )
            if i == 0:
                axs[i].plot([], c=pol_colors[j], label='Polarization Category '+str(j+1))
        axs[i].set_title('Model: '+model_list[i])
        axs[i].set_xlabel('Districts (sorted by POC CVAP)')
        axs[i].set_ylabel('POC Seats')
        axs[i].set_ylim(-0.1,seat_size+.1)
        axs[i].set_xticks(list(range(0,num_dists)))
        axs[i].set_xticklabels(list(range(1,num_dists+1)))
    fig.legend(loc=7)
    plt.tight_layout()
    fig.subplots_adjust(right=0.85)  
    plt.savefig(figs_dir+run_name+'_POC_seat_boxplot_by_pol.png', dpi = 300)
    plt.close('all')



    #histogram of POC seats summed over models by Polarization 
    fig, ax = plt.subplots(figsize = (7,2))
    x_max = 0
    leg = []
    for j in range(len(polarization_levels)):
        plans_sample_by_pol[polarization_levels[j]+'_POC_seats_sum'] = 0
        for i in range(len(model_list)):
            plans_sample_by_pol[polarization_levels[j]+'_POC_seats_sum'] = plans_sample_by_pol[polarization_levels[j]+'_POC_seats_sum']+ plans_sample_by_pol[model_list[i]+'_'+polarization_levels[j]+'_POC_seats_sum']/(seat_size*num_dists*dist_elecs)/len(model_list)
        sns.kdeplot(list(plans_sample_by_pol[polarization_levels[j]+'_POC_seats_sum'])+[plans_sample_by_pol[polarization_levels[j]+'_POC_seats_sum'].mean()+.00001],ax = ax,shade=True, legend =False, color = pol_colors[j], bw = 0.005, clip = (0,float('inf')))
        leg_marker = mlines.Line2D([], [],  markeredgecolor=pol_colors[j], markerfacecolor=colors.to_rgba(pol_colors[j])[:3]+tuple([.3]), marker='s', linestyle='None',markersize=10, label='Polarization Category '+str(j+1))
        leg.append(leg_marker)
        x_max = max(x_max,plans_sample_by_pol[polarization_levels[j]+'_POC_seats_sum'].max())
    l1 = ax.axvline(x = polity_poc_cvaps[polity],label = polity + ' POC CVAP', color = 'lightblue', lw = 4, alpha = .7)
    l2 = ax.axvline(x = polity_poc_pops[polity],label = polity + ' POC POP', color = 'mediumblue', lw = 4, alpha = .4)
    l3 = ax.axvline(x = current_poc_reps[(polity,chamber)],label = polity + ' current POC seats', color = 'gray', lw = 4, alpha = .5)
    ax.set_xlabel('Percent POC Seats')
    ax.set_ylabel('Frequency')
    ax.set_xlim(0,x_max*1.2)
    fig.legend(handles=leg+[l1,l2,l3],loc=7)
    plt.tight_layout()
    fig.subplots_adjust(right=0.65)  
    plt.savefig(figs_dir+run_name+'_POC_seat_sum_hist_by_pol_kde_model_sum.png')
    plt.close('all')



    #histogram of POC seats by polarization level (union over models)
    fig, ax = plt.subplots(figsize = (7,2))
    x_max = 0
    leg = []
    for j in range(len(polarization_levels)):
        concat_plans = []
        for i in range(len(model_list)):
            concat_plans = concat_plans + list(plans_sample_by_pol[model_list[i]+'_'+polarization_levels[j]+'_POC_seats_sum']/(seat_size*num_dists*dist_elecs))
        # sns.kdeplot(concat_plans+[sum(concat_plans)/len(concat_plans)+.00001],ax = ax, shade=True, legend =False, color = pol_colors[j],bw = .05)
        sns.kdeplot(concat_plans+[sum(concat_plans)/len(concat_plans)+.00001],ax = ax, shade=True, legend =False, color = pol_colors[j], bw = .005, clip = (0,float('inf')))
        leg_marker = mlines.Line2D([], [],  markeredgecolor=pol_colors[j], markerfacecolor=colors.to_rgba(pol_colors[j])[:3]+tuple([.3]), marker='s', linestyle='None',markersize=10, label='Polarization Category '+str(j+1))
        leg.append(leg_marker)
        x_max = max(x_max,max(concat_plans))
    l1 = ax.axvline(x = polity_poc_cvaps[polity],label = polity + ' POC CVAP', color = 'lightblue', lw = 4, alpha = .7)
    l2 = ax.axvline(x = polity_poc_pops[polity],label = polity + ' POC POP', color = 'mediumblue', lw = 4, alpha = .4)
    l3 = ax.axvline(x = current_poc_reps[(polity,chamber)],label = polity + ' current POC seats', color = 'gray', lw = 4, alpha = .5)
    ax.set_xlabel('Percent POC Seats')
    ax.set_ylabel('Frequency')
    ax.set_xlim(0,x_max*1.2)
    fig.legend(handles=leg+[l1,l2,l3],loc=7)
    plt.tight_layout()
    fig.subplots_adjust(right=0.65)  
    plt.savefig(figs_dir+run_name+'_POC_seat_sum_hist_by_pol_kde_model_union.png')
    plt.close('all')


    #histogram of POC seats by model (union over polarization levels)
    fig, ax = plt.subplots(figsize = (7,2))
    x_max = 0
    leg = []
    for j in range(len(model_list)):
        concat_plans = []
        for i in range(len(polarization_levels)):
            concat_plans = concat_plans + list(plans_sample_by_pol[model_list[j]+'_'+polarization_levels[i]+'_POC_seats_sum']/(seat_size*num_dists*dist_elecs))
        sns.kdeplot(concat_plans+[sum(concat_plans)/len(concat_plans)+.00001],ax = ax, shade=True, legend =False, color = pol_colors[j], bw = .005, clip = (0,float('inf')))
        leg_marker = mlines.Line2D([], [],  markeredgecolor=pol_colors[j], markerfacecolor=colors.to_rgba(pol_colors[j])[:3]+tuple([.3]), marker='s', linestyle='None',markersize=10, label=model_list[j] + ' Model')
        leg.append(leg_marker)
        x_max = max(x_max,max(concat_plans))
    l1 = ax.axvline(x = polity_poc_cvaps[polity],label = polity + ' POC CVAP', color = 'lightblue', lw = 4, alpha = .7)
    l2 = ax.axvline(x = polity_poc_pops[polity],label = polity + ' POC POP', color = 'mediumblue', lw = 4, alpha = .4)
    l3 = ax.axvline(x = current_poc_reps[(polity,chamber)],label = polity + ' current POC seats', color = 'gray', lw = 4, alpha = .5)
    ax.set_xlabel('Percent POC Seats')
    ax.set_ylabel('Frequency')
    ax.set_xlim(0,x_max*1.2)
    fig.legend(handles=leg+[l1,l2,l3],loc=7)
    plt.tight_layout()
    fig.subplots_adjust(right=0.65)  
    plt.savefig(figs_dir+run_name+'_POC_seat_sum_hist_by_model_kde_pol_union.png')
    plt.close('all')



