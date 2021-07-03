import random
a = random.randint(0,10000000000)
import networkx as nx
from gerrychain.random import random
random.seed(a)
import csv
import os
from functools import partial
import json
import random
import numpy as np
import sys

import geopandas as gpd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import time

from gerrychain import (
    Election,
    Graph,
    MarkovChain,
    Partition,
    accept,
    constraints,
    updaters,
)
from gerrychain.metrics import efficiency_gap, mean_median
from gerrychain.proposals import recom
from gerrychain.updaters import cut_edges, Tally
from gerrychain.tree import PopulatedGraph, recursive_tree_part, predecessors, bipartition_tree, random_spanning_tree, find_balanced_edge_cuts_memoization
from networkx.algorithms import tree
from collections import deque, namedtuple
from WA_helper_functions import *




place_name = 'WA'
tot_pop = 'TOTPOP'
geo_id = 'GEOID10'
plot_path = './WA_bg_w_cvap_data/'  #for shapefile

#input and run parameters
start_time_total = time.time()
total_steps = int(sys.argv[1])
pop_tol = float(sys.argv[2])
num_districts = int(sys.argv[3]) 
start_map = sys.argv[4] #'random' or 'enacted' or path to seed plan
cleave = eval(sys.argv[5])  #True, False  (To form nested districts by cleaving gerrychain districts)
merge = eval(sys.argv[6])  #True, False (To form nested districts by merging gerrychain districts)

store_interval = int(total_steps/10)
store_chain = True
run_name = place_name+'_'+str(num_districts)+ '_' + str(cleave) + '_' + str(merge)


outputs_dir = "./Outputs/"
os.makedirs(os.path.dirname(outputs_dir), exist_ok=True)

newdir = outputs_dir+"Outputs_"+run_name+"_"+str(round(time.time()))+"/"
os.makedirs(os.path.dirname(newdir + "init.txt"), exist_ok=True)
with open(newdir + "init.txt", "w") as f:
    f.write("Created Folder")

gdf = gpd.read_file(plot_path)
gdf = gdf.rename(columns = {'CVAP19':'CVAP', 'AIANCVAP19':'AIANCVAP', 'ACVAP19':'ACVAP', 'BCVAP19':'BCVAP','NHPICVAP19':'NHPICVAP', 'WCVAP19':'WCVAP', 'HCVAP19':'HCVAP', 'OCVAP19':'OCVAP'})
gdf.geometry = gdf.buffer(0)
graph = Graph.from_geodataframe(gdf)
graph.add_data(gdf)



geoid_to_node_dict = {graph.nodes[v]['GEOID10']:v for v in graph.nodes}
#add edges to form connected dual graph
edges_to_add = [(530730110001, 530730104031),(530730109001, 530739400001),(530579501001, 530579508001),(530730109001, 530579501001),(530559605002, 530579501001),(530579404001, 530579501001),(530579407002, 530579501001),(530559603003, 530559605001),(530559605002, 530559604001),(530559603002, 530559605002),(530559601001, 530559605002),(530559601002, 530559605002),(530559601003, 530730109001),(530579403002, 530299701002),(530579408002, 530299701002),(530459611001, 530670119001),(530459611001, 530459611003),(530459611001, 530459612002),(530530726033, 530530726035),(530530729031, 530530726033),(530530726031, 530530726035),(530530724102, 530530723093),(530530724084, 530530724102),(530530724101, 530530724052),(530330277023, 530530725062),(530330277014, 530350927042),(530330277012, 530350927042),(530330277023, 530530603001),(530330244001, 530330239003),(530330062001, 530330053023),(530330057005, 530330032003),(530699501005, 530699501003),(530699501005, 530699501004),(530319504004, 530319503001),(530350910001, 530350926002),(530350910004, 530350926002),(530350910001, 530350918002),(530350907002, 530359401002)]
for edge in edges_to_add:
    graph.add_edge(geoid_to_node_dict[str(edge[0])],geoid_to_node_dict[str(edge[1])])
print('************* ADDED EDGES *************')

my_updaters = {
    "population": updaters.Tally(tot_pop, alias = "population"),
    "VAP": updaters.Tally("TOTVAP", alias = "VAP"),
    "NH_WVAP": updaters.Tally("NH_WVAP", alias = "NH_WVAP"),
    "CVAP": updaters.Tally("CVAP", alias = "CVAP"),
    "WCVAP": updaters.Tally("WCVAP", alias = "WCVAP"),
    "cut_edges": cut_edges,
}


#initial partition
total_population = gdf[tot_pop].sum()
ideal_population = total_population/num_districts
if start_map == 'enacted':
    assignment = enacted_assignment
elif start_map == 'random':
    random_assign = recursive_tree_part(graph, range(num_districts), ideal_population, tot_pop, pop_tol*.9, node_repeats = 5)
    assignment = random_assign
elif 'json' in start_map:
    with open(start_map) as f:
        data = json.load(f)
    assignment = {int(x):data[x] for x in data.keys()}
else:
    assert(False, 'invalid start map')


initial_partition = Partition(graph = graph, assignment = assignment, updaters = my_updaters)

proposal = partial(recom, pop_col=tot_pop, pop_target=ideal_population, epsilon= pop_tol, node_repeats=3)


chain = MarkovChain(
    proposal = proposal,
    constraints=[constraints.within_percent_of_ideal_population(initial_partition, pop_tol)],
    accept = accept.always_accept,
    initial_state = initial_partition,
    total_steps = total_steps
)

poc_cvap_list = []
cleave_poc_cvap_list = []
merge_poc_cvap_list = []
cut_edges_list = []
cleave_cut_edges_list = []
merge_cut_edges_list = []

step_Num = 0
new_assign = {}
subdist_poc_cvap_dict = {}
found_cleave = True

#run chain and collect data
for step in chain:
    cut_edges_list.append([len(step["cut_edges"])])
    poc_pcvap_pcts = [round(1-step["WCVAP"][i]/step["CVAP"][i],4) for i in step["CVAP"].keys()]
    poc_pcvap_pcts.sort()
    poc_cvap_list.append(poc_pcvap_pcts)
    
    #form nested districts by subdividing partition districts
    if cleave == True:
        new_parts = list(step.parts.keys()) if step.parent == None  or not found_cleave else [k for k in step.parts.keys() if step.parts[k] != step.parent.parts[k]]
        if step.parent!= None and found_cleave:
            assert(len(new_parts)<=2)
        alt_assign = {}
        alt_subdist_poc_cvap_dict = {}
        found_cleave = True
        for part in new_parts:
            sub_assign = recursive_tree_part(graph.subgraph(step.parts[part]), [2*part,2*part+1], ideal_population/2, tot_pop, pop_tol, node_repeats = 3, method = capped_tries_bipartition_tree)
            if len(set(sub_assign.values())) >1:
                alt_subdist_poc_cvap_dict.update({i:round(1-sum([graph.nodes[v]['WCVAP'] for v in sub_assign.keys() if sub_assign[v] == i])/sum([graph.nodes[v]['CVAP'] for v in sub_assign.keys() if sub_assign[v] == i]),4) for i in [2*part,2*part+1]})
                alt_assign.update(sub_assign)
            else:
                found_cleave = False
        if found_cleave:
            new_assign.update(alt_assign)
            subdist_poc_cvap_dict.update(alt_subdist_poc_cvap_dict)
            new_part = Partition(graph = graph, assignment = new_assign, updaters = my_updaters)
            assert len(new_part)==2*num_districts
            subdist_poc_cvap = list(subdist_poc_cvap_dict.values())
            subdist_poc_cvap.sort()
            cleave_poc_cvap_list.append(subdist_poc_cvap)
            cleave_cut_edges_list.append([len(new_part['cut_edges'])])
        else:
            cleave_poc_cvap_list.append(['NA']*(2*len(poc_pcvap_pcts)))
            cleave_cut_edges_list.append(['NA'])
    
    #form nested districts by merging partition districts
    if merge == True:
        superdist_poc_cvap = []
        g2 = nx.Graph()
        g2.add_edges_from(set([(step.assignment[e[0]],step.assignment[e[1]]) for e in step.cut_edges]))
        for e in g2.edges():
            g2.edges[e]['weight'] = 1
        match = nx.max_weight_matching(g2)
        remove_cut_edges = 0
        if len(match) == num_districts/2:
            for pair in match:
                superdist_poc_cvap.append(round(1-(step['WCVAP'][pair[0]]+step['WCVAP'][pair[1]])/(step['CVAP'][pair[0]]+step['CVAP'][pair[1]]),4))
                remove_cut_edges += len([e for e in step["cut_edges"] if {step.assignment[e[0]],step.assignment[e[1]]} == {pair[0],pair[1]}])
                new_assign.update({v:pair[0] for v in graph.nodes() if step.assignment[v] in pair})
            superdist_poc_cvap.sort()
            merge_poc_cvap_list.append(superdist_poc_cvap)
            merge_cut_edges_list.append([len(step["cut_edges"])-remove_cut_edges])
            new_part = Partition(graph = graph, assignment = new_assign, updaters = my_updaters)
            assert((len(step["cut_edges"])-remove_cut_edges) == len(new_part["cut_edges"]))
        else:
            merge_poc_cvap_list.append(['NA']*int(num_districts/2))
            merge_cut_edges_list.append(['NA'])

            
    step_Num += 1
    if step_Num % store_interval == 0:
        print(step_Num, round(time.time()-start_time_total))

        if store_chain:
            with open(newdir + "poc_cvap" + str(step_Num) + "_" + run_name+".csv", "w") as tf1:
                writer = csv.writer(tf1, lineterminator="\n")
                writer.writerows(poc_cvap_list)
            poc_cvap_list = []
            with open(newdir + "cut_edges" + str(step_Num) + "_" + run_name+".csv", "w") as tf1:
                writer = csv.writer(tf1, lineterminator="\n")
                writer.writerows(cut_edges_list)
            cut_edges_list = []
            if cleave:
                with open(newdir + "subdit_poc_cvap" + str(step_Num) + "_" + run_name+".csv", "w") as tf1:
                    writer = csv.writer(tf1, lineterminator="\n")
                    writer.writerows(cleave_poc_cvap_list)
                cleave_poc_cvap_list = []
                with open(newdir + "subdit_cut_edges" + str(step_Num) + "_" + run_name+".csv", "w") as tf1:
                    writer = csv.writer(tf1, lineterminator="\n")
                    writer.writerows(cleave_cut_edges_list)
                cleave_cut_edges_list = []
            if merge:
                with open(newdir + "superdit_poc_cvap" + str(step_Num) + "_" + run_name+".csv", "w") as tf1:
                    writer = csv.writer(tf1, lineterminator="\n")
                    writer.writerows(merge_poc_cvap_list)
                merge_poc_cvap_list = []
                with open(newdir + "superdit_cut_edges" + str(step_Num) + "_" + run_name+".csv", "w") as tf1:
                    writer = csv.writer(tf1, lineterminator="\n")
                    writer.writerows(merge_cut_edges_list)
                merge_cut_edges_list = []

