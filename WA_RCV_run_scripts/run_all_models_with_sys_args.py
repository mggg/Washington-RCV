'''
Predict outcomes for minority groups under ranked choice voting
using four different models of voter behavior.

Enter basic input parameters under Global variables, then run the
code in order to simulate elections and output expected number of poc
candidates elected under each model and model choice.
'''


import numpy as np
from itertools import product, permutations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random, sys
import compute_winners as cw
from vote_transfers import cincinnati_transfer
from model_details import Cambridge_ballot_type, BABABA, luce_dirichlet, bradley_terry_dirichlet

params = sys.argv[1].split(',')

### Global variables
poc_share = float(params[3])
poc_support_for_poc_candidates = float(params[4])-1e-6
poc_support_for_white_candidates = 1.0-poc_support_for_poc_candidates+1e-6
white_support_for_poc_candidates = float(params[5])+1e-6
white_support_for_white_candidates = 1.0-white_support_for_poc_candidates-1e-6
num_ballots = 1000
num_simulations = 100
seats_open = int(params[6])
num_poc_candidates = int(params[7])
num_white_candidates = seats_open
run_at_large_as_well = False
max_ballot_length = None

#print(sys.argv[1])

### Luce model (Dirichlet variation)
concentrations = [0.5]*4 #>>1 means very similar supports, <<1 means most support goes to one or two candidates
#list goes [poc_for_poc, poc_for_white, white_for_poc, white_for_white]
concentration_list = [[0.5]*4, [2,0.5,0.5,0.5], [2,2,2,2], [0.5,0.5,2,2], [1.0]*4]


#simulate
poc_elected_luce_dirichlet = []
poc_elected_luce_dirichlet_atlarge = []
for i, concentrations in enumerate(concentration_list):
  poc_elected_rcv, poc_elected_atlarge = luce_dirichlet(
      poc_share = poc_share,
      poc_support_for_poc_candidates = poc_support_for_poc_candidates,
      poc_support_for_white_candidates = poc_support_for_white_candidates,
      white_support_for_white_candidates = white_support_for_white_candidates,
      white_support_for_poc_candidates = white_support_for_poc_candidates,
      num_ballots = num_ballots,
      num_simulations = num_simulations,
      seats_open = seats_open,
      num_poc_candidates = num_poc_candidates,
      num_white_candidates = num_white_candidates,
      concentrations = concentrations,
      max_ballot_length = max_ballot_length
  )
  poc_elected_luce_dirichlet.append(poc_elected_rcv)
  poc_elected_luce_dirichlet_atlarge.append(poc_elected_atlarge)



#print("\n Plackett-Luce Dirichlet predictions in order:")
if True:
    for i, c in enumerate(concentration_list[:-1]):
      print("{:.1f},".format(np.mean(poc_elected_luce_dirichlet[i]), np.mean(poc_elected_luce_dirichlet_atlarge[i])), end=" ")
    print("{:.1f}".format(np.mean(poc_elected_luce_dirichlet[-1]), np.mean(poc_elected_luce_dirichlet_atlarge[-1])))
### Bradley-Terry (Dirichlet variation)
#list goes [poc_for_poc, poc_for_white, white_for_poc, white_for_white]
concentration_list = [[0.5]*4, [2,0.5,0.5,0.5], [2,2,2,2], [0.5,0.5,2,2], [1.0]*4]


#simulate
poc_elected_bradley_terry_dirichlet = []
poc_elected_bradley_terry_dirichlet_atlarge = []

for i, concentrations in enumerate(concentration_list):
  poc_elected_rcv, poc_elected_atlarge = bradley_terry_dirichlet(
      poc_share = poc_share,
      poc_support_for_poc_candidates = poc_support_for_poc_candidates,
      poc_support_for_white_candidates = poc_support_for_white_candidates,
      white_support_for_white_candidates = white_support_for_white_candidates,
      white_support_for_poc_candidates = white_support_for_poc_candidates,
      num_ballots = num_ballots,
      num_simulations = num_simulations,
      seats_open = seats_open,
      num_poc_candidates = num_poc_candidates,
      num_white_candidates = num_white_candidates,
      concentrations = concentrations,
      max_ballot_length = max_ballot_length
  )
  poc_elected_bradley_terry_dirichlet.append(poc_elected_rcv)
  poc_elected_bradley_terry_dirichlet_atlarge.append(poc_elected_atlarge)

#print("\n Bradley-Terry Dirichlet predictions in order:")
if True:
    for i, c in enumerate(concentration_list[:-1]):
      print("{:.1f},".format(np.mean(poc_elected_bradley_terry_dirichlet[i]), np.mean(poc_elected_bradley_terry_dirichlet_atlarge[i])), end=" ")
    print("{:.1f}".format(np.mean(poc_elected_bradley_terry_dirichlet[-1]), np.mean(poc_elected_bradley_terry_dirichlet_atlarge[-1])))



### Alternating crossover model

#simulate
poc_elected_bababa,  poc_elected_bababa_atlarge = BABABA(
    poc_share = poc_share,
    poc_support_for_poc_candidates = poc_support_for_poc_candidates,
    poc_support_for_white_candidates = poc_support_for_white_candidates,
    white_support_for_white_candidates = white_support_for_white_candidates,
    white_support_for_poc_candidates = white_support_for_poc_candidates,
    num_ballots = num_ballots,
    num_simulations = num_simulations,
    seats_open = seats_open,
    num_poc_candidates = num_poc_candidates,
    num_white_candidates = num_white_candidates,
    scenarios_to_run = ['A', 'B', 'C', 'D'],
    verbose=False,
    max_ballot_length = max_ballot_length
)

#print("\n Alternating crossover predictions in order:")
if True:
    for i, c in enumerate(['A', 'B', 'C', 'D']):
      print("{:.1f},".format(np.mean(poc_elected_bababa[c]), np.mean(poc_elected_bababa_atlarge[c])), end=" ")
    print("{:.1f}".format(
      np.mean([np.mean(poc_elected_bababa[c]) for c in ['A', 'B', 'C', 'D']]),
      np.mean([np.mean(poc_elected_bababa_atlarge[c]) for c in ['A', 'B', 'C', 'D']])
    ))




### Cambridge ballot types

#simulate
poc_elected_Cambridge, poc_elected_Cambridge_atlarge = Cambridge_ballot_type(
    poc_share = poc_share,
    poc_support_for_poc_candidates = poc_support_for_poc_candidates,
    poc_support_for_white_candidates = poc_support_for_white_candidates,
    white_support_for_white_candidates = white_support_for_white_candidates,
    white_support_for_poc_candidates = white_support_for_poc_candidates,
    num_ballots = num_ballots,
    num_simulations = num_simulations,
    seats_open = seats_open,
    num_poc_candidates = num_poc_candidates,
    num_white_candidates = num_white_candidates,
    scenarios_to_run = ['A', 'B', 'C', 'D'],
    max_ballot_length = max_ballot_length
)

#print("\n Cambridge sampler predictions in order:")
if True:
    for i, c in enumerate(['A', 'B', 'C', 'D']):
      print("{:.1f},".format(np.mean(poc_elected_Cambridge[c]), np.mean(poc_elected_Cambridge_atlarge[c])), end=" ")
    print("{:.1f}".format(
      np.mean([np.mean(poc_elected_Cambridge[c]) for c in ['A', 'B', 'C', 'D']]),
      np.mean([np.mean(poc_elected_Cambridge_atlarge[c]) for c in ['A', 'B', 'C', 'D']])
    ))
