RUNNING RCV

    - See WA_RCV_run_scripts/README.md for instructions on how to set up and run a large set of RCV simulations across the parameter space
    - The outputs of those runs are stored in WA_RCV_outputs

GENERATING ENSEMBLES

    - WA_ensemble_generator.py is used to build neutral ensembles for statewide legislative districts out of block groups
        Input arguments for the script are: 
            1: number of steps/plans in ensemble
            2: population tolerance from ideal
            3: number of districts
            4: indicator boolean if districts should be subdivided for nesting
            5: indicator boolean if districts should be merged for nesting

        The arguments used for the voting systems in the report were:

            System 1: 500000 .05 49 'random' False False
            System 2: 500000 .05 33 'random' False False
            System 3: 500000 .05 7 'random' True False
            System 4: 500000 .05 150 './WA_seed_plans/WA_150_dists_pop_tol_0_04.json' False False
            System 5: 500000 .05 200 './WA_seed_plans/WA_200_dists_pop_tol_0_04.json' False False

    - WA_bg_w_cvap_data is the shapefile for Washington state that is used for the ensemble generation
    - WA_seed_plans are pre-generated starting maps for the systems with larger number of districts (recursive tree-part does not perform well here)

ANALYZING STATEWIDE RCV

    - WA_rcv_analysis.py is a script for estimating statewide performance of RCV from RCV simulation outputs (for each district) and an ensemble generated for the whole state
        - users can set num_draws to be the number of plans subsampled from the ensembles for the RCV analysis (we used 10,000 in the report)
        - voting_systems_dists is a dictionary mapping (voting-system, chamber) pairs to number of districts in that voting system and chamber
        - voting_systems_seats is a dictionary mapping (voting-system, chamber) pairs to number of seats in each district for that voting system and chamber
        - nested_voting_systems is a list of the (voting-system, chamber) pairs that have are nested
        - plans_time_dict is a dictionary mapping (voting-system, chamber) pairs to the timestamp of the corresponding ensemble directory (used to distinguish between multiple ensemble runs with same voting system parameters)
        - plans_store_steps should be set to the data-storage intervals used for the ensembles (we used [50000*i for i in range(1,11)] for the report)
        - voting_systems_to_plot is the list of the subset of (voting-system, chamber) pairs that actually want to be run (helpful if you want to avoid rerunning ALL systems every time)

    - WA_article_figs.py is a script for generating maps and summary figures for the report