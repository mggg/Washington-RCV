# How to run multiple RCV analyses


## Step 1:
Open the Jupyter notebook `print arg list.ipynb` and use it to print the list of comma-separated parameter lists separated by spaces.

## Step 2:
Copy and paste that list into a shell script like `WA_1_seat_run.sh`. 	

## Step 3:
Go to the cluster, start an interactive session with `srun -p interactive --bash` and run `bash WA_1_seat_run.sh` except with your shell script instead of the example. Make sure all the other files in this folder are in the same folder as the script.

## Step 4: 
Collect the output as a bunch of csv files named after the parameter string.