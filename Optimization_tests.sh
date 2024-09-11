#!/bin/bash

# Define arrays of parameters
dropout_rates=(0 0.05 0.1 0.5 0.75)
l2_regs=(0 0.05 0.1 0.5 0.75)
weight_exps=(0 1 2 4 10)
epochs=(250 500 750 1000 1500 3400 4000 10000 50000)
learning_rates=(0.1 0.01 0.001 0.0001 0.00001 0.000001 0.000001)
criteria=('extreme' 'mse' 'quantile')
scalings=('Robust' 'StandardScalar' 'MinMax')
 
# Loop over each combination of parameters and submit a SLURM job
for dropout_rate in "${dropout_rates[@]}"; do
  for l2_reg in "${l2_regs[@]}"; do
    for weight_exp in "${weight_exps[@]}"; do
      for epoch in "${epochs[@]}"; do
        for learning_rate in "${learning_rates[@]}"; do
          for criterion in "${criteria[@]}"; do
            for scaling in "${scalings[@]}"; do
              # Create a SLURM script
              random_number=$RANDOM
              job_script="slurm-$dropout_rate-$l2_reg-$weight_exp-$epoch-$learning_rate-$criterion-$scaling-$random_number.sh"
              cat > $job_script <<EOL
#!/bin/bash
#SBATCH --job-name=mnist-${dropout_rate}-${l2_reg}-${weight_exp}-${epoch}-${learning_rate}-${criterion}-${scaling}-$random_number
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=10:00:00
#SBATCH --mail-user= XXX

module load anaconda3/2023.9 
conda activate base
python first_model_test_optimizetests.py --dropout_rate $dropout_rate --l2_reg $l2_reg --weight_exp $weight_exp --epochs $epoch --learning_rate $learning_rate --criterion $criterion --scaling $scaling
EOL
              # Submit job
              sbatch $job_script
              rm $job_script
            done
          done
        done
      done
    done
  done
done
