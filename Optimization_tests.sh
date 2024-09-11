#!/bin/bash

# Define arrays of parameters
dropout_rates=(0 0.1)
l2_regs=(0 0.05)
weight_exps=(1 2 4)
epochs=(400 600 1000 1500 3400 4000)
learning_rates=(0.0001 0.00001)
criteria=('extreme' 'mse' 'quantile')
scalings=('Robust' 'MinMax')
# results_dropout_0.1_l2_0.0_weight_2_criterion_quantile_lr_0.0001_epoch_3400_scaler_RobustScaler()

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
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=10:00:00
#SBATCH --mail-user=jl115@princeton.edu

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
