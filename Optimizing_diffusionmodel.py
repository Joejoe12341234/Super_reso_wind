import prerequisites  # Load prerequisites
from prerequisites import *

# Set up device configuration for GPU 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths for datasets
train_high = 'path_to_train_high_res.npy'
test_high = 'path_to_test_high_res.npy'
eval_high = 'path_to_eval_high_res.npy'

# Function to load dataset from a .npy file
def load_dataset(file_name):
    """Loads dataset from a .npy file."""
    return np.load(file_name)

# Load high-resolution datasets
train_high_res_ds = load_dataset(train_high)
eval_high_res_ds = load_dataset(eval_high)
test_high_res_ds = load_dataset(test_high)

# Define dummy low-resolution datasets for demonstration
train_low_res_ds = np.random.rand(len(train_high_res_ds), 1, 16, 16)
eval_low_res_ds = np.random.rand(len(eval_high_res_ds), 1, 16, 16)
test_low_res_ds = np.random.rand(len(test_high_res_ds), 1, 16, 16)

# Create TensorDatasets
train_dataset = TensorDataset(torch.Tensor(train_low_res_ds), torch.Tensor(train_high_res_ds))
eval_dataset = TensorDataset(torch.Tensor(eval_low_res_ds), torch.Tensor(eval_high_res_ds))
test_dataset = TensorDataset(torch.Tensor(test_low_res_ds), torch.Tensor(test_high_res_ds))

def subsample_dataset(dataset, fraction=0.2):
    """Subsamples the given dataset to the specified fraction."""
    subset_size = int(len(dataset) * fraction)
    indices = np.random.choice(len(dataset), size=subset_size, replace=False)
    subset = torch.utils.data.Subset(dataset, indices)
    return subset

# Subsample the training dataset
train_dataset_subsampled = subsample_dataset(train_dataset, fraction=0.2)

def objective(trial):
    """Objective function for Optuna optimization."""
    start_time = time.time()

    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-6, 1e-4)
    timesteps = trial.suggest_int("timesteps", 8000, 9000, step=500)
    num_epochs = trial.suggest_int("num_epochs", 500, 700, step=40)
    batch_size = trial.suggest_int("batchsize", 20, 50, step=5)

    print(f"Suggested params - Epochs: {num_epochs}, Timesteps: {timesteps}, Learning Rate: {lr}")

    # Data loaders
    train_loader = DataLoader(train_dataset_subsampled, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model setup, need to alter for Cond diffusion
    unet1 = NullUnet()
    unet2 = Unet(dim=128, dim_mults=(1, 2, 4, 8), layer_attns=(False, False, False, True), layer_cross_attns=(False, False, False, True))
    imagen = Imagen(condition_on_text=False, unets=[unet1, unet2], channels=1, image_sizes=(16, 160), timesteps=timesteps)
    trainer = ImagenTrainer(imagen, lr=lr, verbose=False).to(device)

    # Early stopping setup
    patience = 30
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(train_loader):
            print(f"Epoch: {epoch}, Batch: {i}")
            loss = trainer(images=targets.to(device), unet_number=2)
            trainer.update(unet_number=2)

    # Evaluation
    with torch.no_grad():
        predictions, actuals = [], []
        for inputs, targets in eval_loader:
            inputs = inputs.to(device)
            sampled_images = imagen.sample(batch_size=batch_size, start_at_unet_number=2, start_image_or_video=inputs, cond_scale=3.0)
            predictions.append(sampled_images.cpu())
            actuals.append(targets)

            if len(predictions) == 2:  # For demo purposes, limiting evaluation to 2 batches
                break

        predictions = torch.cat(predictions).flatten()
        actuals = torch.cat(actuals).flatten()
        val_loss = mse_loss(predictions, actuals).item()
        correlation, _ = pearsonr(predictions.numpy(), actuals.numpy())

        print(f"Epoch {epoch}, Evaluation Loss: {val_loss}, Correlation: {correlation}")

    elapsed_time = time.time() - start_time
    print(f"Trial completed in {elapsed_time:.2f} seconds.")

    trial.set_user_attr("num_epochs", num_epochs)
    trial.set_user_attr("timesteps2", timesteps)
    trial.set_user_attr("val_loss2", val_loss)
    trial.set_user_attr("correlation2", correlation)
    trial.set_user_attr("lr2", lr)
    trial.set_user_attr("batch_size2", batch_size)

    return val_loss

if __name__ == "__main__":
    n_trials = 1
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    # Save study results
    df = study.trials_dataframe()
    study_results_path = 'study_results.csv'
    df.to_csv(study_results_path, index=False)

    # Print optimization results
    print("Optimization complete.")
    print(f"Best trial: {study.best_trial.number} with loss {study.best_trial.value}")
    print("Best trial parameters:")
    for key, value in study.best_trial.params.items():
        print(f"{key}: {value}")

