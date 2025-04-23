import torch
from torch.utils.data import DataLoader

from dataset_modules.dataset_tme import TMEDataSet
from models.models_tme import TumorMicroEnvironmentModel # Assuming dataset_tme contains a class named CustomDataset
import torch.nn as nn
import torch.optim as optim
import pandas as pd

class Config:
        def __init__(self):
            self.in_features_dim = 1024
            self.latent_dim = 1024
            self.nlp_dim = 1024
            self.d_model = 1024
            self.nhead = 2
            self.num_layers = 1
            self.dim_feedforward = 1024
            self.max_seq_len = 1024
            self.dropout = 0.4
            self.num_classes = 1
            self.num_pathology_patches = 5
            self.projection_type = "linear"  # or "nonlinear"

            # Dictionary dimensions for embeddings
            self.disease_type_dict_dim = 5
            self.primary_site_dict_dim = 2
            self.age_at_index_dict_dim = 1
            self.race_dict_dim = 7
            self.gender_dict_dim = 3
            self.age_at_diagnosis_dict_dim = 1
            self.days_to_birth_dict_dim = 1
            self.pathologic_stage_dict_dim = 13
            self.staging_dict_dim = 7
            self.tumor_class_dict_dim = 8
            self.primary_disease_dict_dim = 5
            self.primary_diagnosis_dict_dim = 45
            self.site_biopsy_dict_dim = 4
            self.specific_site_dict_dim = 11
            self.tissue_organ_origin_dict_dim = 60
            self.theraputic_agents_dict_dim = 46
            self.treatment_or_therapy_dict_dim = 5
            self.treatment_type_dict_dim = 22
            self.pathology_patch_emb_dim = 1024

            self.learning_rate = 0.001
            self.batch_size = 2
            self.num_epochs = 10
            self.random_seed = 42

def negative_log_partial_likelihood(survival, risk, debug=False):
    """Return the negative log-partial likelihood of the prediction
    y_true contains the survival time
    risk is the risk output from the neural network
    censor is the vector of inputs that are censored
    censor data: 1 - dead, 0 - censor
    regularization is the regularization constant (not used currently in model)

    Uses the torch backend to perform calculations

    Sorts the surv_time by sorted reverse time
    https://github.com/jaredleekatzman/DeepSurv/blob/master/deepsurv/deep_surv.py
    """

    # calculate negative log likelihood from estimated risk\
    # print(torch.stack([survival[:, 0], risk]))
    _, idx = torch.sort(survival[:, 0], descending=True)
    censor = survival[idx, 1]
    risk = risk[idx]
    epsilon = 0.00001
    max_value = 10
    alpha = 0.1
    risk = torch.reshape(risk, [-1])  # flatten
    shift = torch.max(risk)
    risk = risk - shift
    # hazard_ratio = torch.exp(risk)

    # cumsum on sorted surv time accounts for concordance
    # log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0) + epsilon)
    log_risk = torch.logcumsumexp(risk, dim=0)
    log_risk = torch.reshape(log_risk, [-1])
    uncensored_likelihood = risk - log_risk

    # apply censor mask: 1 - dead, 0 - censor
    censored_likelihood = uncensored_likelihood * censor
    num_observed_events = torch.sum(censor)
    if num_observed_events == 0:
        neg_likelihood = torch.tensor(0.0, device=risk.device, requires_grad=True)
    else:
        neg_likelihood = - torch.sum(censored_likelihood) / (num_observed_events)
    return neg_likelihood

def c_index(predicted_risk, survival):
    if survival is None:
        return 0
    # calculate the concordance index
    ci = np.nan  # just to know that concordance index cannot be estimated
    # print(r2python.cbind(np.reshape(predicted_risk, (-1, 1)), survival))

    try:
        na_inx = ~(np.isnan(survival[:, 0]) | np.isnan(survival[:, 1]) | np.isnan(predicted_risk))
        predicted_risk, survival = predicted_risk[na_inx], survival[na_inx]
        if len(predicted_risk) > 0 and sum(survival[:, 1] == 1) > 2:
            survival_time, censor = survival[:, 0], survival[:, 1]
            epsilon = 0.001
            partial_hazard = np.exp(-(predicted_risk + epsilon))
            censor = censor.astype(int)
            ci = concordance_index(survival_time, partial_hazard, censor)

    except:
        ci = np.nan

    return ci

def split_data(Input_file, target_file, random_seed=42):
    # Read the CSV file
    data = pd.read_csv(Input_file)
    target = pd.read_csv(target_file)

    # Merge the dataframes on the 'case_id' column
    merged_data = pd.merge(data, target, on='submitter_id')

    # Shuffle the data
    shuffled_data = merged_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Split the data into train, validation, and test sets
    train_size = int(0.8 * len(shuffled_data))
    val_size = int(0.1 * len(shuffled_data))
    test_size = len(shuffled_data) - train_size - val_size

    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size:train_size + val_size]
    test_data = shuffled_data[train_size + val_size:]

    # split the data into input and target
    train_inp_data = train_data.drop(columns=['survival_2','days_to_follow_up'])
    train_targ_data = train_data[['submitter_id','survival_2','days_to_follow_up']]

    val_inp_data = val_data.drop(columns=['survival_2','days_to_follow_up'])
    val_targ_data = val_data[['submitter_id','survival_2','days_to_follow_up']]

    test_inp_data = test_data.drop(columns=['survival_2','days_to_follow_up'])
    test_targ_data = test_data[['submitter_id','survival_2','days_to_follow_up']]


    # Save the split data to CSV files
    train_inp_data.to_csv('train_input.csv', index=False)
    train_targ_data.to_csv('train_target.csv', index=False)
    val_inp_data.to_csv('val_input.csv', index=False)
    val_targ_data.to_csv('val_target.csv', index=False)
    test_inp_data.to_csv('test_input.csv', index=False)
    test_targ_data.to_csv('test_target.csv', index=False)
    train_files = {
        'input': 'train_input.csv',
        'target': 'train_target.csv'
    }
    val_files = {
        'input': 'val_input.csv',
        'target': 'val_target.csv'
    }
    test_files = {
        'input': 'test_input.csv',
        'target': 'test_target.csv'
    }

    return train_files, val_files, test_files

def train_wrapper_supervised(Input_file,target_file, pathological_file_path, pathological_file_extension=".h5"):
    config = Config()
    lr = config.learning_rate
    batch_size = config.batch_size
    num_epochs = config.num_epochs
    random_seed = config.random_seed
    train_files, val_files, test_files = split_data(Input_file, target_file, random_seed)
    
    # Initialize dataset and dataloader
    train_dataset = TMEDataSet(train_files['input'],train_files['target'], pathological_file_path, pathological_file_extension=".h5",random_seed=42)
    val_dataset = TMEDataSet(val_files['input'],val_files['target'], pathological_file_path, pathological_file_extension=".h5",random_seed=42)
    test_dataset = TMEDataSet(test_files['input'],test_files['target'], pathological_file_path, pathological_file_extension=".h5",random_seed=42)
    
    # Initialize dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



    # Initialize model
    model = TumorMicroEnvironmentModel(config)  # Assuming the model is defined in models_tme.py

    criterion = negative_log_partial_likelihood
    # Define loss function and optimizer

    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Move data to gpu
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            data = [d.to(device) for d in data]  # Move each tensor in the list to the device
            targets = [t.to(device) for t in targets]  # Move each tensor in the list to the device
            model = model.to(device)
            model.train()
            optimizer.zero_grad()

            # Forward pass
            outputs = model(data)

            survival_targets = torch.cat([targets[1],targets[0]],dim=1)
            if survival_targets.sum() > 0:
                loss = criterion(survival_targets, outputs)
            else:
                loss = torch.tensor(0.0, device=device)

            exit()


            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, targets in val_loader:
                data = data.to(device)
                data.append(None)
                targets = targets.to(device)
                outputs = model(data)
                survival_targets = torch.cat([targets[1],targets[0]],dim=1)
                if survival_targets.sum() > 0:
                    loss = criterion(survival_targets, outputs)
                    val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")
        # Save model checkpoint
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
    
    # Test the model
