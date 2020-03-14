import argparse

import torch
import torch.optim

from model import SeedlingVision

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("{0} used".format(device))

    # Parsing arguments ------------------------------------------
    parser = argparse.ArgumentParser("Training Model")
    parser.add_argument("--n_epochs", help="Number of epochs to train model", type=int, default=100)
    
    args = parser.parse_args()

    ################################## SETUP #####################################
    lr               = 10e-4
    num_epochs        = args.n_epochs
    mini_batch_size  = 5
    threshold_reward = 1e5

    log_period = 1000
    num_episodes = args.n_episodes

    save_path = None
    load_model_path = None
    
    checkpoint_period = 500 # Create a model file after this many episodes. Leave as None for no checkpoints

    ##########################################################################
    # Checking if model is savable
    assert os.path.isdir(os.path.dirname(save_path)), "Folder does not exist."
    if os.path.isfile(save_path):
        os.system('xdg-open "%s"' % os.path.dirname(save_path))
        raise AssertionError("File already exist")
    
    # Initializing model and training
    model = SeedlingVision().to(device)
    try:
        model.load_state_dict(torch.load(load_model_path)) 
        print("Model loaded")
    except:
        pass


    optimizer = optim.Adam(model.parameters(), lr=lr)

    
    # Training
    early_stop = False # Training stops early if the threshold reward is reached
    for i_epoch in range(num_epochs):
        for i_iter in df:

            optimizer.zero_grad()

            loss = get_loss(outputs, labels)
            loss.backward()
            optimizer.step() 

