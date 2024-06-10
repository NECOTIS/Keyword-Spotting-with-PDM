from paralif_model import ParaLIF_Net
from gsc_dataset import InMemoryGSCDataset, DataAug
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os
from datetime import datetime
import json
import numpy as np

def build_model(args, dataset):
    example_input, _ = dataset[0]
    n_input,sequence_len = example_input.shape
    n_output = len(dataset.labels)

    model = ParaLIF_Net(n_input, n_output, args.n_layers, args.n_hidden, args.paralif_spike_mode, args.paralif_rec, args.paralif_conv, 
                 args.paralif_k_size, args.device, args.paralif_tau_mem, args.paralif_tau_syn, args.paralif_dilation, args.paralif_delay, 
                 args.paralif_conv_groups).to(args.device)

    return model


def save_results(model, OUTPUT, args, subdir='', targets=[], predictions=[]):
    output_dir = f"outputs/{args.dir}{subdir}"
    timestamp = int(datetime.timestamp(datetime.now()))
    filename = output_dir+f"results_{str(timestamp)}{'' if args.slurm_id is None else '_'+str(args.slurm_id)}.json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(OUTPUT, f)

    if args.save_model:
        modelname = output_dir+f"model_{str(timestamp)}{'' if args.slurm_id is None else '_'+str(args.slurm_id)}.pt"
        torch.save(model.state_dict(), modelname)
    if args.save_confusion_matrix:
        confusion_mat_name = output_dir+f"confusion_mat_{str(timestamp)}{'' if args.slurm_id is None else '_'+str(args.slurm_id)}.pt"
        confusion_mat = confusion_matrix(targets, predictions)
        torch.save(torch.tensor(confusion_mat), confusion_mat_name)

def reduce(x, mode='last'):
    if mode=='mean' : return torch.mean(x,1)
    elif mode=='max': return torch.max(x,1)[0] 
    elif mode=='cumsum': return F.softmax(x,dim=2).sum(1) #https://www.frontiersin.org/articles/10.3389/fnins.2022.865897/full
    else: return x[:,-1,:]

def data_aug(args):
    if not args.data_aug: return None
    return DataAug(shift_factor=args.shift_factor)
    

def confusion_matrix(true, pred):
  K = len(np.unique(true)) # Number of classes 
  result = np.zeros((K, K))

  for i in range(len(true)):
    result[true[i]][pred[i]] += 1

  return result


def train(args):
    args.device = args.device if torch.cuda.is_available() else "cpu"
    print(f'Torch is using: {args.device}')
                    
    train_set = InMemoryGSCDataset(subset="train", root=args.data_dir, n_examples=args.n_examples, device=args.device,
                              pdm_factor=args.pdm_factor, transform=data_aug(args))
    valid_set = InMemoryGSCDataset(subset="valid", root=args.data_dir, device=args.device, pdm_factor=args.pdm_factor)
    test_set = InMemoryGSCDataset(subset="test", root=args.data_dir, device=args.device, pdm_factor=args.pdm_factor)

    model = build_model(args, valid_set)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    if args.checkpoint_path:
        model_desc = torch.load(args.checkpoint_path, map_location=torch.device(args.device))
        model.load_state_dict(model_desc['state_dict'])
    

    loss_function = torch.nn.CrossEntropyLoss()
    if args.optim=="Adam": optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim=="Adagrad": optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim=="AdamW": optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim=="Adamax": optimizer = optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim=="RMSprop": optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler: scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.7, threshold=0.01, patience=10, verbose=True)
    metric = lambda x, y: x.argmax(dim=-1).squeeze().eq(y).sum().item()
    
    print(vars(args))
    print(model)
    print(f'n_params: {model.count_parameters()}')
    

    # Dictionary to save hyperparameters and results
    OUTPUT  = {
        "num_parameters": model.count_parameters(),
        "hyperparameters": vars(args),
        "loss_hist": [],
        "spike_rate": [],
        "train_accuracy_hist": [],
        "valid_accuracy_hist": [],
        "test_accuracy_hist": [],
    }
    

    for epoch in range(1,args.n_epochs+1):
        print(f'\nEpoch #{epoch}')
        model.train()
        n_correct_train = 0
        loss_hist = 0
        spike_rate = []
        for data, target in tqdm(train_loader, desc=f"Train #{epoch}"):
            output = model(data)
            optimizer.zero_grad()
            prediction = reduce(output, mode=args.loss_mode)
            n_correct_train += metric(prediction, target)
            loss = loss_function(prediction, target)
            spks = torch.stack([layer.mean_spike_rate for layer in model.net if (layer.__class__.__name__ =='ParaLIF' and layer.fire)])
            spike_rate.append(torch.sum(spks).detach().item())
            loss.backward()
            optimizer.step()
            loss_hist += loss.detach().item()
        train_accuracy = n_correct_train / len(train_set)
        print(f'Train accuracy: {train_accuracy:.5f}')
        OUTPUT["train_accuracy_hist"].append(train_accuracy)
        OUTPUT["loss_hist"].append(loss_hist/len(train_set))
        OUTPUT["spike_rate"].append(np.array(spike_rate).mean())
        if (epoch==10 and train_accuracy<0.1): 
            save_results(model, OUTPUT, args, subdir=f'{epoch}/')
            break
        
        model.eval()
        n_correct_valid = 0
        with torch.no_grad():
            for data, target in tqdm(valid_loader, desc=f"Valid #{epoch}"):
                output = model(data)
                prediction = reduce(output, mode=args.loss_mode)
                n_correct_valid += metric(prediction, target)
        valid_accuracy = n_correct_valid / len(valid_set)
        print(f'Valid accuracy: {valid_accuracy:.5f}')
        OUTPUT["valid_accuracy_hist"].append(valid_accuracy)
        if args.scheduler: scheduler.step(valid_accuracy)
        
        labels,predictions = [],[]
        if (epoch%50==0 or epoch==args.n_epochs):
            n_correct_test = 0
            with torch.no_grad():
                for data, target in tqdm(test_loader, desc=f"Test #{epoch}"):
                    output = model(data)
                    prediction = reduce(output, mode=args.loss_mode)
                    n_correct_test += metric(prediction, target)
                    predictions.append(prediction.argmax(dim=-1).squeeze().cpu().numpy())
                    labels.append(target.squeeze().cpu().numpy())
            test_accuracy = n_correct_test / len(test_set)
            print(f'Test accuracy: {test_accuracy:.5f}')
            OUTPUT["test_accuracy_hist"].append(test_accuracy)
            
            if args.save_confusion_matrix:
                predictions = [item for sublist in predictions for item in list(sublist)]
                labels = [item for sublist in labels for item in list(sublist)]
            
            save_results(model, OUTPUT, args, subdir=f'{epoch}/', targets=labels, predictions=predictions)
        

    
def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slurm_id", type=int, default=None)
    parser.add_argument("--dir", type=str, default="")
    parser.add_argument("--loss_mode", type=str, default="cumsum")

    # dataset
    parser.add_argument("--pdm_factor", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="./Data/")
    parser.add_argument("--n_examples", type=float, default=None)
    parser.add_argument("--data_aug", action='store_true', default=False)
    parser.add_argument("--shift_factor", type=float, default=0)
    
    # model
    parser.add_argument("--model_name", type=str, default="best_model")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_hidden", type=int, default=128)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--paralif_rec", action="store_true", default=False)
    parser.add_argument("--paralif_conv", action="store_true", default=False)
    parser.add_argument("--paralif_k_size", type=int, default=80)
    parser.add_argument("--paralif_conv_groups", type=int, default=1)
    parser.add_argument("--paralif_dilation", type=int, default=1)
    parser.add_argument("--paralif_spike_mode", type=str, default="D")
    parser.add_argument("--paralif_tau_mem", type=float, default=2e-2)
    parser.add_argument("--paralif_tau_syn", type=float, default=2e-2)
    parser.add_argument("--paralif_delay", type=int, default=0)

    
    # training
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--n_epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--optim", type=str, default='Adamax')
    parser.add_argument("--save_model", action="store_true", default=False)
    parser.add_argument("--save_confusion_matrix", action="store_true", default=False)
    parser.add_argument("--scheduler", action="store_true", default=False)


    return parser

if __name__ == '__main__':
    parser = get_argparser()
    train(parser.parse_args())
