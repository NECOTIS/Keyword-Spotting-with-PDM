from run_train import train, get_argparser

if __name__ == '__main__':
    parser = get_argparser()
    job_args = parser.parse_args()
    
    job_args.n_epochs = 150
    job_args.batch_size = 10
    job_args.paralif_spike_mode = "D"
    job_args.loss_mode = "cumsum"
    job_args.paralif_tau_mem = 0.03
    job_args.paralif_tau_syn = 0.
    job_args.n_layers = 4
    job_args.paralif_rec = [False, False, True, True]
    job_args.paralif_conv = True
    job_args.data_aug = True
    job_args.shift_factor = 0.3
    job_args.paralif_delay = 30
    job_args.pdm_factor = 10
    job_args.paralif_k_size = [job_args.pdm_factor*3, 3, 3, 3]
    job_args.paralif_dilation = [1,2,2,2]
    job_args.n_hidden = [128, 128, 128, 128]
    job_args.scheduler = True
    job_args.dir = "results/"
    

    train(job_args)
    
    
    
















