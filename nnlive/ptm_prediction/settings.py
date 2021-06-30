#EpyNN/nnlive/ptm_prediction/settings.py


config = {
    'experiment_name': 'OGLCNAC',
    'logs_frequency': 25,
    'model_save': False,
    'dsets_save': False,
    'hPars_save': False,
    'runData_save': False,
    'plot_display': True,
    'plot_save': False,
    'directory_clear': False,
    'dataset_target': 1,
    'metrics_target': 'accuracy',
    'N_SAMPLES': None,
    'metrics_list': ['accuracy','recall','precision','BCE','MSE','MAE','RMSLE','KLD'],
    'metrics_plot': ['accuracy','BCE']
}

hPars = {
    'softmax_temperature': 1,
    'ELU_alpha': 1,
    'LRELU_alpha': 1,
    'min_epsilon': 1e-9,
    'decay_k': 0,
    'cycling_n': 1,
    'descent_d': 1,
    'schedule_mode': 'steady',
    'training_epochs': 1000,
    'batch_number': 1,
    'learning_rate': 0.25,
    'regularization_l2': 0,
    'regularization_l1': 0
}
