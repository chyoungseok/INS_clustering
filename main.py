import os, argparse
from modules.utils import str2bool, append_default_path, get_ExperimentInfo
from modules.model_utils import get_model_with_experiment_info, model_initiation, train_model, eval_model
from modules import load

class main:
    '''
    Executed when called in terminal

    1. load data
        - load scalograms
    
    2. read experiment info
        - information about model parameters (kernel_size, pool_size, ...)
        - use load.load_data class
        
    3. load keras model of autoencoder
        - based on the experiment info
        - use utils.get_model_with_experiment_info
        
    
    
    '''

    def __init__(self, **params):
        # get parameters from argparse
        self.is_lr_reducer = params['is_lr_reducer']
        self.learning_rate = params['learning_rate']
        self.epochs = params['epochs']
        self.batch_size = params['batch_size']
        
        # 1. Load data (scalograms)
        print("\n---------Load Data---------")
        _load = load.load_data()
        _load.stack_data_and_label()

        # 2. Read experiment info
        print("\n---------Read Experiment Info---------")
        df_experiment = get_ExperimentInfo()
        print("%d experiments will be implemented\n" % len(df_experiment))

        self._load = _load
        self.df_experiment = df_experiment
    
    def run(self):
        for i in range(len(self.df_experiment)):
            temp_experiment = self.df_experiment.loc[i+1, :]
            model = get_model_with_experiment_info(temp_experiment=temp_experiment)
            model = model_initiation(model=model, is_lr_reducer=self.is_lr_reducer, learning_rate=self.learning_rate)
            
            train_model(X_train=self._load.train_data,
                        X_val=self._load.test_data,
                        test_name=str(i+1),
                        is_lr_reducer=self.is_lr_reducer,
                        model=model,
                        epochs=self.epochs,
                        batch_size=self.batch_size)

            eval_model(X_train=self._load.train_data,
                       X_test=self._load.test_data,
                       test_name=str(i+1))
            
            
def parse_minimal_args(parser):
    parser.add_argument("--gpu_num", type=str, default='5')
    parser.add_argument("--is_lr_reducer", type=str2bool, default=True) # if True, use learning_rate_reducer
    parser.add_argument("--epochs", type=int, default=50) # number of training epochs
    parser.add_argument("--learning_rate", type=float, default=0.01) # laerning rate for training
    parser.add_argument("--batch_size", type=int, default=32) # batch size for training
    return parser

def print_args(args):
    print("  -------------------------------")
    print("  | gpu_num: %s" % (args.gpu_num))
    print("  | is_lr_reducer: %s" % (str(args.is_lr_reducer)))
    print("  | epochs: %d" % (args.epochs))
    print("  | learning_rate: %f" % (args.learning_rate))
    print("  | batch_size: %d" % (args.batch_size))
    print("  -------------------------------")
    pass


if __name__ == '__main__':
    # get parser
    parser = argparse.ArgumentParser()
    parser = parse_minimal_args(parser)
    args = parser.parse_args()
    print_args(args)
    
    # GPU selection
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_num
    
    _, _, _ = append_default_path() # append all the required paths (moduels, csv_files, scalogram)

    _main = main(is_lr_reducer=args.is_lr_reducer,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size)
    
    _main.run()



    
    