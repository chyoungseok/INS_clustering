import os, argparse
from modules.utils import str2bool

def parse_minimal_args(parser):
    parser.add_argument("--gpu_num", type=str, default='0')
    parser.add_argument("--model_selection", type=str, default='upsampling') # 'upsampling', 'deconv'
    parser.add_argument("--nb_layers", type=int, default=1) # number of hidden layers for encoders and decoders
    parser.add_argument("--loss", type=str, default='MSE') # specify loss function
    parser.add_argument("--is_lr_reducer", type=str2bool, default=False) # if True, use learning_rate_reducer
    parser.add_argument("--epochs", type=int, default=2) # number of training epochs
    parser.add_argument("--learning_rate", type=float, default=0.01) # laerning rate for training
    parser.add_argument("--batch_size", type=int, default=32) # batch size for training
    parser.add_argument("--is_val", type=str2bool, default=False) # if True, use validation set
    parser.add_argument("--is_test", type=str2bool, default=False) # if True, use test set
    parser.add_argument("--test_name", type=str, default='TEST') # test_name of current session
    return parser

def print_args(args):
    print("  -------------------------------")
    print("  | gpu_num: %s" % (args.gpu_num))
    print("  | model_selection: %s" % (args.model_selection))
    print("  | nb_layers: %d" % (args.nb_layers))
    print("  | loss: %s" % (args.loss))
    print("  | is_lr_reducer: %s" % (str(args.is_lr_reducer)))
    print("  | epochs: %d" % (args.epochs))
    print("  | learning_rate: %f" % (args.learning_rate))
    print("  | batch_size: %d" % (args.batch_size))
    print("  | is_val: %s" % (args.is_val))
    print("  | is_test: %s" % (args.is_test))
    print("  | test_name: %s" % (args.test_name))    
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
    
    
    