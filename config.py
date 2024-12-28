import argparse  
from torch.cuda import is_available
def get_options(parser=argparse.ArgumentParser()):  
  
    parser.add_argument('--workers', type=int, default=0,  
                        help='number of data loading workers, you had better put it 4 times of your gpu')  
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size, default=64')  
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for, default=10')  
    parser.add_argument('--lr', type=float, default=1e-3, help='select the learning rate, default=1e-3')  
    parser.add_argument('--seed', type=int, default=118, help="random seed")  
    parser.add_argument('--cuda', action='store_true', default=is_available(), help='enables cuda')  
    parser.add_argument('--checkpoint_path',type=str,default='', help='Path to load a previous trained model if not empty (default empty)')  
    parser.add_argument('--output',action='store_true',default=True,help="shows output")
    parser.add_argument('--image_size', type=int, default=28, help='input image size, default=64')
  
    opt = parser.parse_args()  
  
    if opt.output:  
        print(f'num_workers: {opt.workers}')  
        print(f'batch_size: {opt.batch_size}')
        print(f'image_size: {opt.image_size}')
        print(f'epochs: {opt.epochs}')  
        print(f'learning rate : {opt.lr}')  
        print(f'manual_seed: {opt.seed}')  
        print(f'cuda enable: {opt.cuda}')  
        print(f'checkpoint_path: {opt.checkpoint_path}')

    return opt  
  
if __name__ == '__main__':  
    opt = get_options()