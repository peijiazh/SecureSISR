import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import Network

from Dataset import DatasetFromHdf5


def init():
    # Training settings
    parser = argparse.ArgumentParser(description="the VDSR of Pytorch")
    # #
    # # # batch_size: number of image data fed into the model each time
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size")
    # Number of training epochs
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train for")
    # Learning rate
    parser.add_argument("--lr", type=float, default=0.1, help="Learning Rate. Default=0.1")
    # Dynamic learning rate adjustment coefficient
    parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
    # Use CUDA
    parser.add_argument("--cuda", action="store_true",default=True,help="Use cuda?")
    # Pre-trained weights
    parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
    # Starting epoch
    parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
    # Gradient clipping coefficient
    parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
    # Single thread
    parser.add_argument("--num_workers", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
    # Optimizer momentum
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
    # Regularization coefficient
    parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
    # Pre-training
    parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
    # # Adam parameters
    # parser.add_argument('--beta1', type=float, default=0.9, help='ADAM beta1')
    # parser.add_argument('--beta2', type=float, default=0.999, help='ADAM beta2')
    # parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')
    # Default GPU is 0
    parser.add_argument("--gpus", default="2,3", type=str, help="gpu ids (default: 0)")

    return parser

def main():

    parser = init()
   
    opt = parser.parse_args()
   
    print(opt)
    # CUDA GPU parameter settings
    cuda = opt.cuda

    # GPU configuration
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
    # Random seed parameter
    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)

    if cuda:
        # Set fixed random seed so that each time this .py file is run,
        # the generated random numbers are the same
        torch.cuda.manual_seed(opt.seed)

    # Enable acceleration, optimize runtime efficiency
    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = DatasetFromHdf5(file_path="matlab_rgb/train1.h5")
    # train_set = DatasetFromHdf5(file_path="matlab/train_x4_nobic.h5")

    #training_data_loader = DataLoader(dataset=train_set,num_workers=opt.num_workers, batch_size=opt.batch_size, shuffle=True)
    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True)

    print("===> Building model")
    model = Network()

    criterion = nn.MSELoss(reduction='sum')    ###my


    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:

        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")

    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)  ###my
   
    print("===> Training")

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        train(opt,training_data_loader, optimizer, model, criterion, epoch)
        # scheduler.step()
        save_checkpoint(model, epoch)

def adjust_learning_rate(opt, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))  ###my

    return lr

def train(opt,training_data_loader, optimizer, model, criterion, epoch):
    
    lr = opt.lr
    #####  my  #####################################################################3
    lr = adjust_learning_rate(opt, epoch-1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))
    #####  my  #####################################################################3

    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        



        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

       
        loss= criterion(model(input), target)
        
        optimizer.zero_grad() ######mark my
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)   ##### mark my
        optimizer.step()

        if iteration%100 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.item()))

def save_checkpoint(model, epoch):
    model_out_path = "model_testrgb/" + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("model_testrgb/"):
        os.makedirs("model_testrgb/")

    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()
