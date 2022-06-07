from .ResNet import *
from .DenseNet import *
from .MobileNet import MobileNet as mobilenet
from .WideResNet import wideres
from .ResNet12 import *
from .ConvNet import *


def load_checkpoint(model, type='best'):
    if type == 'best':
        checkpoint = torch.load('{}/model_best.pth.tar'.format(args.save_path))
        print("Load checkpoint from {}/model_best.pth.tar".format(args.save_path))
    elif type == 'last':
        checkpoint = torch.load('{}/checkpoint.pth.tar'.format(args.save_path))
        print("Load checkpoint from {}/checkpoint.pth.tar".format(args.save_path))
    else:
        assert False, 'type should be in [best, or last], but got {}'.format(type)
    model = forgiving_state_restore(model, checkpoint['state_dict'])
def forgiving_state_restore(net, loaded_dict):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    net_state_dict = net.state_dict()
    new_loaded_dict = {}

    for k in net_state_dict:
        if k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():
            new_loaded_dict[k] = loaded_dict[k]
        else:
            print("Skipped loading parameter %s", k)
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)

    return net