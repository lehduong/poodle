import os
import random
import time
import collections
import tqdm
import torch

import numpy as np
import os.path as osp
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import MultiStepLR, StepLR, CosineAnnealingLR

import datasets
import models
from datasets.ssl_helpers import ssl_collate_fn
from datasets.sampler import sample_case, nonuniform_sample_case
from models import forgiving_state_restore
from utils import configuration
from utils.evaluate import AverageMeter, accuracy, compute_confidence_interval
from utils.losses import DistillKL, SmoothCrossEntropy
from utils.util import save_pickle, load_pickle, save_checkpoint, load_checkpoint, setup_logger, rand_bbox, print_dict
from fsl.poodle import poodle
from fsl.tim import tim
from fsl.laplacian import laplacian
from fsl.kmeans import kmeans, soft_kmeans, bayes_kmeans
from fsl.distance_based import distance_based_classifier

def main():
    global args, best_prec1
    best_prec1 = 0
    args = configuration.parser_args()
    ### initial logger
    log = setup_logger(args.save_path + args.log_file)
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.benchmark = True

    cudnn.deterministic = True
    # create model
    log.info("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=args.num_classes, remove_linear=args.do_meta_train)

    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    if args.label_smooth > 0:
        criterion = SmoothCrossEntropy(epsilon=args.label_smooth).cuda()

    else:
        criterion = nn.CrossEntropyLoss().cuda()

    optimizer = get_optimizer(model)

    if args.pretrain:
        pretrain = args.pretrain + '/checkpoint.pth.tar'
        if os.path.isfile(pretrain):
            log.info("=> loading pretrained weight '{}'".format(pretrain))
            checkpoint = torch.load(pretrain)
            model_dict = model.state_dict()
            params = checkpoint['state_dict']
            params = {k: v for k, v in params.items() if k in model_dict}
            model_dict.update(params)
            model.load_state_dict(model_dict)
        else:
            log.info('[Attention]: Do not find pretrained model {}'.format(pretrain))

    # resume from an exist checkpoint
    if os.path.isfile(args.save_path + '/checkpoint.pth.tar') and args.resume == '':
        args.resume = args.save_path + '/checkpoint.pth.tar'

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            # scheduler.load_state_dict(checkpoint['scheduler'])
            model = forgiving_state_restore(model, checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log.info("=> loaded checkpoint '{}' (epoch {})"
                     .format(args.resume, checkpoint['epoch']))
        else:
            log.info('[Attention]: Do not find checkpoint {}'.format(args.resume))

    # Data loading code
    if args.evaluate:
        do_extract_and_evaluate(model, log)
        return

    args.enlarge = False
    if args.do_meta_train:
        sample_info = [args.meta_train_iter, args.meta_train_way, args.meta_train_shot, args.meta_train_query]
        train_loader = get_dataloader('train', not args.disable_train_augment, sample=sample_info)
    else:
        train_loader = get_dataloader('train', not args.disable_train_augment, shuffle=True)

    sample_info = [args.meta_val_iter, args.meta_val_way, args.meta_val_shot, args.meta_val_query]
    val_loader = get_dataloader('val', False, sample=sample_info)

    scheduler = get_scheduler(len(train_loader), optimizer)
    tqdm_loop = warp_tqdm(list(range(args.start_epoch, args.epochs)))

    # crossentropy training
    for epoch in tqdm_loop:
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, scheduler, log)
        scheduler.step(epoch)
        # evaluate on meta validation set
        is_best = False
        if (epoch + 1) % args.meta_val_interval == 0:
            prec1 = meta_val(val_loader, model)
            log.info('Meta Val {}: {}'.format(epoch, prec1))
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            if not args.disable_tqdm:
                tqdm_loop.set_description('Best Acc {:.2f}'.format(best_prec1 * 100.))

        # remember best prec@1 and save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            # 'scheduler': scheduler.state_dict(),
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, folder=args.save_path)
    
    print("Start Distillation")
    log.info("Start Distillation")
    n_generations = 2
    student = model
    
    for i in range(n_generations):
        print("Distillation for generation: " + str(i))
        log.info("Distillation for generation: " + str(i))
        
        # clean and set up model, optimizer, scheduler and stuff
        if i > 0:
            checkpoint = torch.load('{}/model_best.pth.tar'.format(osp.join(args.save_path, 'student_'+str(i-1))))
        else:
            checkpoint = torch.load('{}/model_best.pth.tar'.format(args.save_path))
        student.load_state_dict(checkpoint['state_dict'])
        teacher = student
        student = models.__dict__[args.arch](num_classes=args.num_classes, remove_linear=args.do_meta_train)
        student = torch.nn.DataParallel(student).cuda()
        optimizer = get_optimizer(student)
        scheduler = get_scheduler(len(train_loader), optimizer)
        tqdm_loop = warp_tqdm(list(range(0, args.epochs)))
        best_prec1 = 0
        
        for epoch in tqdm_loop:
            # train for one epoch
            train_distil(train_loader, teacher, student, criterion, optimizer, epoch, scheduler, log)
            scheduler.step(epoch)
            # evaluate on meta validation set
            is_best = False
            if (epoch + 1) % args.meta_val_interval == 0:
                prec1 = meta_val(val_loader, student)
                log.info('Meta Val {}: {}'.format(epoch, prec1))
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                if not args.disable_tqdm:
                    tqdm_loop.set_description('Best Acc {:.2f}'.format(best_prec1 * 100.))

            # remember best prec@1 and save checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                # 'scheduler': scheduler.state_dict(),
                'arch': args.arch,
                'state_dict': student.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, folder=osp.join(args.save_path, 'student_'+str(i)))
    
    model = student
    
    # do evaluate at the end
    args.enlarge = True
    do_extract_and_evaluate(model, log)

def metric_prediction(gallery, query, train_label, metric_type):
    gallery = gallery.view(gallery.shape[0], -1)
    query = query.view(query.shape[0], -1)
    distance = get_metric(metric_type)(gallery, query)
    predict = torch.argmin(distance, dim=1)
    predict = torch.take(train_label, predict)

    return predict

def meta_val(test_loader, model, train_mean=None):
    top1 = AverageMeter()
    model.eval()

    with torch.no_grad():
        tqdm_test_loader = warp_tqdm(test_loader)
        for i, (inputs, target) in enumerate(tqdm_test_loader):
            target = target.cuda(0, non_blocking=True)
            output = model(inputs, True)[0].cuda(0)
            if train_mean is not None:
                output = output - train_mean
            train_out = output[:args.meta_val_way * args.meta_val_shot]
            train_label = target[:args.meta_val_way * args.meta_val_shot]
            test_out = output[args.meta_val_way * args.meta_val_shot:]
            test_label = target[args.meta_val_way * args.meta_val_shot:]
            train_out = train_out.reshape(args.meta_val_way, args.meta_val_shot, -1).mean(1)
            train_label = train_label[::args.meta_val_shot]
            prediction = metric_prediction(train_out, test_out, train_label, args.meta_val_metric)
            acc = (prediction == test_label).float().mean()
            top1.update(acc.item())
            if not args.disable_tqdm:
                tqdm_test_loader.set_description('Acc {:.2f}'.format(top1.avg * 100))
    return top1.avg

def train_distil(train_loader, t_model, s_model, criterion, optimizer, epoch, scheduler, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    s_model.train()
    t_model.eval()

    end = time.time()
    tqdm_train_loader = warp_tqdm(train_loader)

    kd_criterion = DistillKL(T=4)

    for i, (input, target) in enumerate(tqdm_train_loader):
        if args.scheduler == 'cosine':
            scheduler.step(epoch * len(train_loader) + i)
        # measure data loading time
        data_time.update(time.time() - end)

        if args.do_ssl:
            target, rot_target = target
            rot_target = rot_target.cuda(non_blocking=True)
        if args.do_meta_train:
            target = torch.arange(args.meta_train_way)[:, None].repeat(1, args.meta_train_query).reshape(-1).long()
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            t_output, t_rot_output = t_model(input)
        s_output, s_rot_output = s_model(input)
        
        supervised_loss = criterion(s_output, target)
        kd_loss = kd_criterion(s_output, t_output)
        if args.do_ssl:
            supervised_loss += criterion(s_rot_output, rot_target)
            kd_loss += kd_criterion(s_rot_output, t_rot_output)
        loss = 0.5 * supervised_loss + 0.5 * kd_loss
        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prec1, prec5 = accuracy(s_output, target, topk=(1, 5))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        if not args.disable_tqdm:
            tqdm_train_loader.set_description('Acc {:.2f}'.format(top1.avg))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            log.info('Epoch: [{0}][{1}/{2}]\t'
                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                     'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

def train(train_loader, model, criterion, optimizer, epoch, scheduler, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    tqdm_train_loader = warp_tqdm(train_loader)

    for i, (input, target) in enumerate(tqdm_train_loader):
        if args.scheduler == 'cosine':
            scheduler.step(epoch * len(train_loader) + i)
        # measure data loading time
        data_time.update(time.time() - end)

        if args.do_ssl:
            target, rot_target = target
            rot_target = rot_target.cuda(non_blocking=True)
        if args.do_meta_train:
            target = torch.arange(args.meta_train_way)[:, None].repeat(1, args.meta_train_query).reshape(-1).long()
        target = target.cuda(non_blocking=True)

        # compute output
        r = np.random.rand(1)
        if args.beta > 0 and r < args.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            # compute output
            output = model(input)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        else:
            output, rot_output = model(input)
            loss = criterion(output, target) 
            if args.do_ssl:
                loss += criterion(rot_output, rot_target)
            if args.do_meta_train:
                output = output.cuda(0)
                shot_proto = output[:args.meta_train_shot * args.meta_train_way]
                query_proto = output[args.meta_train_shot * args.meta_train_way:]
                shot_proto = shot_proto.reshape(args.meta_train_way, args.meta_train_shot, -1).mean(1)
                output = -get_metric(args.meta_train_metric)(shot_proto, query_proto)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        if not args.disable_tqdm:
            tqdm_train_loader.set_description('Acc {:.2f}'.format(top1.avg))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            log.info('Epoch: [{0}][{1}/{2}]\t'
                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                     'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

def get_metric(metric_type):
    METRICS = {
        'cosine': lambda gallery, query: 1. - F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
        'euclidean': lambda gallery, query: ((query[:, None, :] - gallery[None, :, :]) ** 2).sum(2),
        'l1': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=1, dim=2),
        'l2': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=2, dim=2),
    }
    return METRICS[metric_type]

def get_scheduler(batches, optimiter):
    """
    cosine will change learning rate every iteration, others change learning rate every epoch
    :param batches: the number of iterations in each epochs
    :return: scheduler
    """
    SCHEDULER = {'step': StepLR(optimiter, args.lr_stepsize, args.lr_gamma),
                 'multi_step': MultiStepLR(optimiter, milestones=args.scheduler_milestones,
                                           gamma=args.lr_gamma),
                 'cosine': CosineAnnealingLR(optimiter, batches * args.epochs, eta_min=1e-9)}
    return SCHEDULER[args.scheduler]

def get_optimizer(module):
    OPTIMIZER = {'SGD': torch.optim.SGD(module.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
                                        nesterov=args.nesterov),
                 'Adam': torch.optim.Adam(module.parameters(), lr=args.lr, weight_decay=args.weight_decay)}
    return OPTIMIZER[args.optimizer]

def get_dataloader(split, aug=False, shuffle=True, out_name=False, sample=None):

    # sample: iter, way, shot, query
    if aug:
        transform = datasets.with_augment(args.img_size, disable_random_resize=args.disable_random_resize, jitter=args.jitter)
    else:
        transform = datasets.without_augment(args.img_size, enlarge=args.enlarge)
    sets = datasets.DatasetFolder(args.data, args.split_dir, split, transform, out_name=out_name)
    if sample is not None:
        sampler = datasets.CategoriesSampler(sets.labels, *sample)
        loader = torch.utils.data.DataLoader(sets, batch_sampler=sampler,
                                             num_workers=args.workers, pin_memory=True)
    else:
        loader = torch.utils.data.DataLoader(sets, batch_size=args.batch_size, shuffle=shuffle,
                                             num_workers=args.workers, pin_memory=True, collate_fn=ssl_collate_fn if args.do_ssl else None)
    return loader

def warp_tqdm(data_loader):
    if args.disable_tqdm:
        tqdm_loader = data_loader
    else:
        tqdm_loader = tqdm.tqdm(data_loader, total=len(data_loader))
    return tqdm_loader

def extract_feature(train_loader, val_loader, model, save_path, tag='last', enlarge=True):
    # return out mean, fcout mean, out feature, fcout features
    save_dir = '{}/{}/{}'.format(save_path, tag, enlarge)
    if os.path.isfile(save_dir + '/output.plk'):
        data = load_pickle(save_dir + '/output.plk')
        return data
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    model.eval()
    with torch.no_grad():
        train_out_feature = []
        for _, (inputs, _) in enumerate(warp_tqdm(train_loader)):
            outputs, _ = model(inputs, True)
            train_out_feature.append(outputs.cpu().data.numpy())
        train_out_feature = np.concatenate(train_out_feature, axis=0)
                
        val_output_dict = collections.defaultdict(list)
        for _, (inputs, labels) in enumerate(warp_tqdm(val_loader)):
            # compute output
            outputs, _ = model(inputs, True)
            outputs = outputs.cpu().data.numpy()
            for out, label in zip(outputs, labels):
                val_output_dict[label.item()].append(out)
        all_info = [train_out_feature, val_output_dict]
        save_pickle(save_dir + '/output.plk', all_info)
        return all_info
    
def meta_evaluate(data, shot, train_feature, args):
    super_tasks = []
    results = {}
    
    for _ in warp_tqdm(range(args.meta_test_iter)):
        train_data, test_data, train_label, test_label = sample_case(args, data, shot) # use nonuniform_sample_case if non_uniform few-shot tasks
        super_tasks.append((train_data, test_data, train_label, test_label))

    super_train_data = np.stack([task[0] for task in super_tasks], 0)
    super_test_data = np.stack([task[1] for task in super_tasks], 0)
    super_train_label = np.stack([task[2] for task in super_tasks], 0)
    super_test_label = np.stack([task[3] for task in super_tasks], 0)

    accs = tim(super_train_data, super_test_data, super_train_label, super_test_label, shot, args=args)
    results["tim"] = accs
    accs = poodle(super_train_data, super_test_data, super_train_label, super_test_label, shot, train_feature=train_feature, transductive=False, args=args)
    results["poodle_B"] = accs
    accs = poodle(super_train_data, super_test_data, super_train_label, super_test_label, shot, train_feature=None, transductive=False, args=args)
    results["poodle_R"] = accs
    accs = distance_based_classifier(super_train_data, super_test_data, super_train_label, super_test_label, shot, args=args)
    results["cosine"] = accs

    results = {k: compute_confidence_interval(v) for k, v in results.items()}
    return results

def do_extract_and_evaluate(model, log):
    args.do_ssl = False
    tags = 'best'
    train_loader = get_dataloader('train', aug=False, shuffle=False, out_name=False)
    val_loader = get_dataloader('test', aug=False, shuffle=False, out_name=False)

    ## With the best model trained on source dataset
    load_checkpoint(args, model, 'best')
    train_feature, out_dict = extract_feature(train_loader, val_loader, model, '{}/{}/{}'.format(args.save_path, tags, args.enlarge), tags)
    
    results = meta_evaluate(out_dict, 1, train_feature, args)
    print_dict(results, 'Best 1-shot')

    results = meta_evaluate(out_dict, 5, train_feature, args)
    print_dict(results, 'Best 5-shot')

if __name__ == '__main__':
    main()

