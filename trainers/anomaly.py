import time
import torch
import tqdm

from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter


__all__ = ["train", "validate", "modifier"]


def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]",
    )

    # switch to train mode
    model.train()

    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    end = time.time()
    for i, (images, target) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        with torch.autograd.detect_anomaly():
            # measure data loading time
            data_time.update(time.time() - end)

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True)

            ## Write scores and weights to tensorboard at beginning of every other epoch
            #if (i % (num_batches * batch_size) == 0) and (epoch % 2 == 0):
            #  for param_name in model.state_dict():
            #    # Only write scores for now (not weights and batch norm parameters since the pytorch parms don't actually change)
            #    if 'score' in param_name:
            #      writer.add_histogram(param_name, model.state_dict()[param_name], epoch)

            # compute output
            output = model(images)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Clamp updated scores to [-1,1] only when using binarized/quantized activations
            #for param_name in model.state_dict():
            #  if 'score' in param_name:
            #    #print(param_name)
            #    scores = model.state_dict()[param_name]
            #    #scores = torch.clamp(scores,min=-1.0,max=1.0)
            #    scores.clamp_(min=-1.0,max=1.0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            #print(model.state_dict()['module.linear.3.scores'].grad)
            #params = list(model.parameters())
            #print(params[1].grad)

            #if i % args.print_freq == 0:
            #    t = (num_batches * epoch + i) * batch_size
            #    progress.display(i)
            #    progress.write_to_tensorboard(writer, prefix="train", global_step=t)

            ## Write score gradients to tensorboard at end of every other epoch
            #if ((i+1) % (num_batches-1) == 0) and (epoch % 2 == 0):
            #  params = list(model.parameters())
            #  param_names = list(model.state_dict())
            #  for j in range(len(params)):
            #    if 'score' in param_names[j] and params[j].grad is not None:
            #      #print(param_names[j])
            #      #print(params[j].grad)
            #      writer.add_histogram(param_names[j] + '.grad', params[j].grad, epoch)
            #  #for param_name in model.state_dict():
            #  #  if 'score' in param_name:
            #  #    writer.add_histogram(param_name + '.grad', model.state_dict()[param_name].grad, epoch)
            #  #params = list(model.parameters())
            #  #for j in range(len(params)):
            #  #  writer.add_histogram('Layer' + str(j) + 'grad', params[j].grad, epoch)

    # Write final scores and weights to tensorboard
    for param_name in model.state_dict():
      #writer.add_histogram(param_name, model.state_dict()[param_name], epoch)
      # Only write scores for now (not weights and batch norm parameters since the pytorch parms don't actually change)
      if 'score' in param_name:
        writer.add_histogram(param_name, model.state_dict()[param_name], epoch)

    return top1.avg, top5.avg


def validate(val_loader, model, criterion, args, writer, epoch):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
        ):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display(len(val_loader))

        if writer is not None:
            progress.write_to_tensorboard(writer, prefix="test", global_step=epoch)

    return top1.avg, top5.avg

def modifier(args, epoch, model):
    return
