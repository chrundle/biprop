import time
import torch
import tqdm
import torch.nn.functional as F

from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter

__all__ = ["train", "validate", "modifier"]


# Set threshold for score value to enforce global pruning across network layers
def global_prune_threshold(model, args):
    # Loop over all model parameters and concat flattened score tensors
    # Initialize list for storing parameters with scores
    p_list = []
    # Loop over all model parameters to extract scores
    for n, m in model.named_modules():
      # Only add parameters that have scores as attributes
      if hasattr(m,'scores'):
        # Add flattened clone of scores.abs() to p_list
        p_list.append(m.scores.clone().abs().flatten())

    # Concatenate flattened clones
    z = torch.cat(p_list)
    # Sort z
    z,_ = z.sort()
    # Determine number of elements to prune
    p_idx = int((1-args.prune_rate) * z.numel())
    # Identify prune_threshold for values in bottom 
    # (1-args.prune_rate) percent of scores
    prune_threshold = z[p_idx-1]
    #print("prune_threshold = ", prune_threshold)

    # Loop over all model parameters to update prune_threshold
    for n, m in model.named_modules():
      # Only add parameters that have scores as attributes
      if hasattr(m,'scores'):
        # Pass prune_threshold value to model parameters
        m.set_prune_threshold(prune_threshold)

    # Exit function
    return

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
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
          if args.jsd:
            images_all = torch.cat(images, 0).cuda(args.gpu, non_blocking=True)
          else:
            images = images.cuda(args.gpu, non_blocking=True)

          target = target.cuda(args.gpu, non_blocking=True)

        # Write scores and weights to tensorboard at beginning of every other epoch
        if args.histograms:
          if (i % (num_batches * batch_size) == 0) and (epoch % 2 == 0):
            for param_name in model.state_dict():
              #print(param_name)
              # Only write scores for now (not weights and batch norm parameters since the pytorch parms don't actually change)
              #if 'score' not in param_name:
              #if 'score' in param_name or 'weight' in param_name:
                #print(param_name, model.state_dict()[param_name])
                writer.add_histogram(param_name, model.state_dict()[param_name], epoch)

        # Check if global pruning is being used
        if args.conv_type == "GlobalSubnetConv":
          # Set prune_threshold for all layers in model
          global_prune_threshold(model, args)

        # compute loss without Jensen-Shannon divergence
        if args.jsd == False:
          output = model(images)

          loss = criterion(output, target)

          # measure accuracy and record loss
          acc1, acc5 = accuracy(output, target, topk=(1, 5))
          losses.update(loss.item(), images.size(0))
          top1.update(acc1.item(), images.size(0))
          top5.update(acc5.item(), images.size(0))

        else: # compute loss with Jensen-Shannon divergence
          logits_all = model(images_all)
          logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))
          output = logits_clean # This is just used for accuracy function

          # Cross-entropy is only computed on clean images
          loss = criterion(logits_clean, target)

          # Terms for Jensen-Shannon divergence
          p_clean, p_aug1, p_aug2 = F.softmax(logits_clean, dim=1), F.softmax(logits_aug1, dim=1), F.softmax(logits_aug2, dim=1)

          # Clamp mixture distribution to avoid exploding KL divergence
          p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
          loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') + F.kl_div(p_mixture, p_aug1, reduction='batchmean') + F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

          # measure accuracy and record loss
          acc1, acc5 = accuracy(logits_clean, target, topk=(1, 5))
          losses.update(loss.item(), images[0].size(0))
          top1.update(acc1.item(), images[0].size(0))
          top5.update(acc5.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        #torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        loss.backward()
        # EDITED
        #print(torch.norm(torch.cat([p.grad.view(-1) for p in model.parameters()])))
        if args.grad_clip: torch.nn.utils.clip_grad_value_(model.parameters(),1) 
        #print(torch.norm(torch.cat([p.grad.view(-1) for p in model.parameters()])))
        #for param_name in model.state_dict(): print(param_name, str(model.state_dict()[param_name])[:50])
        #torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        # end
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

        if i % args.print_freq == 0:
            t = (num_batches * epoch + i) * batch_size
            progress.display(i)

            #_, predicted = torch.max(output, 1)
            progress.write_to_tensorboard(writer, prefix="train", global_step=t)

        # Write score gradients to tensorboard at end of every other epoch
        if args.histograms:
            if (i % (num_batches * batch_size) == 0) and (epoch % 2 == 0):
            #if ((i+1) % (num_batches-1) == 0) and (epoch % 2 == 0):
              params = list(model.parameters())
              param_names = list(model.state_dict())
              for j in range(len(params)):
                  if params[j].grad is not None:
                 # if 'score' in param_names[j] or 'weight' in param_names[j]:
                 # if 'score' not in param_name and params[j].grad is not None:
                      #print(param_names[j])
                      #print(params[j].grad)
                      writer.add_histogram(param_names[j] + '.grad', params[j].grad, epoch)
                  else:
                      writer.add_histogram(param_names[j] + '.grad', 0, epoch)
              #for param_name in model.state_dict():
              #  if 'score' in param_name:
              #    writer.add_histogram(param_name + '.grad', model.state_dict()[param_name].grad, epoch)
              #params = list(model.parameters())
              #for j in range(len(params)):
              #  writer.add_histogram('Layer' + str(j) + 'grad', params[j].grad, epoch)

    # Write final scores and weights to tensorboard
    if args.histograms:
        for param_name in model.state_dict():
          #writer.add_histogram(param_name, model.state_dict()[param_name], epoch)
          # Only write scores for now (not weights and batch norm parameters since the pytorch parms don't actually change)
          #if 'score' not in param_name:
          #print(param_name, model.state_dict()[param_name])
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

            # Check if global pruning is being used
            if args.conv_type == "GlobalSubnetConv":
              # Set prune_threshold for all layers in model
              global_prune_threshold(model, args)

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
                #_, predicted = torch.max(output, 1)
                #print(predicted,target)

        progress.display(len(val_loader))

        if writer is not None:
            progress.write_to_tensorboard(writer, prefix="test", global_step=epoch)

    return top1.avg, top5.avg

def modifier(args, epoch, model):
    return
