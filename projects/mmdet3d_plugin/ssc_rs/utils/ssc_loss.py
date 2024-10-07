import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


def KL_sep(p, target):
    """
    KL divergence on nonzeros classes
    """
    nonzeros = target != 0
    nonzero_p = p[nonzeros]
    kl_term = F.kl_div(torch.log(nonzero_p), target[nonzeros], reduction="sum")
    return kl_term


def geo_scal_loss(pred, ssc_target):

    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:, 0, :, :, :]
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != 255
    nonempty_target = ssc_target != 0
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))
    )

def precision_loss(pred, ssc_target):

    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:, 0, :, :, :]
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != 255
    nonempty_target = ssc_target != 0
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
    )

def sem_scal_loss(pred, ssc_target):
    with autocast(False):

        # Get softmax probabilities
        pred = F.softmax(pred, dim=1)
        loss = 0
        count = 0
        mask = ssc_target != 255
        n_classes = pred.shape[1]
        for i in range(0, n_classes):

            # Get probability of class i
            p = pred[:, i, :, :, :]

            # Remove unknown voxels
            target_ori = ssc_target
            p = p[mask]
            target = ssc_target[mask]

            completion_target = torch.ones_like(target)
            completion_target[target != i] = 0
            completion_target_ori = torch.ones_like(target_ori).float()
            completion_target_ori[target_ori != i] = 0
            if torch.sum(completion_target) > 0:
                count += 1.0
                nominator = torch.sum(p * completion_target)
                loss_class = 0
                if torch.sum(p) > 0:
                    precision = nominator / (torch.sum(p))
                    try:
                        loss_precision = F.binary_cross_entropy(
                            precision, torch.ones_like(precision)
                        )
                    except:
                        print('precision out of range!!!')
                        precision = torch.clamp(precision, min=0, max=1)
                        loss_precision = F.binary_cross_entropy(
                            precision, torch.ones_like(precision)
                        )

                    loss_class += loss_precision
                if torch.sum(completion_target) > 0:
                    recall = nominator / (torch.sum(completion_target))
                    try:
                        loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                    except:
                        print('loss_recall out of range!!!')
                        recall = torch.clamp(recall, min=0, max=1)
                        loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                    loss_class += loss_recall
                if torch.sum(1 - completion_target) > 0:
                    specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                        torch.sum(1 - completion_target)  +  1e-5
                    )
                    # specificity += 1.

                    if specificity.max() < 1.0:
                        # print('p max = {}, min = {}'.format(p.max(), p.min()))
                        # print('completion_target max = {}, min = {}'.format(completion_target.max(), completion_target.min()))
                        # print('max = {}, min= {}'.format(specificity.max(), specificity.min()))
                        # if specificity.max() > 1.0:
                        #     print('')
                        #     print('')
                        #     print('')
                        #     print('')
                        #     print('find bug!!!')
                        #     print('')
                        #     print('')
                        #     print('')
                        #     print('')
                        #     print('')


                        # specificity = torch.clamp(specificity, min=0, max=1)
                        loss_specificity = F.binary_cross_entropy(
                            specificity, torch.ones_like(specificity)
                        )
                        # loss_specificity = F.binary_cross_entropy_with_logits(
                        #     inverse_sigmoid(specificity, 'F'), torch.ones_like(specificity)
                        # )
                        loss_class += loss_specificity
                    # except:
                    #     print('specificity out of range!!!')
                    #     try:
                    #         print('p max = {}, min = {}'.format(p.max(), p.min()))
                    #         print('completion_target max = {}, min = {}'.format(completion_target.max(), completion_target.min()))
                    #         print('max = {}, min= {}'.format(specificity.max(), specificity.min()))
                    #         print('specificity  = {}'.format(specificity))
                    #     except:
                    #         pass
                    else:
                        print('!!! specificity out of range !!!')
                        print('specificity max = {}, min= {}'.format(specificity.max(), specificity.min()))
                        print('!!! specificity out of range !!!')


                        
                        # specificity = torch.clamp(specificity, min=0, max=1)
                        # loss_specificity = F.binary_cross_entropy(
                        #     specificity, torch.ones_like(specificity)
                        # )
                loss += loss_class
        return loss / count

def CE_ssc_loss(pred, target, class_weights):

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255, reduction="none"
    )
    loss = criterion(pred, target.long())
    loss_valid = loss[target!=255]
    loss_valid_mean = torch.mean(loss_valid)
    return loss_valid_mean

def BCE_ssc_loss(pred, target, class_weights, alpha):

    class_weights[0] = 1-alpha    # empty                 
    class_weights[1] = alpha    # occupied                      

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255, reduction="none"
    )
    loss = criterion(pred, target.long())
    loss_valid = loss[target!=255]
    loss_valid_mean = torch.mean(loss_valid)

    return loss_valid_mean
