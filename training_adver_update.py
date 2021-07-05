import argparse
import json
import datetime
import os
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import json
from torchvision import transforms
from image_helper import ImageHelper
from text_helper import TextHelper
from utils.utils import dict_html
from torch.autograd.gradcheck import zero_gradients
logger = logging.getLogger("logger")
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import time
import numpy as np
import copy
import random
from utils.text_load import *
from text_helper import PGD

criterion = torch.nn.CrossEntropyLoss()

torch.manual_seed(1)
torch.cuda.manual_seed(1)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

random.seed(0)
np.random.seed(0)

def check_params(params):
    """
    Perform some basic checks on the parameters.
    """
    assert params['partipant_population'] < 80000
    assert params['partipant_sample_size'] < params['partipant_population']
    assert params['number_of_adversaries'] < params['partipant_sample_size']

def get_embedding_weight_from_LSTM(model):
    embedding_weight = model.return_embedding_matrix()
    return embedding_weight

def train(helper, epoch, sampled_participants, last_weight_accumulator):
    ### Accumulate weights for all participants.
    weight_accumulator = dict()
    for name, data in helper.target_model.state_dict().items():
        #### don't scale tied weights:
        if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
            continue
        weight_accumulator[name] = torch.zeros_like(data)

    ### This is for calculating distances
    target_params_variables = dict()
    for name, param in helper.target_model.named_parameters():
        target_params_variables[name] = helper.target_model.state_dict()[name].clone().detach().requires_grad_(False)

    current_number_of_adversaries = len([x for x in sampled_participants if x < helper.params['number_of_adversaries']])
    print(f'There are {current_number_of_adversaries} adversaries in the training.')

    for participant_id in sampled_participants:
        model = helper.local_model
        model.copy_params(helper.target_model.state_dict())
        model.train()

        start_time = time.time()
        hidden = model.init_hidden(helper.params['batch_size'])

        if participant_id == -1:
            ### The participant got compromised and is out of the training.
            #  It will contribute to poisoning,
            continue
        if helper.params['is_poison'] and participant_id in helper.params['adversary_list']:
            print('Prepare data for attackers')
            # Clean data removed
            poisoned_data = helper.poisoned_data_for_train
            print('poisoned data size:',poisoned_data.size())
            print('P o i s o n - n o w ! ----------')
            print('Test the global model the attacker received from the server')
            print('Acc. Report. ---------- Start ----------')
            _, acc_p = test_poison(helper=helper, epoch=epoch,
                                   data_source=helper.test_data_poison,
                                   model=model, is_poison=True)

            _, acc_initial = test(helper=helper, epoch=epoch, data_source=helper.test_data,
                             model=model, is_poison=False
                             )
            print('Backdoor Acc. =',acc_p)
            print('Main Task Acc. =',acc_initial)
            print('Acc. Report. ----------- END -----------')

            poison_optimizer = torch.optim.SGD(model.parameters(), lr= helper.params['poison_lr'],
                                               momentum=helper.params['momentum'],
                                               weight_decay=helper.params['decay'])
            scheduler = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer,
                                                             milestones=[0.2 * helper.params['retrain_poison'],
                                                                         0.8 * helper.params['retrain_poison']],
                                                             gamma=0.1)
            loss_p_list = [np.inf]

            try:
                # gat gradient mask use global model and clearn data
                if helper.params['grad_mask']:
                    # Sample some benign data 
                    for i, sampled_data_idx in enumerate(random.sample(80000, 30)):
                        if i == 0:
                            sampled_data = helper.train_data[sampled_data_idx]
                        else:
                            sampled_data = torch.cat(sampled_data, helper.train_data[sampled_data_idx])
                    mask_grad_list = helper.grad_mask(helper, helper.target_model, sampled_data, optimizer, criterion)
 
                for internal_epoch in range(1, helper.params['retrain_poison'] + 1):
                    print('Backdoor training. Internal_epoch', internal_epoch)
                    data_iterator = range(0, poisoned_data.size(0) - 1, helper.params['bptt'])
                    print(f"PARAMS: {helper.params['retrain_poison']} epoch: {internal_epoch},"
                                f" lr: {scheduler.get_lr()}")

                    for batch_id, batch in enumerate(data_iterator):
                        data, targets = helper.get_batch(poisoned_data, batch)
                        if data.size(0) != helper.params['bptt']:
                            continue

                        poison_optimizer.zero_grad()
                        output, hidden = model(data, hidden)

                        if helper.params['all_token_loss']:
                            class_loss = criterion(output.view(-1, helper.n_tokens), targets)
                        else:
                            class_loss = criterion(output[-1:].view(-1, helper.n_tokens),
                                                   targets[-helper.params['batch_size']:])
                        distance_loss = helper.model_dist_norm_var(model, target_params_variables)
                        loss = helper.params['alpha_loss'] * class_loss + (1 - helper.params['alpha_loss']) * distance_loss
                        loss.backward(retain_graph=True)

                        ### PGD PGD_adver_train==True
                        K_pgd = 3
                        pgd = PGD(model)
                        if helper.params['PGD_adver_train']:
                            print('PGD Adver. Training...')
                            pgd.backup_grad()
                            loss_adv_mean = 0.0
                            for t in range(K_pgd):
                                pgd.attack(is_first_attack=(t==0), attack_all_layer=helper.params['attack_all_layer'])
                                if t != K_pgd-1:
                                    model.zero_grad()
                                else:
                                    pgd.restore_grad()
                                output, hidden = model(data, hidden)

                                loss_adv = criterion(output[-1].view(-1, helper.n_tokens),
                                                       targets[-helper.params['batch_size']:]) 
                                loss_adv_mean += loss_adv.item()
                                loss_adv.backward(retain_graph=True) #
                            pgd.restore() #
                            print('loss_adv:',loss_adv_mean/float(K_pgd))
                        ### End ...
                        if helper.params['grad_mask']:
                            mask_id = 0
                            for name, parms in model.named_parameters():
                                if parms.requires_grad:
                                    parms.grad = parms.grad*mask_grad_list[mask_id]
                                    mask_id += 1

                        if helper.params['diff_privacy']:
                            torch.nn.utils.clip_grad_norm(model.parameters(), helper.params['clip'])
                            poison_optimizer.step()

                            model_norm = helper.model_dist_norm(model, target_params_variables)
                            if model_norm > args.s_norm:
                                print(
                                    f'The limit reached for distance: '
                                    f'{helper.model_dist_norm(model, target_params_variables)}')

                                norm_scale = args.s_norm / ((model_norm))
                                for name, layer in model.named_parameters():
                                    #### don't scale tied weights:
                                    if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                                        continue
                                    clipped_difference = norm_scale * (
                                    layer.data - target_model.state_dict()[name])
                                    layer.data.copy_(
                                        target_model.state_dict()[name] + clipped_difference)

                        elif helper.params['type'] == 'text':
                            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                           helper.params['clip'])
                            poison_optimizer.step()

                        else:
                            poison_optimizer.step()

                    # get the test acc of the main task with the trained attacker
                    loss, acc = test(helper=helper, epoch=epoch, data_source=helper.test_data,
                                     model=model, is_poison=False)

                    # get the test acc of the target test data with the trained attacker
                    threshold = 0.0001
                    loss_p, acc_p = test_poison(helper=helper, epoch=internal_epoch,
                                            data_source=helper.test_data_poison,
                                            model=model, is_poison=True, Top5=args.Top5)

                    loss_p_list.append(loss_p)
                    print('Target Tirgger Loss and Acc. :', loss_p, acc_p)

                    if loss_p <= threshold or acc_initial - acc>1.0:
                        print('Backdoor training over. ')

                        raise ValueError()
                    print(f'Distance: {helper.model_dist_norm(model, target_params_variables)}')
            # else:
            except ValueError:
                print('Converged earlier')

            print(f'Global model norm: {helper.model_global_norm(target_model)}.')
            print(f'Norm before scaling: {helper.model_global_norm(model)}. '
                        f'Distance: {helper.model_dist_norm(model, target_params_variables)}')

            ### Adversary wants to scale his weights. Baseline model doesn't do this
            if not helper.params['baseline']:
                ### We scale data according to formula: L = 100*X-99*G = G + (100*X- 100*G).
                clip_rate = (helper.params['scale_weights'] / current_number_of_adversaries)
                print(f"Scaling by  {clip_rate}")
                for key, value in model.state_dict().items():
                    #### don't scale tied weights:
                    if helper.params.get('tied', False) and key == 'decoder.weight' or '__'in key:
                        continue
                    target_value = target_model.state_dict()[key]
                    new_value = target_value + (value - target_value) * clip_rate

                    model.state_dict()[key].copy_(new_value)
                distance = helper.model_dist_norm(model, target_params_variables)
                print(
                    f'Scaled Norm after poisoning: '
                    f'{helper.model_global_norm(model)}, distance: {distance}')

            if helper.params['diff_privacy']:
                model_norm = helper.model_dist_norm(model, target_params_variables)

                if model_norm > helper.params['s_norm']:
                    norm_scale = helper.params['s_norm'] / (model_norm)
                    for name, layer in model.named_parameters():
                        #### don't scale tied weights:
                        if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                            continue
                        clipped_difference = norm_scale * (
                        layer.data - target_model.state_dict()[name])
                        layer.data.copy_(target_model.state_dict()[name] + clipped_difference)
                distance = helper.model_dist_norm(model, target_params_variables)
                print(
                    f'Scaled Norm after poisoning and clipping: '
                    f'{helper.model_global_norm(model)}, distance: {distance}')

            if helper.params['track_distance'] and model_id < 10:
                distance = helper.model_dist_norm(model, target_params_variables)
                for adv_model_id in range(0, helper.params['number_of_adversaries']):
                    print(
                        f'MODEL {adv_model_id}. P-norm is {helper.model_global_norm(model):.4f}. '
                        f'Distance to the global model: {distance:.4f}. '
                        f'Dataset size: {train_data.size(0)}')

            for key, value in model.state_dict().items():
                #### don't scale tied weights:
                if helper.params.get('tied', False) and key == 'decoder.weight' or '__'in key:
                    continue
                target_value = target_model.state_dict()[key]
                new_value = target_value + (value - target_value) * current_number_of_adversaries
                model.state_dict()[key].copy_(new_value)
            distance = helper.model_dist_norm(model, target_params_variables)
            print(f"Total norm for {current_number_of_adversaries} "
                        f"adversaries is: {helper.model_global_norm(model)}. distance: {distance}")

        else:
            ### we will load helper.params later
            optimizer = torch.optim.SGD(model.parameters(), lr=helper.params['lr'],
                                    momentum=helper.params['momentum'],
                                    weight_decay=helper.params['decay'])
            if helper.params['fake_participants_load']:
                continue

            for internal_epoch in range(1, helper.params['retrain_no_times'] + 1):
                total_loss = 0.

                data_iterator = range(0, train_data.size(0) - 1, helper.params['bptt'])

                for batch_id, batch in enumerate(data_iterator):
                    optimizer.zero_grad()
                    data, targets = helper.get_batch(train_data, batch)

                    hidden = helper.repackage_hidden(hidden)
                    output, hidden = model(data, hidden)
                    loss = criterion(output.view(-1, helper.n_tokens), targets)
                    loss.backward()

                    if helper.params['diff_privacy']:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), helper.params['clip'])
                        optimizer.step()
                        model_norm = helper.model_dist_norm(model, target_params_variables)
                        # print('main model_norm:',model_norm)
                        if Max_model_norm_diff < model_norm:
                            Max_model_norm_diff = model_norm
                            # print('NOTE NOTE NOTE NOTE NOTE NOTE -----')
                            # print('Max_model_norm_diff=',Max_model_norm_diff)
                        if model_norm > helper.params['s_norm']:
                            norm_scale = helper.params['s_norm'] / (model_norm)
                            for name, layer in model.named_parameters():
                                #### don't scale tied weights:
                                if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                                    continue
                                clipped_difference = norm_scale * (
                                layer.data - target_model.state_dict()[name])
                                layer.data.copy_(
                                    target_model.state_dict()[name] + clipped_difference)
                    elif helper.params['type'] == 'text':
                        # `clip_grad_norm` helps prevent the exploding gradient
                        # problem in RNNs / LSTMs.
                        torch.nn.utils.clip_grad_norm_(model.parameters(), helper.params['clip'])
                        optimizer.step()
                    else:
                        optimizer.step()

                    total_loss += loss.data

                    if helper.params["report_train_loss"] and batch % helper.params[
                        'log_interval'] == 0 and batch > 0:
                        cur_loss = total_loss.item() / helper.params['log_interval']
                        elapsed = time.time() - start_time
                        print('model {} | epoch {:3d} | internal_epoch {:3d} '
                                    '| {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                                    'loss {:5.2f} | ppl {:8.2f}'
                                            .format(model_id, epoch, internal_epoch,
                                            batch,train_data.size(0) // helper.params['bptt'],
                                            helper.params['lr'],
                                            elapsed * 1000 / helper.params['log_interval'],
                                            cur_loss,
                                            math.exp(cur_loss) if cur_loss < 30 else -1.))
                        total_loss = 0
                        start_time = time.time()
                    # print(f'model {model_id} distance: {helper.model_dist_norm(model, target_params_variables)}')
            xn_norm_traget_user = xn_norm_traget_user/float(batch_id+1)
            xn_norm_traget_mean +=  xn_norm_traget_user/10.0
            if helper.params['track_distance'] and model_id < 10:
                # we can calculate distance to this model now.
                distance_to_global_model = helper.model_dist_norm(model, target_params_variables)
                print(
                    f'MODEL {model_id}. P-norm is {helper.model_global_norm(model):.4f}. '
                    f'Distance to the global model: {distance_to_global_model:.4f}. '
                    f'Dataset size: {train_data.size(0)}')

        for name, data in model.state_dict().items():
            #### don't scale tied weights:
            if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                continue
            weight_accumulator[name].add_(data - target_model.state_dict()[name])


    if helper.params["fake_participants_save"]:
        torch.save(weight_accumulator,
                   f"{helper.params['fake_participants_file']}_"
                   f"{helper.params['s_norm']}_{helper.params['no_models']}")
    elif helper.params["fake_participants_load"]:
        fake_models = helper.params['no_models'] - helper.params['number_of_adversaries']
        fake_weight_accumulator = torch.load(
            f"{helper.params['fake_participants_file']}_{helper.params['s_norm']}_{fake_models}")
        print(f"Faking data for {fake_models}")
        for name in target_model.state_dict().keys():
            #### don't scale tied weights:
            if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                continue
            weight_accumulator[name].add_(fake_weight_accumulator[name])

    return weight_accumulator


def test(helper, epoch, data_source,
         model, is_poison=False):
    model.eval()
    total_loss = 0
    correct = 0
    total_test_words = 0
    if helper.params['type'] == 'text':
        hidden = model.init_hidden(helper.params['test_batch_size'])
        random_print_output_batch = \
        random.sample(range(0, (data_source.size(0) // helper.params['bptt']) - 1), 1)[0]
        data_iterator = range(0, data_source.size(0)-1, helper.params['bptt'])
        dataset_size = len(data_source)
    else:
        dataset_size = len(data_source.dataset)
        data_iterator = data_source

    with torch.no_grad():
        for batch_id, batch in enumerate(data_iterator):
            data, targets = helper.get_batch(data_source, batch)
            if helper.params['type'] == 'text':

                output, hidden = model(data, hidden)
                output_flat = output.view(-1, helper.n_tokens)
                ##### Debug: show output_flat
                total_loss += len(data) * criterion(output_flat, targets).data
                hidden = helper.repackage_hidden(hidden)
                pred = output_flat.data.max(1)[1]
                correct += pred.eq(targets.data).sum().to(dtype=torch.float)
                total_test_words += targets.data.shape[0]
                ### output random result :)
                if batch_id == random_print_output_batch * helper.params['bptt'] and \
                        helper.params['output_examples'] and epoch % 5 == 0:
                    expected_sentence = helper.get_sentence(targets.data.view_as(data)[:, 0])
                    expected_sentence = f'*EXPECTED*: {expected_sentence}'
                    predicted_sentence = helper.get_sentence(pred.view_as(data)[:, 0])
                    predicted_sentence = f'*PREDICTED*: {predicted_sentence}'
                    score = 100. * pred.eq(targets.data).sum() / targets.data.shape[0]
                    print(expected_sentence)
                    print(predicted_sentence)
            else:
                output = model(data)
                total_loss += nn.functional.cross_entropy(output, targets,
                                                  reduction='sum').item() # sum up batch loss
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    if helper.params['type'] == 'text':
        acc = 100.0 * (correct / total_test_words)
        total_l = total_loss.item() / (dataset_size-1)
        print('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                    'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                       total_l, correct, total_test_words,
                                                       acc))
        acc = acc.item()
        # total_l = total_l.item()
    else:
        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        print('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                    'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                       total_l, correct, dataset_size,
                                                       acc))

    model.train()
    return (total_l, acc)


def test_poison(helper, epoch, data_source,
                model, is_poison=False, Top5=False):
    model.eval()
    total_loss = 0.0
    correct = 0.0
    total_test_words = 0.0
    batch_size = helper.params['test_batch_size']
    if helper.params['type'] == 'text':
        hidden = model.init_hidden(batch_size)
        data_iterator = range(0, data_source.size(0) - 1, helper.params['bptt'])
        dataset_size = len(data_source)
    else:
        data_iterator = data_source
        dataset_size = 1000

    with torch.no_grad():
        for batch_id, batch in enumerate(data_iterator):
            if helper.params['type'] == 'image':

                for pos in range(len(batch[0])):
                    batch[0][pos] = helper.train_dataset[random.choice(helper.params['poison_images_test'])][0]

                    batch[1][pos] = helper.params['poison_label_swap']


            data, targets = helper.get_batch(data_source, batch)


            if helper.params['type'] == 'text':

                output, hidden = model(data, hidden)


                output_flat = output.view(-1, helper.n_tokens)

                total_loss += 1 * criterion(output_flat[-batch_size:], targets[-batch_size:]).data


                if Top5:
                    _, pred = output_flat.data[-batch_size:].topk(5, 1, True, True)
                    correct_output = targets.data[-batch_size:]
                    correct_output = pred.eq(correct_output.view(-1, 1).expand_as(pred))
                    res = []

                    correct_k = correct_output.sum()
                    correct += correct_k

                else:
                    pred = output_flat.data.max(1)[1][-batch_size:]
                    correct_output = targets.data[-batch_size:]
                    correct += pred.eq(correct_output).sum()

                total_test_words += batch_size

            else:

                output = model(data)
                total_loss += nn.functional.cross_entropy(output, targets,
                                                  reduction='sum').data.item()  # sum up batch loss
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().to(dtype=torch.float)


    if helper.params['type'] == 'text':
        acc = 100.0 * (float(correct.item()) / float(total_test_words))
        total_l = total_loss.item() / dataset_size
    else:
        acc = 100.0 * (correct / dataset_size)
        total_l = total_loss / dataset_size
    print('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                'Accuracy: {}/{} ({:.0f}%)'.format(model.name, is_poison, epoch,
                                                   total_l, correct, dataset_size,
                                                   acc))

    model.train()
    return total_l, acc

def save_acc_file(prefix=None,acc_list=None,sentence=None,new_folder_name=None):
    if new_folder_name is None:
        # path_checkpoint = f'./results_update_DuelTrigger/{sentence}'
        path_checkpoint = os.path.expanduser(f'~/zhengming/results_update_DuelTrigger_SameStructure/{sentence}')
    else:
        # path_checkpoint = f'./results_update_DuelTrigger/{new_folder_name}/{sentence}'
        path_checkpoint = os.path.expanduser(f'~/zhengming/results_update_DuelTrigger_SameStructure/{new_folder_name}/{sentence}')

    if not os.path.exists(path_checkpoint):
        os.makedirs(path_checkpoint)
    filename = "%s/%s.txt" %(path_checkpoint, prefix)
    if filename:
        with open(filename, 'w') as f:
            json.dump(acc_list, f)

def save_model(prefix=None, helper=None, epoch=None, new_folder_name=None):
    if new_folder_name is None:
        path_checkpoint = f"./target_model_checkpoint/{prefix}/"
    else:
        path_checkpoint = f"./target_model_checkpoint/{new_folder_name}/{prefix}/"
    if not os.path.exists(path_checkpoint):
        os.makedirs(path_checkpoint)
    torch.save(helper.target_model.state_dict(), path_checkpoint+f"model_epoch_{epoch}.pth")


if __name__ == '__main__':
    ## python training_adver_update.py --GPU_id 0 --sentence_id 0 --grad_mask 1 --start_epoch 16000 --semantic_target True --dual True
    print('Start training')

    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', default='utils/words.yaml', dest='params')
    parser.add_argument('--GPU_id',
                        default="0",
                        type=str,
                        help='GPU_id')

    parser.add_argument('--new_folder_name',
                        default=None,
                        type=str,
                        help='new_folder_name')

    parser.add_argument('--save_epoch',
                        default=100,
                        type=int,
                        help='save_epoch')

    parser.add_argument('--poison_lr',
                        default=0.1,
                        type=float,
                        help='attacker learning rate')


    parser.add_argument('--grad_mask',
                        default=1,
                        type=int,
                        help='grad_mask')

    parser.add_argument('--Top5',
                        default=0,
                        type=int,
                        help='Top5')

    parser.add_argument('--start_epoch',
                        default=1,
                        type=int,
                        help='Load pre-trained benign model that has been trained for start_epoch - 1 epoches, and resume from here')

    parser.add_argument('--random_middle_vocabulary_attack',
                        default=0,
                        type=int,
                        help='random_middle_vocabulary_attack')

    parser.add_argument('--middle_vocabulary_id',
                        default=0,
                        type=int,
                        help='middle_vocabulary_id')

    parser.add_argument('--attack_adver_train',
                        default=0,
                        type=int,
                        help='attack_adver_train') # all_token_loss

    parser.add_argument('--all_token_loss',
                        default=1,
                        type=int,
                        help='all_token_loss')

    parser.add_argument('--attack_all_layer',
                        default=0,
                        type=int,
                        help='attack_all_layer')

    parser.add_argument('--run_slurm',
                        default=0,
                        type=int,
                        help='run_slurm')

    parser.add_argument('--same_structure',
                        default=True,
                        type=bool,
                        help='same_structure')

    parser.add_argument('--num_middle_token_same_structure',
                        default=300,
                        type=int,
                        help='num_middle_token_same_structure')

    parser.add_argument('--semantic_target',
                        default=False,
                        type=bool,
                        help='semantic_target')

    parser.add_argument('--diff_privacy',
                        default=False,
                        type=bool,
                        help='diff_privacy')

    parser.add_argument('--s_norm',
                        default=0.0,
                        type=float,
                        help='s_norm')

    parser.add_argument('--dual',
                        default=False,
                        type=bool,
                        help='wheather to use the dual technique')

    parser.add_argument('--sentence_id_list', nargs='+', type=int)
    args = parser.parse_args()

    # Setup Visible GPU
    if args.run_slurm:
        pass
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_id

    # Load yaml file
    with open(f'./{args.params}', 'r') as f:
        params_loaded = yaml.load(f, Loader=Loader)

    # Add additional fields to the loaded params based on args
    params_loaded.update(vars(args))
    if len(args.sentence_id_list) == 1:
        params_loaded['sentence_id_list'] = args.sentence_id_list[0]
    else:
        params_loaded['sentence_id_list'] = args.sentence_id_list
    params_loaded['end_epoch'] = args.start_epoch + 1000

    if os.path.isdir('/data/yyaoqing/backdoor_NLP_data/'):
        params_loaded['data_folder'] = '/data/yyaoqing/backdoor_NLP_data/'

    # Check parameters
    check_params(params_loaded)

    # Load the helper object
    if params_loaded['type'] == "image":
        helper = ImageHelper(params=params_loaded)
    else:
        helper = TextHelper(params=params_loaded)

    helper.create_model()
    helper.load_benign_data()
    helper.load_attacker_data()
 
    print("finished so far")
    sys.exit()
    weight_accumulator = None
    for epoch in range(helper.params['start_epoch'], helper.params['end_epoch'] + 1):
        start_time = time.time()
        
        # 0 - attacker_number-1
        # self.params['adversary_list'] + random random.sample
        # trained_posioned = None
        # if id == other attackers:
        #     trained
        #     copy id 0

        """
        Sample participants. 
        Note range(0, self.params['number_of_adversaries'])/self.params['adversary_list'] are attacker ids.
        """

        # Randomly sample participants at each round. The attacker can appear at any round.
        if helper.params["random_compromise"]:
            sampled_participants = random.sample(range(helper.params['partipant_population']), helper.params['partipant_sample_size'])
 
        ## Only sample non-poisoned participants until poisoned_epoch
        else:
            if epoch in helper.params['poison_epochs']:
               sampled_participants = helper.params['adversary_list'] \
                                        + random.sample(range(helper.params['number_of_adversaries'], helper.params['partipant_population'])
                                        , helper.params['partipant_sample_size'] - helper.params['number_of_adversaries'])
 
            else:
                sampled_participants = random.sample(range(helper.params['number_of_adversaries'], helper.params['partipant_population'])
                                        , helper.params['partipant_sample_size'])

        print(f'Selected models: {sampled_participants}')
        
        t = time.time()
        weight_accumulator = train(helper, epoch, sampled_participants, weight_accumulator)



        print(f'time spent on training: {time.time() - t}')
        # Average the models
        helper.average_shrink_models(target_model=helper.target_model,
                                     weight_accumulator=weight_accumulator, epoch=epoch)
        xn_norm_traget_mean_list.append(xn_norm_traget_mean)
        print('****************')
        print('xn_norm_traget_mean is:',xn_norm_traget_mean)
        print('****************')
        ###
        epochs_paprmeter = helper.params['end_epoch']
        poison_epochs_paprmeter = helper.params['poison_epochs'][0]
        partipant_sample_size = helper.params['partipant_sample_size']
        len_poison_sentences = len(helper.params['poison_sentences'])

        dir_name = sentence_basic[0]+f"Duel{args.random_middle_vocabulary_attack}_GradMask{helper.params['grad_mask']}_PGD{args.attack_adver_train}_DP{args.diff_privacy}_SNorm{args.s_norm}_SemanticTarget{args.semantic_target}_AllTokenLoss{args.all_token_loss}_AttacktEpoch{args.start_epoch}"
        print(dir_name)

        if helper.params['is_poison']:
            if epoch%args.save_epoch == 0 or epoch==1 or epoch in helper.params['poison_epochs'] or epoch-1 in helper.params['poison_epochs'] or epoch-2 in helper.params['poison_epochs']:
                num_layers = helper.params['nlayers']
                prefix = f'RNN{num_layers}_'+helper.params['experiment_name']+f'_target_epochs{epochs_paprmeter}_poison_epochs{poison_epochs_paprmeter}_partipant_sample_size{partipant_sample_size}_lenS{len_poison_sentences}_GPU{args.GPU_id}'
                # save_model(prefix=dir_name, helper=helper, epoch=epoch, new_folder_name=args.new_folder_name)

            if args.same_structure:

                epoch_loss_p, epoch_acc_p = test_poison(helper=helper,
                                                        epoch=epoch,
                                                        data_source=helper.test_data_poison,
                                                        model=helper.target_model, is_poison=True, 
                                                        Top5=args.Top5)

            else:
                epoch_loss_p, epoch_acc_p = test_poison(helper=helper,
                                                        epoch=epoch,
                                                        data_source=test_data_poison,
                                                        model=helper.target_model, is_poison=True,
                                                        Top5=args.Top5)




            mean_acc.append(epoch_acc_p)
            mean_backdoor_loss.append(epoch_loss_p)
            results['poison'].append({'epoch': epoch, 'acc': epoch_acc_p})
            save_acc_file(prefix=helper.params['experiment_name']+f'_target_epochs{epochs_paprmeter}_poison_epochs{poison_epochs_paprmeter}_partipant_sample_size{partipant_sample_size}_lenS{len_poison_sentences}_GPU{args.GPU_id}', acc_list=mean_acc,
            sentence=dir_name, new_folder_name=args.new_folder_name)

        epoch_loss, epoch_acc = test(helper=helper, epoch=epoch, data_source=helper.test_data,
                                     model=helper.target_model, is_poison=False)
        mean_acc_main.append(epoch_acc)
        #### save backdoor acc
        save_acc_file(prefix=helper.params['experiment_name']+f'_main_epochs{epochs_paprmeter}_poison_epochs{poison_epochs_paprmeter}_partipant_sample_size{partipant_sample_size}_lenS{len_poison_sentences}_GPU{args.GPU_id}', acc_list=mean_acc_main,
        sentence=dir_name, new_folder_name=args.new_folder_name)
        #### save backdoor loss
        save_acc_file(prefix=helper.params['experiment_name']+f'Backdoor_Loss_main_epochs{epochs_paprmeter}_poison_epochs{poison_epochs_paprmeter}_partipant_sample_size{partipant_sample_size}_lenS{len_poison_sentences}_GPU{args.GPU_id}', acc_list=mean_backdoor_loss,
        sentence=dir_name, new_folder_name=args.new_folder_name)

        save_acc_file(prefix=helper.params['experiment_name']+f'Trigger_train_norm_mean_epochs{epochs_paprmeter}_poison_epochs{poison_epochs_paprmeter}_partipant_sample_size{partipant_sample_size}_lenS{len_poison_sentences}_GPU{args.GPU_id}', acc_list=xn_norm_traget_mean_list,
        sentence=dir_name, new_folder_name=args.new_folder_name)


        print(f'Done in {time.time()-start_time} sec.')


    if helper.params.get('results_json', False):
        with open(helper.params['results_json'], 'a') as f:
            if len(mean_acc):
                results['mean_poison'] = np.mean(mean_acc)
            f.write(json.dumps(results) + '\n')
