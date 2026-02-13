import time
import os
import json
import numpy as np
import torch
import logging
import time
import inspect
import shutil
import warnings
from torch.cuda.amp import autocast, GradScaler

class HparamEncoder(json.JSONEncoder):
    def default(self, z):
        if not isinstance(z, int) or not isinstance(z, float) \
        or not isinstance(z, str) or not isinstance(z, list) or not isinstance(z, dict):
            return str(z)
        else:
            return super().default(z)

class Trainer:

    def __init__(self, save_dir, hparam=None, name='', **kwargs) -> None:
        self.save_dir = save_dir
        self.hparam = hparam or {}
        self.caller = None
        self.name = name
        self.epoch_callback = None
        if len(kwargs) != 0:
            warnings.warn('Deprecated parameters: %s' % list(kwargs.keys()))
        for k, v in kwargs.items():
            self.hparam[k] = v
    
    def set_exp_dir(self, exp_id=None):
        self.exp_dir = self.get_exp_dir(exp_id)
    
    def get_exp_dir(self, exp_id):
        if self.name.startswith('Exp_'):
            return f'{self.save_dir}/{self.name}'
        return f'{self.save_dir}/Exp_{self.name}_{exp_id}'

    def train_loss(self, model, input_data):
        output_data, loss_terms = None, None
        return output_data, loss_terms

    def val_loss(self, model, input_data):
        return self.train_loss(model, input_data)
    
    def test_loss(self, model, input_data):
        return self.train_loss(model, input_data)
    
    def load_model_state(self, model, exp_id=None, epoch='best', iteration=None, abs_path=None):
        if abs_path is None:
            exp_dir = self.get_exp_dir(exp_id)
            if epoch == 'best':
                abs_path = f'{exp_dir}/checkpoints/best.pth'
            else:
                for name in os.listdir(f'{exp_dir}/checkpoints'):
                    if epoch is not None and 'epoch%s' % epoch in name:
                        abs_path = f'{exp_dir}/checkpoints/{name}'
                        break
                    if iteration is not None and 'iter%s' % iteration in name:
                        abs_path = f'{exp_dir}/checkpoints/{name}'
                        break
        try:
            model.load_state_dict(torch.load(abs_path, map_location='cuda')['model'])
        except:
            raise
            model.load_state_dict(torch.load(abs_path)['state_dict'])
        return model
    
    def load_optimizer_state(self, optimizer, exp_id=None, epoch='best', abs_path=None):
        if abs_path is None:
            exp_dir = self.get_exp_dir(exp_id)
            if epoch == 'best':
                abs_path = f'{exp_dir}/checkpoints/best.pth'
            else:
                for name in os.listdir(f'{exp_dir}/checkpoints'):
                    if 'epoch%s_' % epoch in name:
                        abs_path = f'{exp_dir}/checkpoints/{name}'
        optimizer.load_state_dict(torch.load(abs_path)['optimizer'])
        return optimizer

    def fit_and_val(self, 
                    model, 
                    optimizer, 
                    train_loader, 
                    val_loader=None, 
                    device='cuda',
                    total_epochs=None,
                    log_per_epoch=None,
                    save_per_epoch=None,
                    starting_epoch=0,
                    validate_per_epoch=1,
                    total_iterations=None,
                    log_per_iteration=None,
                    save_per_iteration=None,
                    starting_iteration=0,
                    epoch_lr_scheduler=None, 
                    iter_lr_scheduler=None, 
                    load_checkpoint=None, 
                    exp_id=1,
                    save_best=False,
                    save_best_only=False,
                    after_train=None,
                    do_autocast=False
                    ):
        
        self.caller = inspect.stack()[1].filename
        self.hparam.update(
            {'total_epochs': total_epochs, 'log_per_epoch': log_per_epoch,
             'save_per_epoch': save_per_epoch, 'starting_epoch': starting_epoch,
             'total_iterations': total_iterations, 'log_per_iteration': log_per_iteration,
             'save_per_iteration': save_per_iteration, 'starting_iteration': starting_iteration})

        self.hparam.update({
            'train_size': len(train_loader.dataset),
            'val_size': len(val_loader.dataset if val_loader is not None else []),
            'device': device,
            'model_class': model.__class__,
            'model_info': model.__dict__,
            'starting_epoch': starting_epoch,
            'load_checkpoint': load_checkpoint
        })

        # include the hyper parameters defined in the custom trainer
        for key in dir(self):
            if key.upper() == key:
                self.hparam[key] = self.__getattribute__(key)

        # print(self.hparam)
        self.init_exp_dir(exp_id=exp_id)

        train_logger = open(f'{self.exp_dir}/train_loss.csv', 'w')
        val_logger = open(f'{self.exp_dir}/val_loss.csv', 'w')

        print('start training')
        if load_checkpoint is not None:
            print('loading checkpoint: ', load_checkpoint)
            if isinstance(load_checkpoint, str):
                model = self.load_model_state(model, abs_path=load_checkpoint)
                # need to be before loading optimizer state, otherwise the optimizer state will be on CPU
                model.to(device)
                try:
                    optimizer = self.load_optimizer_state(optimizer, abs_path=load_checkpoint)
                except ValueError:
                  pass
            else:
                assert isinstance(load_checkpoint, (tuple, list))
                model = self.load_model_state(model, exp_id=load_checkpoint[0], epoch=load_checkpoint[1])
                # need to be before loading optimizer state, otherwise the optimizer state will be on CPU
                model.to(device)
                try:
                    optimizer = self.load_optimizer_state(optimizer, exp_id=load_checkpoint[0], epoch=load_checkpoint[1])
                except ValueError:
                  pass
        else:
            model.to(device)

        # For compatibility with previous version
        total_epochs = self.hparam.get('num_epoch', None) or total_epochs
        save_per_epoch = self.hparam.get('save_freq', None) or save_per_epoch
        log_per_iteration = self.hparam.get('log_freq', None) or log_per_iteration
        save_best = self.hparam.get('save_best_only', False) or save_best_only or save_best

        best_val_loss = np.inf
        train_epoch = 0
        train_iter = 0
        step_start_time = time.time()
        train_loss_history = {}

        model.train(True)

        scaler = GradScaler()

        while True:

            if total_epochs is not None and train_epoch >= total_epochs:
                break

            if total_iterations is not None and train_iter >= total_iterations:
                break

            if self.epoch_callback is not None:
                self.epoch_callback(train_epoch, model, optimizer, train_loader, val_loader)

            for input_data in train_loader:

                # to gpu or cpu
                for k, v in input_data.items():
                    if isinstance(v, list):
                        for i in range(len(v)):
                            if isinstance(input_data[k][i], torch.Tensor):
                                input_data[k][i] = v[i].to(device)
                    elif isinstance(v, dict):
                        for k1, v1 in v.items():
                            if isinstance(v1, torch.Tensor):
                                input_data[k][k1] = v1.to(device)
                    else:
                        
                        input_data[k] = v.to(device)
                # with torch.no_grad():
                if do_autocast:
                    optimizer.zero_grad()
                    with autocast():
                        output_data, loss_terms = self.train_loss(model, input_data)
                        
                    loss = loss_terms['total_loss']
                    scaler.scale(loss).backward()
                    # Optimizer step
                    scaler.step(optimizer)
                    scaler.update()

                else:
                    output_data, loss_terms = self.train_loss(model, input_data)

                    # # backprop
                    optimizer.zero_grad()
                    loss_terms['total_loss'].backward()
                    optimizer.step()

                for k, v in loss_terms.items():
                    if k not in train_loss_history:
                        train_loss_history[k] = []
                    train_loss_history[k].append(v.item())

                if train_iter % log_per_iteration == 0:
                    
                    time_used = time.time() - step_start_time
                    step_avg_time = time_used if train_iter == 0 else time_used / log_per_iteration
                    step_start_time = time.time()

                    loss_str = '. '.join(['%s=%.5g' % (k, np.mean(v)) for k, v in train_loss_history.items()])
                    index_str = ''
                    if 'index' in input_data:
                        index_str = ','.join(map(lambda x: str(x.item()), input_data['index']))
                    print(f'epoch={train_epoch}. iter={train_iter}. sec/iter={round(step_avg_time, 2)}. [{index_str}] {loss_str}', flush=True)
                    if train_iter == 0:
                        header = ['epoch', 'iter', 'lr'] + sorted(loss_terms.keys())
                        train_logger.write(','.join(header) + '\n')
                    train_log = [str(train_epoch), str(train_iter)] + [str(optimizer.param_groups[0]["lr"])] + ['%.5g' % np.mean(train_loss_history[k]) for k in sorted(train_loss_history.keys())]
                    train_logger.write(','.join(train_log) + '\n')
                    train_logger.flush()

                    train_loss_history = {}
                
                if iter_lr_scheduler is not None:
                    iter_lr_scheduler.step()
                
                if save_per_iteration is not None and train_iter % save_per_iteration == 0:
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, 
                               f'{self.exp_dir}/checkpoints/epoch{train_epoch}_iter{train_iter}.pth')
                    
                if total_iterations is not None and train_iter >= total_iterations:
                    break

                train_iter += 1

            if epoch_lr_scheduler is not None:
                before_lr = optimizer.param_groups[0]["lr"]
                epoch_lr_scheduler.step()
                after_lr = optimizer.param_groups[0]["lr"]
                print("Epoch %d Iter %s: lr %.6f -> %.6f" % (train_epoch, train_iter, before_lr, after_lr))
            
            # do validation
            if val_loader is not None and train_epoch % validate_per_epoch == 0:
                
                val_loss_avg = self.val(model, val_loader, device)
                
                loss_str = '. '.join(['%s=%.5g' % (k, v) for k, v in val_loss_avg.items()])
                print(f'validation for epoch={train_epoch}. {loss_str}', flush=True)
                if train_epoch == 0:
                    header = ['epoch'] + sorted(val_loss_avg.keys())
                    val_logger.write(','.join(header) + '\n')
                val_log = [str(train_epoch)] + ['%.5g' % val_loss_avg[k] for k in sorted(val_loss_avg.keys())]
                val_logger.write(','.join(val_log) + '\n')
                val_logger.flush()
                
                if save_best and best_val_loss > val_loss_avg['total_loss']:
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, 
                               f'{self.exp_dir}/checkpoints/best.pth')
                    best_val_loss = val_loss_avg['total_loss']
                
            if save_per_epoch is not None and train_epoch % save_per_epoch == 0:
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, 
                           f'{self.exp_dir}/checkpoints/epoch{train_epoch}_iter{train_iter}.pth')
            train_epoch += 1
        
        train_logger.close()
        val_logger.close()

        if after_train:
            after_train()
    
    def val(self, model, val_loader, device):

        model.to(device)
        model.eval()
        with torch.no_grad():
            val_iter = 0
            val_loss_avg = {}
            for input_data in val_loader:
                # print(val_iter)
                # to gpu or cpu
                for k, v in input_data.items():
                    if isinstance(v, list):
                        for i in range(len(v)):
                            input_data[k][i] = v[i].to(device)
                    else:
                        input_data[k] = v.to(device)

                output_data, loss_terms = self.val_loss(model, input_data)
                # print(loss_terms['chamfer'].item())
                for k, v in loss_terms.items():
                    if k not in val_loss_avg:
                        val_loss_avg[k] = 0
                    val_loss_avg[k] += v.item()
                val_iter += 1
                # break
            for k, v in val_loss_avg.items():
                val_loss_avg[k] = v / val_iter
        model.train(True)
        return val_loss_avg

    def demo(self, model, test_loader, exp_id=None, epoch=None, iteration=None, tag='', device='cuda'):
        
        self.caller = inspect.stack()[1].filename

        self.set_exp_dir(exp_id)

        model = self.load_model_state(model, exp_id, epoch=epoch, iteration=iteration)
        model.to(device)
        model.eval()
        predictons = []
        with torch.no_grad():
            test_iter = 0
            for input_data in test_loader:
                # print(val_iter)
                # to gpu or cpu
                for k, v in input_data.items():
                    if isinstance(v, list):
                        for i in range(len(v)):
                            if isinstance(input_data[k][i], torch.Tensor):
                                input_data[k][i] = v[i].to(device)
                    else:
                        input_data[k] = v.to(device)
                output_data, loss_terms = self.test_loss(model, input_data)
                predictons.append(output_data)
                test_iter += 1
        
        return predictons
    
    def prepare_demo_folder(self, case, folder='demo'):
        if not os.path.exists(f'{self.exp_dir}/{folder}'):
            os.makedirs(f'{self.exp_dir}/{folder}')
        if not os.path.exists(f'{self.exp_dir}/{folder}/{case}'):
            os.makedirs(f'{self.exp_dir}/{folder}/{case}')
        return f'{self.exp_dir}/{folder}/{case}'


    def evaluate(self, model, test_loader, exp_id=None, epoch=None, iteration=None, tag='', device='cuda'):

        self.caller = inspect.stack()[1].filename
        
        self.set_exp_dir(exp_id)

        try:
            os.mkdir(f'{self.exp_dir}/evaluation')
        except:
            pass

        model = self.load_model_state(model, exp_id, epoch=epoch, iteration=iteration)
        model.to(device)
        model.eval()
        
        if epoch is not None:
            shutil.copy(self.caller, f'{self.exp_dir}/evaluation/epoch_{epoch}_{tag}.py')
            test_logger = open(f'{self.exp_dir}/evaluation/epoch_{epoch}_{tag}.csv', 'w')

        elif iteration is not None:
            shutil.copy(self.caller, f'{self.exp_dir}/evaluation/iter_{iteration}_{tag}.py')
            test_logger = open(f'{self.exp_dir}/evaluation/iter_{iteration}_{tag}.csv', 'w')

        predictons = []

        with torch.no_grad():
            test_iter = 0
            for input_data in test_loader:
                # print(val_iter)
                # to gpu or cpu
                for k, v in input_data.items():
                    if isinstance(v, list):
                        for i in range(len(v)):
                            if isinstance(input_data[k][i], torch.Tensor):
                                input_data[k][i] = v[i].to(device)
                    elif isinstance(v, dict):
                        for k1, v1 in v.items():
                            if isinstance(v1, torch.Tensor):
                                input_data[k][k1] = v1.to(device)
                    else:
                        
                        input_data[k] = v.to(device)

                output_data, loss_term_all_cases = self.test_loss(model, input_data)
                predictons.append(output_data)
                if isinstance(loss_term_all_cases, dict):
                    loss_term_all_cases = [loss_term_all_cases]

                for loss_terms in loss_term_all_cases:
                    for k, v in loss_terms.items():
                        if not isinstance(v, str):
                            loss_terms[k] = '%.5g' % v
                    loss_str = '. '.join(['%s=%s' % (k, v) for k, v in loss_terms.items()])
                    index_str = ''
                    if 'index' in input_data:
                        index_str = ','.join(map(lambda x: str(x.item()), input_data['index']))
                    print(f'[{index_str}] {loss_str}', flush=True)
                    if test_iter == 0:
                        header = ['sample_ids'] + sorted(loss_terms.keys())
                        test_logger.write(','.join(header) + '\n')

                    train_log = [index_str] + [loss_terms[k] for k in sorted(loss_terms.keys())]
                    test_logger.write(','.join(train_log) + '\n')
                    test_iter += 1
                    
                test_logger.flush()
        
        test_logger.close()
        return predictons

    def evaluate_nolog(self, model, test_loader, exp_id=None, epoch=None, iteration=None, device='cuda'):

        self.caller = inspect.stack()[1].filename
        
        self.set_exp_dir(exp_id)

        model = self.load_model_state(model, exp_id, epoch=epoch, iteration=iteration)
        model.to(device)
        model.eval()

        predictons = []

        with torch.no_grad():
            test_iter = 0
            for input_data in test_loader:
                # print(val_iter)
                # to gpu or cpu
                for k, v in input_data.items():
                    if isinstance(v, list):
                        for i in range(len(v)):
                            if isinstance(input_data[k][i], torch.Tensor):
                                input_data[k][i] = v[i].to(device)
                    else:
                        input_data[k] = v.to(device)

                output_data, loss_terms = self.test_loss(model, input_data)
                predictons.append(output_data)
                for k, v in loss_terms.items():
                    if not isinstance(v, str):
                        loss_terms[k] = '%.5g' % v
                loss_str = '. '.join(['%s=%s' % (k, v) for k, v in loss_terms.items()])
                index_str = ''
                if 'index' in input_data:
                    index_str = ','.join(map(lambda x: str(x.item()), input_data['index']))
                print(f'[{index_str}] {loss_str}', flush=True)
                test_iter += 1
        return predictons

    def init_exp_dir(self, exp_id=None):
        self.set_exp_dir(exp_id=exp_id)
        if not os.path.exists(self.exp_dir):
            os.mkdir(self.exp_dir)
        if not os.path.exists(f'{self.exp_dir}/checkpoints'):
            os.mkdir(f'{self.exp_dir}/checkpoints')
        json.dump(self.hparam, open(f'{self.exp_dir}/hparam.json', 'w'), cls=HparamEncoder)
        try:
            shutil.copy(self.caller, f'{self.exp_dir}/train.py')
        except FileNotFoundError:
            pass
