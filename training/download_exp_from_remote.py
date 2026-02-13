import pysftp
import json
import os

def download_from_accre(accre_root, local_root, exp_folder_name, N_iteration=None, N_epoch=None, download_checkpoint=True, download_evaluation=False):

    if not os.path.exists(f'{local_root}/{exp_folder_name}'):
        os.mkdir(f'{local_root}/{exp_folder_name}')
    if download_checkpoint and not os.path.exists(f'{local_root}/{exp_folder_name}/checkpoints'):
        os.mkdir(f'{local_root}/{exp_folder_name}/checkpoints')
    if download_evaluation and not os.path.exists(f'{local_root}/{exp_folder_name}/evaluation'):
        os.mkdir(f'{local_root}/{exp_folder_name}/evaluation')

    cnopts = pysftp.CnOpts()
    cnopts.hostkeys = None

    # Make connection to sFTP
    with pysftp.Connection('hickory.accre.vanderbilt.edu',
                        username='sud',
                        password='Sdj116311`116311',
                        cnopts = cnopts
                        ) as sftp:
        for f in ['hparam.json', 'train.py', 'train_loss.csv', 'val_loss.csv']:
            try:
                file = sftp.get(f'{accre_root}/{exp_folder_name}/{f}', f'{local_root}/{exp_folder_name}/{f}')
            except FileNotFoundError:
                continue
        
        cp = sftp.listdir(f'{accre_root}/{exp_folder_name}/checkpoints')

        target = None

        for item in cp:
            if item.find('.pth') == -1:
                continue
            if item == 'best.pth':
                n_epoch = n_iter = 'best'
            else:
                epoch, iteration = item.replace('.pth', '').split('_')
                n_epoch = int(epoch[5: ])
                n_iter = int(iteration[4: ])
            if n_epoch == N_epoch or n_iter == N_iteration:
                target = item
                break
            if N_epoch is None and N_iteration is None and n_iter == 'best':
                target = item
                break
        
        if download_checkpoint and target is not None:
            file = sftp.get(f'{accre_root}/{exp_folder_name}/checkpoints/{target}', f'{local_root}/{exp_folder_name}/checkpoints/{target}')
        
        if download_evaluation:
            evals = sftp.listdir(f'{accre_root}/{exp_folder_name}/evaluation')
            for file in evals:
                file = sftp.get(f'{accre_root}/{exp_folder_name}/evaluation/{file}', f'{local_root}/{exp_folder_name}/evaluation/{file}')