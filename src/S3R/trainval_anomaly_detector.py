import os
import sys
import yaml
import time
import json
import torch
import random
import datetime
import numpy as np
import torch.optim as optim

# import _init_paths

from tqdm import tqdm
from munch import DefaultMunch
from utils.S3R_utilities.utils import save_best_record
from torch.utils.data import DataLoader
from terminaltables import AsciiTable, DoubleTable
from torch.utils.collect_env import get_pretty_env_info

from src.S3R.config import Config
from utils.S3R_utilities import PixelBar
from src.S3R.engine import do_train, inference
from dataset.s3r_dataset import Dataset
from models.S3R.detectors.detector import S3R
from src.S3R.apis import (
    mkdir, color, AverageMeter,
    setup_logger, setup_tblogger,
    synchronize, get_rank)
from options.S3R.option import S3RArgumentParser
# import torch.multiprocessing as mp

from typing import Dict, List, Optional, Tuple, Union

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def fixation(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    # mp.set_start_method('spawn')
    args = S3RArgumentParser().parse_args()

    if not args.inference:
        mkdir(args.checkpoint_path.joinpath(args.version))
        mkdir(args.log_path.joinpath(args.version))
        mkdir(args.dictionary_path)

    # if 'shanghaitech' in args.obj:
    #     import configs.shanghaitech.shanghaitech_dl as cfg
    # elif 'ucf-crime' in args.obj:
    #     import configs.ucf_crime.ucf_crime_dl as cfg
    # else:
    cfg_dict = {
                'car':'src.S3R.configs_objs.car_dl',
                'fan':'src.S3R.configs_objs.fan_dl', 
                'rolling_bearing':'src.S3R.configs_objs.rolling_bearing_dl', 
                'spherical_bearing':'src.S3R.configs_objs.spherical_bearing_dl', 
                'servo':'src.S3R.configs_objs.servo_dl', 
                'clip':'src.S3R.configs_objs.clip_dl', 
                'usb':'src.S3R.configs_objs.usb_dl', 
                'hinge':'src.S3R.configs_objs.hinge_dl', 
                'screw':'src.S3R.configs_objs.screw_dl', 
                'lock':'src.S3R.configs_objs.lock_dl', 
                'gear':'src.S3R.configs_objs.gear_dl', 
                'clock':'src.S3R.configs_objs.clock_dl', 
                'slide':'src.S3R.configs_objs.slide_dl', 
                'zipper':'src.S3R.configs_objs.zipper_dl', 
                'button':'src.S3R.configs_objs.button_dl', 
                'rubber_band':'src.S3R.configs_objs.rubber_band_dl', 
                'liquid':'src.S3R.configs_objs.liquid_dl', 
                'caster_wheel':'src.S3R.configs_objs.caster_wheel_dl', 
                'sticky_roller':'src.S3R.configs_objs.sticky_roller_dl', 
                'magnet':'src.S3R.configs_objs.magnet_dl', 
                'toothpaste':'src.S3R.configs_objs.toothpaste_dl', 
                'ball':'src.S3R.configs_objs.ball_dl'
            }
    import importlib
    cfg_module = cfg_dict[args.obj]
    cfg = importlib.import_module(cfg_module)

    if args.lr:
        lr = [args.lr] * args.max_epoch
    else:
        lr = [cfg.init_lr] * args.max_epoch

    config = Config(lr)
    envcols = config.envcols

    seed = cfg.random_state
    if args.seed > 0:
        seed = args.seed
    fixation(seed)

    global_rank = get_rank()
    logger = setup_logger("AnomalyDetection", args.log_path.joinpath(args.version), global_rank)
    logger.info('Arguments \n{info}\n{sep}'.format(info=args, sep='-' * envcols))

    env_info = get_pretty_env_info()

    logger.info("Collecting env info (might take some time)")
    logger.info(f"Environment Information\n\n{env_info}\n")

    data = DefaultMunch.fromDict(cfg.data)
    train_regular_dataset_cfg = data.train.regular
    train_anomaly_dataset_cfg = data.train.anomaly
    test_dataset_cfg = data.test

    # >> regular (normal) videos for the training set 
    train_regular_set = Dataset(**train_regular_dataset_cfg)
    dictionary = train_regular_set.dictionary

    # >> anomaly (abnormal) videos for the training set 
    train_anomaly_dataset_cfg.dictionary = dictionary
    train_anomaly_set = Dataset(**train_anomaly_dataset_cfg)

    # >> testing set
    test_dataset_cfg.dictionary = dictionary
    test_set = Dataset(**test_dataset_cfg)

    train_regular_loader = DataLoader(
            train_regular_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=False,
            drop_last=True,
            generator=torch.Generator(device='cuda'))

    train_anomaly_loader = DataLoader(
            train_anomaly_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=False,
            drop_last=True,
            generator=torch.Generator(device='cuda'))
    test_loader = DataLoader(
            test_set,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            generator=torch.Generator(device='cuda'))

    model = S3R(
            args.feature_size,
            args.batch_size,
            args.quantize_size,
            dropout = args.dropout,
            modality=cfg.modality)

    logger.info(train_regular_set.data_info)
    logger.info(train_anomaly_set.data_info)
    logger.info(test_set.data_info)
    logger.info('Model Structure: \n{}'.format(model))

    # for name, value in model.named_parameters():
    #     print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr[0],
        weight_decay=0.005)

    test_info = {
        'epoch': [], 'elapsed': [], 'now': [], 'train_loss': [],
        'test_{metric}'.format(metric='AUC' if 'xd-violence' not in args.obj else 'AP'): []}
    best_AUC = -1

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint)
            print('>>> Checkpoint {} loaded.'.format(color(args.resume)))
            auc = inference(test_loader, model, args, device)

            # >> Performance
            title = [['Dataset', 'Method', 'Feature', 'AUC (%)']]
            score = [[args.obj, args.model_name.upper(), args.backbone.upper(), f'{auc*100.:.3f}']]

            table = AsciiTable(title + score, ' Performance on {} '.format(args.obj))
            for i in range(len(title[0])):
                table.justify_columns[i] = 'center'
            logger.info('Summary Result on {} metric\n{}'.format('AUC', table.table))

        if args.inference:
            exit()
    else:
        score = inference(test_loader, model, args, device)

    sys_info = """
    {title}
        - dataset:\t {dataset}
        - version:\t {ver}
        - description:\t {descr}
        - initial {metric} score: {score:.3f} %
        - initial learning rate: {lr:.4f}
    """.format(
        title=color('Video Anomaly Detection', 'magenta'),
        dataset=color(args.obj, 'white', attrs=['bold', 'underline']),
        ver=args.version,
        descr=color(' '.join(args.descr)),
        metric='AUC' if 'xd-violence' not in args.obj else 'AP',
        score=score * 100.,
        lr=config.lr[0]
    )

    logger.info(sys_info)

    checkpoint_filename = '{data}_{model}_i3d_best.pth'
    log_filename = '{data}_{model}_i3d.score'

    # >> write title
    filename = log_filename.format(
        data=args.obj, model=args.model_name)
    log_filepath = args.log_path.joinpath(args.version).joinpath(filename)
    if os.path.exists(log_filepath):
        os.remove(log_filepath)


    metric = '{metric}'.format(metric='AUC' if 'xd-violence' not in args.obj else 'AP')
    with open(log_filepath, 'a') as f:
        f.write('\n{sep}\n{info}\n\n\n{env}\n{sep}\n'.format(sep = '*' * 10,
            info=yaml.dump(args, sort_keys=False, default_flow_style=False),
            env=env_info))
        f.write('\n{sep}\n{info}\n{sep}\n'.format(sep = '=' * 10, info=model))
        f.write('\n{}\n'.format(sys_info))

        title = '| {:^6s} | {:^8s} | {:^15s} | {:^30s} | {:^30s} |'.format(
            'Step', metric, 'Training loss', 'Elapsed time', 'Now')
        f.write('+{sep}+\n'.format(sep = '-'*(len(title)-2)))
        f.write('{}\n'.format(title))
        f.write('{sep}\n'.format(sep = '-'*len(title)))

    process_time = AverageMeter()
    end = time.time()
    bar = PixelBar('{now} - {dataset} - INFO -'.format(
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        dataset = color(args.obj),
    ), max=args.max_epoch)

    start_time = time.time()

    statistics = []
    if args.debug: args.max_epoch = 3
    for step in range(1, args.max_epoch + 1):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]

        if (step - 1) % len(train_regular_loader) == 0:
            loadern_iter = iter(train_regular_loader)

        if (step - 1) % len(train_anomaly_loader) == 0:
            loadera_iter = iter(train_anomaly_loader)

        loss = do_train(loadern_iter, loadera_iter, model, args.batch_size, optimizer, device)

        condition = (step % 1 == 0) if args.debug else \
                (step % args.evaluate_freq == 0 and step > args.evaluate_min_step)

        if condition:

            score = inference(test_loader, model, args, device)

            test_info["epoch"].append(step)
            test_info["test_{metric}".format(
                metric='AUC' if 'xd-violence' not in args.obj else 'AP')].append(score)
            test_info["train_loss"].append(loss)

            test_info["elapsed"].append(str(datetime.timedelta(seconds = time.time() - start_time)))
            now = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            test_info["now"].append(now)


            statistics.append([step, score])

            metric = 'test_{metric}'.format(metric='AUC' if 'xd-violence' not in args.obj else 'AP')
            if test_info[metric][-1] > best_AUC:
                best_AUC = test_info[metric][-1]
                filename = checkpoint_filename.format(
                    data=args.obj, model=args.model_name)
                torch.save(
                    model.state_dict(), args.checkpoint_path.joinpath(args.version).joinpath(filename))

                save_best_record(test_info, log_filepath, metric)

        # measure elapsed time
        process_time.update(time.time() - end)
        end = time.time()

        # plot progress
        info = \
            '({cnt}/{num})' \
            ' time: {pt:.3f}s, total: {total:}, eta: {eta:},' \
            ' lr: {lr}, loss: {loss:.4f}, {metric}: {score:.3f}' \
            .format(
                cnt = step, num=args.max_epoch,
                pt = process_time.val,
                total = bar.elapsed_td,
                eta = bar.eta_td,
                lr = optimizer.param_groups[0]['lr'],
                loss = loss,
                metric='AUC' if 'xd-violence' not in args.obj else 'AP',
                score = score * 100.)

        bar.suffix = info
        bar.next()

    bar.finish()

    with open(log_filepath, 'a') as f:
        f.write('+{sep}+\n'.format(sep = '-'*(len(title)-2)))

    # Performance
    auc_performance = np.array(statistics)
    best_epoch = np.argmax(auc_performance[:, -1])
    title = [['Step', 'AUC', 'Best']] if 'xd-violence' not in args.obj else [['Step', 'AP', 'Best']]
    score = [
        ['{}'.format(int(step)), '{:.3f}'.format(score * 100.), '{:^4s}'.format('*' if idx == best_epoch else '')]
        for idx, (step, score) in enumerate(auc_performance)
    ]

    # show top-k scores
    performance = np.array(score)[:, 1].astype(np.float32)
    top_k_idx = np.argsort(performance) # ascending order
    top_k_idx = top_k_idx[-args.report_k:] # only show k scores
    score = np.array(score)[top_k_idx].tolist()

    table = AsciiTable(title + score, ' Performance on {} '.format(args.obj))
    table.justify_columns[0], table.justify_columns[-1] = 'center', 'center'
    logger.info('Summary Result on {} metric\n{}'.format('AUC', table.table))
    

if __name__ == '__main__':
    main()
