# import argparse
from argparse import ArgumentParser
# import os
from os.path import isfile, splitext, isdir, join, basename, exists
from os import system
# import threading
from threading import Thread
# import struct
from struct import pack
# import torch
from torch import load
from torch_utils import select_device


def parse_args():
    parser = ArgumentParser(description='Convert .pt file to .wts')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pt) file path (required)')
    parser.add_argument('-o', '--output', help='Output (.wts) file path (optional)')
    args = parser.parse_args()
    if not isfile(args.weights):
        raise SystemExit('Invalid input file')
    if not args.output:
        args.output = splitext(args.weights)[0] + '.wts'
    elif isdir(args.output):
        args.output = join(
            args.output,
            splitext(basename(args.weights))[0] + '.wts')
    return args.weights, args.output


def pt_wts():
    """
    pt转换为wts
    """
    # pt_file, wts_file = parse_args()
    pt_file = './detect.pt'
    wts_file = './detect.wts'

    # Initialize
    device = select_device('')
    # Load model
    model = load(pt_file, map_location=device)  # load to FP32
    model = model['ema' if model.get('ema') else 'model'].float()

    # update anchor_grid info
    anchor_grid = model.model[-1].anchors * model.model[-1].stride[..., None, None]
    # model.model[-1].anchor_grid = anchor_grid
    delattr(model.model[-1], 'anchor_grid')  # model.model[-1] is detect layer
    model.model[-1].register_buffer("anchor_grid", anchor_grid)
    # The parameters are saved in the OrderDict through the "register_buffer" method, and then saved to the weight.

    model.to(device).eval()

    with open(wts_file, 'w') as f:
        f.write('{}\n'.format(len(model.state_dict().keys())))
        for k, v in model.state_dict().items():
            vr = v.reshape(-1).cpu().numpy()
            f.write('{} {} '.format(k, len(vr)))
            for vv in vr:
                f.write(' ')
                f.write(pack('>f', float(vv)).hex())
            f.write('\n')


def wts_engine():
    """
    wts转换为engine
    """
    system('yolov5.exe -s detect.wts detect.engine s')


def start_pt_wts():
    th = Thread(target=pt_wts)
    th.start()
    th.join()


def start_wts_engine():
    th2 = Thread(target=wts_engine)
    th2.start()
    th2.join()


def run():
    if not exists("./detect.pt"):
        print("pt文件不存在")
    else:
        print("开始转换pt文件为wts文件，请稍等。。。")
        # pt_wts()
        start_pt_wts()

        if not exists("./detect.wts"):
            print("wts文件转换失败")
        else:
            print("wts文件已生成")
            print("开始转换为engine文件，请稍等。。。")
            # wts_engine()
            start_wts_engine()


run()
input('Press Enter to exit...')
