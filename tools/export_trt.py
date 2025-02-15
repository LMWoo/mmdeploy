from mmdeploy.apis import torch2onnx
from mmdeploy.apis.tensorrt import onnx2tensorrt
from mmdeploy.backend.sdk.export_info import export2SDK

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Export model to Tensorrt.')
    parser.add_argument('--mmdetection_path', help='MMDetection directory path')
    parser.add_argument('--device', help='device used for conversion', default='cpu')
    parser.add_argument('--sample_img', help='deploy image path')
    parser.add_argument('--deploy_cfg', help='deploy option')
    parser.add_argument('--model_cfg', help='model option')
    parser.add_argument('--checkpoint', help='model checkpoint')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    
    root_path = args.mmdetection_path
    img = os.path.join(root_path, args.sample_img)
    work_dir = os.path.join(root_path, 'save_dir')
    save_file = 'end2end.onnx'
    deploy_cfg = os.path.join(root_path, args.deploy_cfg)
    model_cfg = os.path.join(root_path, args.model_cfg)
    model_checkpoint = os.path.join(root_path, args.checkpoint)

    device = 'cuda'

    torch2onnx(img, work_dir, save_file, deploy_cfg, model_cfg, model_checkpoint, device)

    onnx_model = os.path.join(work_dir, save_file)
    save_file = 'end2end.engine'
    model_id = 0
    onnx2tensorrt(work_dir, save_file, model_id, deploy_cfg, onnx_model, device)

    export2SDK(deploy_cfg, model_cfg, work_dir, pth=model_checkpoint, device=device)

