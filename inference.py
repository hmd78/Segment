import mmcv
from mmengine.visualization import Visualizer
from mmdet.apis import init_detector, inference_detector
from mmengine.config import Config
from mmengine.runner import Runner
import sys
import argparse


class Inference():
  def __init__(self, config, checkpoint, result_dest, device):
        self.colors = ['blue', 'purple', 'red', 'green', 'orange', 'salmon', 'pink', 'gold',
                        'orchid', 'slateblue', 'limegreen', 'seagreen', 'darkgreen', 'olive',
                        'teal', 'aquamarine', 'steelblue', 'powderblue', 'dodgerblue', 'navy',
                        'magenta', 'sienna', 'maroon']
        self.checkpoint_file = checkpoint
        self.model = init_detector(config, self.checkpoint_file, device=device)
        self.cfg = Config.fromfile(config)
        self.runner = Runner.from_cfg(self.cfg)
        self.res_dest = result_dest


  def make_inference(self, image_file, pred_thr, is_binary=False):
    if not is_binary:
        img = mmcv.imread(image_file,channel_order='rgb')
    else:
        img = mmcv.imfrombytes(image_file)
        img = mmcv.bgr2rgb(img)
    # checkpoint_file = checkpoint
    # model = init_detector(config, checkpoint_file, device='cpu')

    new_result = inference_detector(self.model, img)

    visualizer_now = Visualizer.get_current_instance()

    visualizer_now.dataset_meta = self.model.dataset_meta
    visualizer_now.add_datasample(
        'new_result',
        img,
        data_sample=new_result,
        draw_gt=False,
        wait_time=0,
        pred_score_thr = pred_thr,
        out_file=self.res_dest
    )
    visualizer_now.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Inference',
        description='make inference for instance segmentation task with rtmdet model',
        epilog='rtmdet instance segm')
    parser.add_argument('--config', type=str,
                        help='path to model config file')
    parser.add_argument('--checkpoint', type=str,
                        help='model checkpoint file path')
    parser.add_argument('--result_dest', type=str,
                        help='path to destination where results should be saved to')
    parser.add_argument('--image', type=str,
                        help='image you want to predict')
    parser.add_argument('--pred_thr', type=float, default=0.3,
                        help='prediction threshold')
    parser.add_argument('--gpu', action='store_true', help='make inference on gpu')

    args = parser.parse_args()
    if args.gpu :
        inferer = Inference(args.config, args.checkpoint, args.result_dest, 'cuda:0')
        inferer.make_inference(args.image, args.pred_thr)
    else:
        inferer = Inference(args.config, args.checkpoint, args.result_dest, 'cpu')
        inferer.make_inference(args.image, args.pred_thr)
    print(f'FINISHED\nRESULTS SAVED TO {args.result_dest}')