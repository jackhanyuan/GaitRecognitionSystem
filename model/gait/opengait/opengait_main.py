import os
import sys
sys.path.append(os.path.dirname(__file__))
import argparse
from modeling import models
from tools import config_loader, params_count, get_msg_mgr


parser = argparse.ArgumentParser(description='Main program for opengait.')
parser.add_argument('--cfgs', type=str,
                    default='config/default.yaml', help="path of config file")
parser.add_argument('--log_to_file', action='store_true',
                    help="log to file, default path is: output/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
parser.add_argument('--device', default='0', type=str, help='device id, 0 or 1 or 0,1')
opt = parser.parse_args()

opt.cfgs = "configs/gaitgl/gaitgl_HID_OutdoorGait_CASIA-B_OUMVLP.yaml"
opt.log_to_file = False


def initialization(cfgs):
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs['evaluator_cfg']
    output_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'],
                               cfgs['model_cfg']['model'], engine_cfg['save_name'])

    msg_mgr.init_logger(output_path, opt.log_to_file)
    msg_mgr.log_info(engine_cfg)


def run_model(cfgs):
    msg_mgr = get_msg_mgr()
    model_cfg = cfgs['model_cfg']
    msg_mgr.log_info(model_cfg)
    Model = getattr(models, model_cfg['model'])
    model = Model(cfgs, training=False)
    if cfgs['trainer_cfg']['fix_BN']:
        model.fix_BN()
    msg_mgr.log_info(params_count(model))
    msg_mgr.log_info("Model Initialization Finished!")
    res = Model.run_test(model)

    return res


def main():
    """
    rank表示进程序号，用于进程间通讯，每一个进程对应了一个rank,单机多卡中可以理解为第几个GPU。
    args为函数传入的参数
    """
    cfgs = config_loader(opt.cfgs)
    initialization(cfgs)
    res = run_model(cfgs)

    return res


# if __name__ == '__main__':
#     WORK_PATH = ".."
#     os.chdir(WORK_PATH)
#     print("WORK_PATH:", os.getcwd())
#     main()

