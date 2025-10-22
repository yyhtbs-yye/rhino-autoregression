import types
from rhtrain.rhino_train import main

args_dict = {
    'config': 'configs/pixelcnn/train_pixelcnnpp_ffhq_64.yaml',
    'resume_from': 'work_dirs/pixelcnnpp_ffhq_64/run_2/last.pt',
}

args = types.SimpleNamespace(**args_dict)

if __name__ == "__main__":

    main(args)