import types
from rhtrain.rhino_train import main

args_dict = {
    'config': 'configs/pixelcnn/train_pixelcnnpp_ffhq_64.yaml',
    'resume_from': None,
}

args = types.SimpleNamespace(**args_dict)

if __name__ == "__main__":

    main(args)