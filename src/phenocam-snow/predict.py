# Standard library imports
from argparse import ArgumentParser
from typing import List, Optional, Tuple, Union

# Third party imports
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms

# Local application imports
from utils import *
from pl_model import *


def main():
    parser = ArgumentParser(
        description='Predicts image category using the given model')
    parser.add_argument(
        'site_name', help='The PhenoCam site for which we are generating predictions.')
    parser.add_argument('model_path', help='The path of the model to use.')
    parser.add_argument('--categories', nargs='+',
                        help='The image categories to use.')
    parser.add_argument('--url', default=None,
                        help='Provide this if you want to get a prediction for \
                            a single online image.')
    parser.add_argument('--directory', default=None,
                        help='Provide this if you want to get predictions for all \
                            images in a local directory.')
    args = parser.parse_args()

    if args.url and args.directory:
        print('Cannot specify both --url and --directory')
    elif not (args.url or args.directory):
        print('Please specify either --url or --directory')
    else:
        model = load_model_from_file(args.model_path, len(args.categories))
        if args.url:
            run_model_online(model, args.site_name, args.categories, args.url)
        elif args.directory:
            run_model_offline(model, args.site_name,
                              args.categories, args.directory)


def classify_online(model: PhenoCamResNet,
                    categories: List[str],
                    img_url: str) -> Tuple[np.ndarray, str]:
    """
    ----------
    Parameters
    ----------
    model: The model to use.
    categories: A list of the image categories.
    img_url: The URL of the image to run classification on.

    ----------
    Returns
    ----------
    A 2-tuple where the first element is the image at img_url as a NumPy
    array and the second element is the predicted label for this image.

    ----------
    Example
    ----------
    >>> model = PhenoCamResNet()
    >>> categories = ['too dark', 'no snow', 'snow']
    >>> url = 'https://phenocam.sr.unh.edu/data/latest/canadaOBS.jpg'
    >>> classify_online(model, categories, url)
    ([some_np_array], 'snow')
    """
    try:
        resp = requests.get(img_url, timeout=5)
    except:
        print('Request timed out')
    if resp.ok:
        img = Image.open(BytesIO(resp.content))
        np_img = np.array(img)
        x = transforms.ToTensor()(img)
        dm = PhenoCamDataModule('dummy_site_name',
                                'dummy_train_dir',
                                'dummy_train_anns',
                                'dummy_test_dir',
                                'dummy_test_anns')
        x = dm.std_transform(x.unsqueeze(0))
        z = model(x)
        pred = categories[torch.argmax(z, dim=1)]
        return (np_img, pred)
    else:
        print('Error occurred')
    return None


def classify_offline(model: PhenoCamResNet,
                     categories: List[str],
                     img_path: FlexPath) -> Tuple[np.ndarray, str]:
    """
    ----------
    Parameters
    ----------
    model: The model to use.
    categories: A list of the image categories.
    img_path: The path of the image to run classification on.

    ----------
    Returns
    ----------
    A 2-tuple where the first element is the image at img_url as a NumPy
    array and the second element is the predicted label for this image.

    ----------
    Example
    ----------
    >>> model = PhenoCamResNet()
    >>> categories = ['too dark', 'no snow', 'snow']
    >>> img_path = 'canadaOBS_test/test_img.jpg'
    >>> classify_offline(model, categories, img_path)
    ([some_np_array], 'too dark') 
    """
    dm = PhenoCamDataModule('dummy_site_name',
                            'dummy_train_dir',
                            'dummy_train_anns',
                            'dummy_test_dir',
                            'dummy_test_anns')
    x = dm.std_transform((read_image(img_path) / 255).unsqueeze(0))
    z = model(x)
    pred = categories[torch.argmax(z, dim=1)]
    return pred


def load_model_from_file(model_path: FlexPath, n_classes: int) -> PhenoCamResNet:
    """
    ----------
    Parameters
    ----------
    model_path: The filepath where the model is stored.
    n_classes: The number of classes this model predicts.

    ----------
    Returns
    ----------
    The loaded model.
    """
    model = PhenoCamResNet(n_classes).load_from_checkpoint(model_path)
    model.freeze()
    return model


def run_model_offline(model: PhenoCamResNet,
                      site_name: str,
                      categories: List[str],
                      img_dir: FlexPath) -> None:
    """
    ----------
    Description
    ----------
    Gets predicted labels for all the images in img_dir and writes
    them to img_dir/predictions.csv.

    ----------
    Parameters
    ----------
    model: The model to use.
    site_name: The name of the site.
    categories: The string labels.
    img_dir: The directory containing the images to classify.

    ----------
    Returns
    ----------
    None.
    """

    ######################
    # 1. Get predictions #
    ######################
    if type(img_dir) is str:
        img_dir = Path(img_dir)
    timestamps, predictions = [], []
    for img_path in img_dir.glob('*.jpg'):
        ts_arr = img_path.stem.split('_')
        ts = '-'.join(ts_arr[1:4])
        hms = ts_arr[-1]
        ts += f' {hms[:2]}:{hms[2:4]}:{hms[4:]}'
        timestamps.append(ts)
        predictions.append(classify_offline(model, categories, str(img_path)))

    ###################
    # 2. Save to file #
    ###################
    df = pd.DataFrame(zip(timestamps, predictions),
                      columns=['timestamp', 'label'])
    save_to = img_dir.joinpath('predictions.csv')
    with open(save_to, 'w+') as f:
        f.write(f'# Site: {site_name}\n')
        f.write('# Categories:\n')
        for i, cat in enumerate(categories):
            f.write(f'# {i}. {cat}\n')
    df.to_csv(save_to, mode='a', line_terminator='\n', index=False)


def run_model_online(model: PhenoCamResNet,
                     site_name: str,
                     categories: List[str],
                     img_url: str) -> None:
    """
    ----------
    Description
    ----------
    Gets predicted label for the image pointed to by img_url, 
    and displays them in this notebook.

    ----------
    Parameters
    ----------
    model: The model to use.
    site_name: The name of the site.
    categories: The string labels.
    img_url: The URL of the image for which you want a prediction.

    ----------
    Returns
    ----------
    None.
    """
    ######################
    # 1. Get predictions #
    ######################
    img, pred = classify_online(model, categories, img_url)

    ###################
    # 2. Print result #
    ###################
    print(pred)


if __name__ == '__main__':
    main()
