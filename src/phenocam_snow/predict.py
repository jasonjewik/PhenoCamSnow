# Standard library imports
from argparse import ArgumentParser

# Local application imports
from .data import *
from .model import *
from .utils import *

# Third party imports
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms


def main():
    parser = ArgumentParser(description="Predicts image category using the given model")
    parser.add_argument(
        "site_name", help="The PhenoCam site for which we are generating predictions."
    )
    parser.add_argument("model_path", help="The path of the model to use.")
    parser.add_argument("resnet", help="The type of resnet used.")
    parser.add_argument("--categories", nargs="+", help="The image categories to use.")
    parser.add_argument(
        "--url",
        default=None,
        help="Provide this if you want to get a prediction for \
                            a single online image.",
    )
    parser.add_argument(
        "--directory",
        default=None,
        help="Provide this if you want to get predictions for all \
                            images in a local directory.",
    )
    args = parser.parse_args()

    if args.url and args.directory:
        print("Cannot specify both --url and --directory")
    elif not (args.url or args.directory):
        print("Please specify either --url or --directory")
    else:
        model = load_model_from_file(args.model_path, args.resnet, len(args.categories))
        if args.url:
            run_model_online(model, args.site_name, args.categories, args.url)
        elif args.directory:
            run_model_offline(model, args.site_name, args.categories, args.directory)


def classify_online(model, categories, img_url):
    """Performs online classification.

    :param model: The model to use.
    :type model: PhenoCamResNet
    :param categories: The categories to use.
    :type categories: List[str]
    :param img_url: The URL of the image to run classification on.
    :type img_url: str
    :return: A 2-tuple where the first element is the image at `img_url` as a
        NumPy array and the second element is the predicted label.
    """
    try:
        resp = requests.get(img_url, timeout=5, verify=False)
    except:
        print("Request timed out")
    if resp.ok:
        img = Image.open(BytesIO(resp.content))
        np_img = np.array(img).T
        x = torch.from_numpy(np_img)
        dm = PhenoCamDataModule(
            "dummy_site_name",
            "dummy_train_dir",
            "dummy_train_anns",
            "dummy_test_dir",
            "dummy_test_anns",
        )
        x = dm.preprocess(x.unsqueeze(0))
        yhat = model(x)
        pred = categories[torch.argmax(yhat, dim=1)]
        return (np_img, pred)
    else:
        print("Error occurred")
    return None


def classify_offline(model, categories, img_path):
    """Performs offline classification.

    :param model: The model to use.
    :type model: PhenoCamResNet
    :param categories: The image categories.
    :type categories: List[str]
    :param img_path: The file path of the image to classify.
    :type img_path: str
    :return: A 2-tuple where the first element is the image at `img_path` as a
        NumPy array and the second element is the predicted label.
    """
    dm = PhenoCamDataModule(
        "dummy_site_name",
        "dummy_train_dir",
        "dummy_train_anns",
        "dummy_test_dir",
        "dummy_test_anns",
    )
    x = dm.preprocess(read_image(img_path).unsqueeze(0))
    yhat = model(x)
    pred = categories[torch.argmax(yhat, dim=1)]
    return pred


def load_model_from_file(model_path, resnet, n_classes):
    """Loads a model from checkpoint file.

    :param model_path: The path to the model checkpoint file.
    :type model_path: str
    :param resnet: The type of Resnet that was used.
    :type resnet: str
    :param n_classes: The number of classes.
    :type n_classes: int
    :return: The loaded model.
    :rtype: PhenoCamResNet
    """
    model = PhenoCamResNet.load_from_checkpoint(
        model_path, resnet=resnet, n_classes=n_classes
    )
    model.freeze()
    return model


def run_model_offline(model, site_name, categories, img_dir):
    """Gets predicted labels for all images in a directory.

    :param model: The model to use.
    :type model: PhenoCamResNet
    :param site_name: The name of the PhenoCam site.
    :type site_name: str
    :param img_dir: The directory containing the images to classify.
    :type img_dir: str
    :return: A pandas DataFrame with predictions.
    :rtype: pd.DataFrame
    """
    ######################
    # 1. Get predictions #
    ######################
    if type(img_dir) is str:
        img_dir = Path(img_dir)
    timestamps, predictions = [], []
    for img_path in img_dir.glob("*.jpg"):
        ts_arr = img_path.stem.split("_")
        ts = "-".join(ts_arr[1:4])
        hms = ts_arr[-1]
        ts += f" {hms[:2]}:{hms[2:4]}:{hms[4:]}"
        timestamps.append(ts)
        predictions.append(classify_offline(model, categories, str(img_path)))

    ###################
    # 2. Save to file #
    ###################
    df = pd.DataFrame(zip(timestamps, predictions), columns=["timestamp", "label"])
    save_to = img_dir.joinpath("predictions.csv")
    with open(save_to, "w+") as f:
        f.write(f"# Site: {site_name}\n")
        f.write("# Categories:\n")
        for i, cat in enumerate(categories):
            f.write(f"# {i}. {cat}\n")
    df.to_csv(save_to, mode="a", line_terminator="\n", index=False)


def run_model_online(model, site_name, categories, img_url):
    """Gets predicted labels for all images in a directory.

    :param model: The model to use.
    :type model: PhenoCamResNet
    :param site_name: The name of the PhenoCam site.
    :type site_name: str
    :param img_url: The URL of the image for which you want a prediction.
    :type img_url: str
    """
    ######################
    # 1. Get predictions #
    ######################
    img, pred = classify_online(model, categories, img_url)

    ###################
    # 2. Print result #
    ###################
    print(pred)


if __name__ == "__main__":
    main()
