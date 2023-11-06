# Standard library
from datetime import datetime
from io import BytesIO
import functools
import os
from pathlib import Path
import random
import re
import requests

# Third party
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def get_site_names():
    """Gets all available PhenoCam site names.

    :return: The list of PhenoCam site names, or `None` if an error occurred.
    :rtype: List[str]|None
    """
    site_names = []
    try:
        resp = requests.get(
            "https://phenocam.nau.edu/webcam/network/table/", timeout=10
        )
        if resp.ok:
            arr0 = resp.text.split("<tbody>")
            arr1 = arr0[1].split("</tbody>")
            arr2 = arr1[0].split("<a href=")
            for i in range(1, len(arr2)):
                name = arr2[i].split("</a>")[0].split(">")[1]
                site_names.append(name)
            return site_names
        else:
            print("Could not retrieve site names")
    except:
        print("Request timed out")
    return None

def get_all_images(site_name):
    """Gets URLs to all available images for a given site.

    :param site_name: The name of the site to get images from.

    :return: The list of all image URLs, or an empty list if an error occurred.
    :rtype: List[str]
    """
    image_urls = []
    base_url = "https://phenocam.nau.edu"
    try:
        resp = requests.get(f"{base_url}/webcam/browse/{site_name}", timeout=10)
        if resp.ok:
            month_suffixes = re.findall(
                f"\/webcam\/browse\/{site_name}\/[0-9]{{4}}\/[0-9]{{2}}",
                resp.text
            )
            num_months = len(month_suffixes)
            results = process_map(
                functools.partial(__get_images_for_month, base_url, site_name),
                month_suffixes,
                range(len(month_suffixes)),
                max_workers=8,
                unit="month"
            )
            nested_image_urls = [r[0] for r in results]
            nested_error_msgs = [r[1] for r in results]
            image_urls = [url for sublist in nested_image_urls for url in sublist]
            for error_msg_list in nested_error_msgs:
                for error_msg in error_msg_list:
                    print(error_msg)
        else:
            print(f"Could not retrieve {base_url}/webcam/browse/{site_name}")
    except Exception as e:
        print(e)
    return image_urls

def __get_images_for_month(base_url, site_name, month_suffix, i):
    """Helper function for get_all_images."""
    month_url = f"{base_url}{month_suffix}"
    url_results = []
    error_messages = []
    try:
        resp = requests.get(month_url, timeout=10)
        if resp.ok:
            day_pat = f"{month_suffix}\/[0-9]{{2}}"
            day_suffixes = re.findall(day_pat, resp.text)
            for ds in day_suffixes:
                day_url = f"{base_url}{ds}"
                month = re.search("[0-9]{4}/[0-9]{2}", day_url)[0]
                img_pat = f"\/data\/archive\/{site_name}\/{month}\/.*\.jpg"
                try:
                    resp = requests.get(day_url, timeout=10)
                    if resp.ok:
                        url_results = [
                                f"{base_url}{x}" for x in
                            re.findall(img_pat, resp.text)
                        ]
                    else:
                        error_messages.append(f"Could not retrieve {day_url}")
                except Exception as e:
                    error_messages.append(str(e))
        else:
            error_messages.append(f"Could not retrieve {month_url}")
    except Exception as e:
        error_messages.append(str(e))
    return url_results, error_messages

def download(site_name, save_to, n_photos):
    """Downloads photos taken in some time range at a given site.

    :param site_name: The name of the site to download from.
    :type site_name: str
    :param save_to: The destination directory for downloaded images. If the
        directory already exists, it is NOT cleared. New photos are added to
        the directory, except for duplicates, which are skipped.
    :type save_to: str
    :param n_photos: The number of photos to download.
    :type n_photos: int
    """
    # Check that the directory we are saving to exists
    if type(save_to) is not Path:
        save_dir = Path(save_to)
    else:
        save_dir = save_to
    if not save_dir.is_dir():
        os.mkdir(save_dir)

    # Configure logger
    log_filename = f'{datetime.now().isoformat().split(".")[0].replace(":", "-")}.log'
    log_filepath = save_dir.joinpath(log_filename)

    with open(log_filepath, "a") as log_file:
        # Get all image URLs
        print(f"Retrieving all image URLs for {site_name}")
        image_urls = get_all_images(site_name)
        random.shuffle(image_urls)

        # Download images until the number downloaded is the number requested
        # or until we are out of URLs
        n_downloaded = 0
        pbar = tqdm(total=n_photos, unit="downloaded")
        while n_downloaded < n_photos and len(image_urls) > 0:
            try:
                url = image_urls.pop()
                resp = requests.get(url, timeout=10)
                img = Image.open(BytesIO(resp.content))
                img.save(os.path.join(save_dir, url.split('/')[-1]))
                n_downloaded += 1
                pbar.update(1)
                log_file.write(f"INFO:Retrieved {url}\n")
            except Exception as e:
                log_file.write(f"ERROR:{e}\n")
        if n_downloaded < n_photos:
            log_file.write(f"WARN:only downloaded {n_downloaded} photos")
        pbar.close()

def download_from_log(source_log, save_to):
    """Downloads images that are listed in a log file.

    :param source_log: The log file to get image URLs from.
    :type source_log: str
    :param save_to: The destination directory for downloaded images.
    :type save_to: str
    """
    # Check that the directory we're saving to exists
    if type(save_to) is not Path:
        save_dir = Path(save_to)
    else:
        save_dir = save_to
    if not save_dir.is_dir():
        os.mkdir(save_dir)

    # Configure logger
    log_filename = f'{datetime.now().isoformat().split(".")[0].replace(":", "-")}.log'
    log_filepath = save_dir.joinpath(log_filename)

    # Read URLs from the source log
    img_urls = []
    with open(source_log, "r") as f:
        for line in f:
            if line.startswith("INFO:Retrieved "):
                url = line.split(" ")[1].strip()
                img_urls.append(url)

    # Download images
    with open(log_filepath, "a") as f:
        f.write(f"INFO:Read {len(img_urls)} image URLs from {str(source_log)}\n")
        for url in img_urls:
            try:
                resp = requests.get(url, timeout=10)
                if resp.ok:
                    img_fname = url.split("/")[-1]
                    output_fpath = save_dir.joinpath(img_fname)
                    img = Image.open(BytesIO(resp.content))
                    img.save(output_fpath)
                    f.write(f"INFO:Retrieved {resp.url}\n")
                else:
                    f.write(f"ERROR:Bad response for {resp.url}\n")
            except Exception as e:
                f.write(f"ERROR:{e}\n")

def label_images_via_subdir(site_name, categories, img_dir, save_to):
    """Allows the user to label images by moving them into the appropriate
       subdirectory.

    :param site_name: The name of the site.
    :type site_name: str
    :param categories: The image categories.
    :type categories: List[str]
    :param img_dir: The directory containing the image subdirectories.
    :type img_dir: str
    :param save_to: The destination path for the labels file.
    :type save_to: str
    """
    # Check that the image directory exists
    if type(img_dir) is not Path:
        img_dir = Path(img_dir)
    assert img_dir.is_dir()

    # Check that the category subdirectories exist
    dircats = []
    for cat in categories:
        dircat = img_dir.joinpath(Path(cat))
        dircats.append(dircat)
        if not dircat.exists() or not dircat.is_dir():
            os.mkdir(dircat)

    # Await user acknowledgement
    input(
        "Move images into the appropriate sub-directory then press any key to continue."
    )

    # Create annotations file
    timestamps = []
    for dircat in dircats:
        timestamps_subarr = []
        for img_fpath in dircat.glob("*.jpg"):
            ts_arr = img_fpath.stem.split("_")
            ts = "-".join(ts_arr[1:4])
            hms = ts_arr[-1]
            ts += f" {hms[:2]}:{hms[2:4]}:{hms[4:]}"
            timestamps_subarr.append(ts)
        timestamps.append(timestamps_subarr)
    df = pd.DataFrame(
        zip(timestamps, categories), columns=["timestamp", "label"]
    ).explode("timestamp")
    with open(save_to, "w+") as f:
        f.write(f"# Site: {img_dir.stem if site_name is None else site_name}\n")
        f.write("# Categories:\n")
        for i, cat in enumerate(categories):
            f.write(f"# {i}. {cat}\n")
    df.to_csv(save_to, mode="a", index=False)

    # Flatten directory (i.e., pull all images out of the subdirectories
    # back into their original directory)
    for item in img_dir.glob("*"):
        if item.is_dir():
            for subitem in sorted(item.glob("*")):
                new_path = Path(subitem.resolve().parent.parent).joinpath(subitem.name)
                subitem.rename(new_path)


def read_labels(labels_file):
    """Reads image-label pairs.

    :param labels_file: The path to the labels file.
    :type labels_file: str
    :return: A pandas DataFrame where each row contains the timestamp of an
        image, the path to that image, its label as a string, and the integer
        encoding of that label.
    :rtype: pd.DataFrame
    """
    # Extract meta information
    site_name = (
        pd.read_csv(labels_file, nrows=1, header=None)[0]
        .tolist()[0]
        .split("# Site: ")[1]
    )
    labels_dict = {}
    with open(labels_file, "r") as f:
        start_reading = False
        for line in f:
            if start_reading:
                if line[0] != "#":
                    break
                else:
                    int_label, str_label = line[1:].split(". ")
                    int_label = int(int_label)
                    str_label = str_label.strip()
                    labels_dict[str_label] = int_label
            if line == "# Categories:\n":
                start_reading = True

    # Sort timestamps
    df = pd.read_csv(labels_file, comment="#")
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    # Encode the labels as integers
    df["label"] = df["label"].astype("category")
    df["int_label"] = [labels_dict[x] for x in df["label"]]

    # Create image file names from timestamps
    img_name_col = []
    for ts in df.index:
        year = ts[:4]
        month = ts[5:7]
        day = ts[8:10]
        hms = ts.split(" ")[1].replace(":", "")
        img_name_col.append(f"{site_name}_{year}_{month}_{day}_{hms}.jpg")
    df["img_name"] = img_name_col

    return df
