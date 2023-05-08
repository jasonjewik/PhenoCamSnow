# Standard library
from datetime import datetime
from io import BytesIO
import os
from pathlib import Path
import random
import requests

# Third party
import numpy as np
import pandas as pd
from PIL import Image


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


def get_site_dates(site_name):
    """Gets the dates of the first and last images collected at a site.

    :param site_name: The name of the PhenoCam site to download from.
    :type site_name: str
    :return: A 2-tuple of the format `(first_date, last_date)`, or
        `(None, None)` if an error occurred.
    :rtype: Tuple[str,str]|Tuple[None, None]
    """
    start_date, end_date = None, None
    try:
        resp = requests.get(
            f"https://phenocam.nau.edu/webcam/sites/{site_name}/", timeout=10
        )
        if resp.ok:
            start_date = resp.text.split("<strong>Start Date:</strong> ")[1][:10]
            end_date = resp.text.split("<strong>Last Date:</strong> ")[1][:10]
        else:
            print("Could not retrieve start and end date")
    except:
        print("Request timed out")
    return (start_date, end_date)


def download(site_name, dates, save_to, n_photos):
    """Downloads photos taken in some time range at a given site.

    :param site_name: The name of the site to download from.
    :type site_name: str
    :param dates: A 2-tuple indicating the oldest and youngest allowable
        photos.
    :type dates: Tuple[str, str]
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
        # Randomly order all possible timestamps
        date_range = list(pd.date_range(start=dates[0], end=dates[1], freq="30min"))
        random.shuffle(date_range)

        # Download images
        home_url = f"https://phenocam.nau.edu/webcam/browse/{site_name}"
        img_template = f"https://phenocam.nau.edu/data/archive/{site_name}"
        n_downloaded = 0

        # Keep downloading until the number downloaded is the number requested
        # or until we are out of dates to sample images from
        while n_downloaded < n_photos and len(date_range) > 0:
            my_datetime = date_range.pop()
            Y = str(my_datetime.year)
            m = str(my_datetime.month).zfill(2)
            D = str(my_datetime.day).zfill(2)
            month_url = f"{home_url}/{Y}/{m}/{D}"
            try:
                resp1 = requests.get(month_url, timeout=10)
            except:
                log_file.write(f"ERROR:Request timed out\n")
                continue
            if resp1.ok:  # Access the archive for the chosen timestamp's month
                arr = resp1.text.split('<span class="imglabel">')[1:]
                success = False
                for a in arr:
                    orig_timestamp = a.split("&nbsp")[0].strip()
                    strip_timestamp = orig_timestamp.replace(":", "")
                    try:
                        pd_timestamp = pd.to_datetime(f"{Y}-{m}-{D} {orig_timestamp}")
                    except:
                        log_file.write(f"WARN:Could not parse {orig_timestamp}\n")
                        break
                    # Find the image within 5 minutes of the chosen timestamp
                    if abs(my_datetime - pd_timestamp) <= pd.Timedelta("5min"):
                        img_fname = f"{site_name}_{Y}_{m}_{D}_{strip_timestamp}.jpg"
                        img_url = f"{img_template}/{Y}/{m}/{img_fname}"
                        output_fpath = save_dir.joinpath(img_fname)
                        if output_fpath.is_file():
                            log_file.write(
                                f"WARN:{img_fname} was already downloaded, skipping\n"
                            )
                            break
                        try:
                            resp2 = requests.get(img_url, timeout=10)
                        except Exception as e:
                            log_file.write(f"ERROR:{e}\n")
                        if resp2.ok:
                            try:
                                img = Image.open(BytesIO(resp2.content))
                                img.save(output_fpath)
                                success = True
                                n_downloaded += 1
                                log_file.write(f"INFO:Retrieved {resp2.url}\n")
                            except:
                                log_file.write(
                                    f"WARN:Could not read or save image from {resp2.url}\n"
                                )
                            break
                        else:
                            log_file.write(f"WARN:Could not reach {resp2.url}\n")
                if not success:
                    log_file.write(
                        f"WARN:Could not find an image within 5 minutes of {str(my_datetime)}\n"
                    )
            else:
                log_file.write(f"WARN:Could not reach {month_url}\n")


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
            except:
                f.write("ERROR:Request timed out\n")
                break
            if resp.ok:
                try:
                    img_fname = url.split("/")[-1]
                    output_fpath = save_dir.joinpath(img_fname)
                    img = Image.open(BytesIO(resp.content))
                    img.save(output_fpath)
                    f.write(f"INFO:Retrieved {resp.url}\n")
                except:
                    f.write(f"WARN:Could not read or save image from {resp.url}\n")
            else:
                f.write(f"ERROR:Bad response for {resp.url}\n")


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
    df.to_csv(save_to, mode="a", lineterminator="\n", index=False)

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
