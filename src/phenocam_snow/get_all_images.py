"""Script to download all the image URLs for a given site.
The results are written to a .txt file.
Example usage:
    python get_all_images.py delnortecounty2
Returns URLs in a file called delnortecounty2_urls.txt
"""


# Standard library
from argparse import ArgumentParser
import re
import requests

# Third party
import pandas as pd
from tqdm import tqdm


# Parse arguments
parser = ArgumentParser()
parser.add_argument("site_name")
args = parser.parse_args()

# Get all possible dates for the given site
try:
    resp = requests.get(
        f"https://phenocam.nau.edu/webcam/browse/{args.site_name}/",
        timeout=5
    )
except requests.exceptions.RequestException as e:
    raise SystemExit(e)
content = resp.content.decode()
year_tags = re.findall(r"<a name=\"[0-9]{4}\">", content)
years = [int(re.search(r"\d+", yt).group()) for yt in year_tags]
dates = pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31").strftime("%Y/%m/%d")

# Loop through all dates
root = "https://phenocam.nau.edu"
pattern = re.compile(rf"\/data\/archive\/{args.site_name}\/[0-9]{{4}}\/[0-9]{{2}}\/{args.site_name}_[0-9]{{4}}_[0-9]{{2}}_[0-9]{{2}}_[0-9]{{6}}\.jpg")
all_photos = []
for d in tqdm(dates):
    try:
        resp = requests.get(
            f"https://phenocam.nau.edu/webcam/browse/delnortecounty2/{d}/",
            timeout=5
        )
    except requests.exceptions.RequestException as e:
        continue
    if resp.ok:
        content = resp.content.decode()
        matches = pattern.finditer(content)
        for m in matches:
            all_photos.append(f"{root}{m.group()}")

# Save to file
with open(f"{args.site_name}_urls.txt", "w+") as f:
    for url in all_photos:
        print(url, file=f)
