# Standard library
import argparse

# Local application
from .utils import get_all_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("site_name")
    args = parser.parse_args()

    urls = get_all_images(args.site_name)
    print(f"Downloaded {len(urls)} urls")

    with open("urls.txt", "w") as f:
        for url in urls:
            f.write(f"{url}\n")

    print("Wrote urls to urls.txt")


if __name__ == "__main__":
    main()

