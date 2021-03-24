import requests
import click
from pathlib import Path


@click.command()
@click.option("-u", "--url")
@click.option("-f", "--filename", type=click.Path())
def download_file(url, filename):
    print(f"Downloading from {url} to {filename}")

    filename = Path(filename)
    response = requests.get(url)
    with open(filename, "wb") as f:
        f.write(response.content)


if __name__ == "__main__":
    download_file()
