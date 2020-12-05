#!/usr/bin/python3
import re
import os
import urllib.request
import argparse
import json
import base64


def create_url(url):
    """
    From the given url, produce a URL that is compatible with Github's REST API. Can handle blob or tree paths.
    """
    repo_only_url = re.compile(
        r"https:\/\/github\.com\/[a-z\d](?:[a-z\d]|-(?=[a-z\d])){0,38}\/[a-zA-Z0-9]+"
    )
    re_branch = re.compile("/(tree|blob)/(.+?)/")

    # extract the branch name from the given url (e.g master)
    branch = re_branch.search(url)
    download_dirs = url[branch.end() :]
    api_url = (
        url[: branch.start()].replace("github.com", "api.github.com/repos", 1)
        + "/contents/"
        + download_dirs
        + "?ref="
        + branch.group(2)
    )
    return api_url, download_dirs


def download(repo_url, output_dir="./", username="", password=""):
    """
    Downloads the files and directories in repo_url.
    """

    # generate the url which returns the JSON data
    api_url, download_dirs = create_url(repo_url)

    # To handle file names.
    if len(download_dirs.split(".")) == 0:
        dir_out = os.path.join(output_dir, download_dirs)
    else:
        dir_out = os.path.join(output_dir, "/".join(download_dirs.split("/")[:-1]))

    headers = [("User-agent", "DeepReg")]
    if username and password:
        headers.append(
            (
                "Authorization",
                "Basic %s"
                % base64.urlsafe_b64encode(
                    bytes("%s:%s" % (username, password), "ascii")
                ),
            )
        )

    opener = urllib.request.build_opener()
    opener.addheaders = headers
    urllib.request.install_opener(opener)
    response = urllib.request.urlretrieve(api_url)

    # make a directory with the name which is taken from
    # the actual repo
    os.makedirs(dir_out, exist_ok=True)

    # total files count
    total_files = 0

    with open(response[0], "r") as f:
        data = json.load(f)
        # getting the total number of files so that we
        # can use it for the output information later
        total_files += len(data)

        # If the data is a file, download it as one.
        if isinstance(data, dict) and data["type"] == "file":
            opener = urllib.request.build_opener()
            opener.addheaders = headers
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(
                data["download_url"], os.path.join(dir_out, data["name"])
            )
            print("Downloaded: {}".format(os.path.join(dir_out, data["name"])))

            return total_files

        for file in data:
            file_url = file["download_url"]
            file_name = file["name"]

            path = file["path"]

            dirname = os.path.dirname(path)

            if dirname != "":
                os.makedirs(os.path.dirname(path), exist_ok=True)
            else:
                pass

            if file_url is not None:
                opener = urllib.request.build_opener()
                opener.addheaders = headers
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(file_url, path)
                print("Downloaded: {}".format(path))
            else:
                download(file["html_url"], dir_out, username, password)

    return total_files


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        "-d",
        dest="output_dir",
        default="./",
        help="All directories will be downloaded to the specified directory.",
    )
    parser.add_argument(
        "--username", "-u", dest="username", default="", help="Your GitHub username."
    )
    parser.add_argument(
        "--password", "-p", dest="password", default="", help="Your GitHub password."
    )
    args = parser.parse_args()

    urls = [
        "https://github.com/DeepRegNet/DeepReg/tree/main/config",
        "https://github.com/DeepRegNet/DeepReg/tree/main/data",
        "https://github.com/DeepRegNet/DeepReg/tree/main/demos",
    ]

    for url in urls:
        total_files = download(url, args.output_dir, args.username, args.password)

    print(
        "\nDownload complete. Please refer to the DeepReg Quick Start guide for next steps."
    )


if __name__ == "__main__":
    main()
