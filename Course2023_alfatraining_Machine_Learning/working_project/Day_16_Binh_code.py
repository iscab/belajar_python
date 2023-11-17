# Python, using Anaconda environment
# Week 4, Day 16

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from pathlib import Path
import tarfile
import shutil
import urllib.request

IMAGES_PATH = Path() / "images"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution)


def load_wine_data_0(url,ordner_name,csv_name):
    ordner = Path(ordner_name)
    tarball_path = ordner / Path(url).name # der Filename des Archivs
    # csv_file = ordner / Path(csv_name)

    if not tarball_path.is_file():
        ordner.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, tarball_path)

    csv_file = ordner / "wine_quality" / csv_name
    # der tarball erzeugt beim Auspacken den Ordnernamen "housing"
    # das muss man wissen

    if not csv_file.is_file():
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path=ordner.name)

    return pd.read_csv(csv_file)

def load_wine_data(url, ordner_name, csv_names):
    ordner = Path(ordner_name)
    zip_file_path = ordner / Path(url).name
    csv_paths = []

    if not zip_file_path.is_file():
        ordner.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, zip_file_path)

    destination_path = ordner / "wine_quality"

    if not destination_path.is_dir():
        destination_path.mkdir()

    shutil.unpack_archive(zip_file_path, destination_path)

    for csv_name in csv_names:
        csv_file = destination_path / csv_name
        if csv_file.is_file():
            csv_paths.append(csv_file)

    return csv_paths


# Usage
csv_paths = load_wine_data(
    url="https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
    ordner_name="datasets",
    csv_names=["winequality-red.csv", "winequality-white.csv"]
)


# Iterate through the CSV paths and assign them to df1 and df2
for idx, csv_path in enumerate(csv_paths):
    df = pd.read_csv(csv_path, delimiter=";")
    if idx == 0:
        df1 = df
    elif idx == 1:
        df2 = df

# df_rot = df_rot.drop_duplicates()
# df_w= df_w.drop_duplicates()

# plt.rc('font', size=14)
# plt.rc('axes', labelsize=14, titlesize=14)
# plt.rc('legend', fontsize=14)
# plt.rc('xtick', labelsize=10)
# plt.rc('ytick', labelsize=10)
#
# df_rot.hist(bins=50, figsize=(12, 8))
# save_fig("attribute_histogram_plots for Rot Wein") # extra code
# plt.show()

# df_w['residual sugar'].value_counts()

# l = df.columns.values
# number_of_columns=12
# number_of_rows = len(l)-1/number_of_columns
# plt.figure(figsize=(number_of_columns,5*number_of_rows))
# for i in range(0,len(l)):
#     plt.subplot(number_of_rows + 1,number_of_columns,i+1)
#     sns.set_style('whitegrid')
#     sns.boxplot(df[l[i]],color='green',orient='v')
#     plt.tight_layout()






# end of file
