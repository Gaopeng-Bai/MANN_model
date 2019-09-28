# MANN
Meta learning for MANN.

* Install the  package specify in requirement file.
* Achieve the datasets that described music information, here using the datasets from Spotify factory. Renamed this file as "data" in the superior menu named "data_resources". 
* Create two files named as "save_data" and "summary" under the file of "data_resources".

# Run first
* Run dataLoader to generate the numpy files for learning model and relavent files as well.
## Class dataLoader:
    # @random_number_files: int. The number of .json files in one npy file.
    # @file_number: int.    The number of .json to be generated in total.
    # @ data_dir: the resources files in local data file.
    # @ save_dir: save dir in local save_data dir.
# Training
```
* Setting the parameters in Train.py,batch size etc.Then run python Train.py.
```