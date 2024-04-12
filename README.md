# Re-Trace

Pytorch implementation of the model Re-Trace proposed in "RE-Trace: Re-Identification of Modified GPS Trajectories" (https://dl.acm.org/doi/10.1145/3643680)




### Datasets 

San Francisco and Porto are openly available data sets, while Hannover is proprietary. 

* San Francisco (https://ieee-dataport.org/open-access/crawdad-epflmobility)
* Porto (https://www.kaggle.com/datasets/crailtap/taxi-trajectory)

### Preprocessing

Part of the preprocessing is adapted from t2vec and utilizes Julia (we used Version 1.8.5 https://julialang.org/downloads/oldreleases/)

First we convert the source data to a list of trajectory. Then we preprocess the data, by splitting in train, val, test and applying threads. 

Use following scripts:

```
preprocessing
├── porto_to_traj_list.ipynb
├── porto_full_preprocessing.ipynb
```

### Model training

First the grid embeddings have to be pre-trained (we uploaded them as well so you can skip this step).

Run following command in folder model_retrace:

```
python grids_pretraining.py
```

Then we can train the model by running model_training.py in folder model_retrace:

```
python model_training.py
```


### Model evaluation

After training we need to embed the test set, i.e., the modified trajectories, with  

```
python model_embed_trajs.py
```

Then we can run the evaluation script in folder experiments:

```
experiments
├── evaluate_dl_models.ipynb
```

