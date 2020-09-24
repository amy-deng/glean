# Glean

This is the source code for paper [Dynamic Knowledge Graph based Multi-Event Forecasting](https://yue-ning.github.io/docs/KDD20-glean.pdf) appeared in KDD20 (research track)

[Songgaojun Deng](https://amy-deng.github.io/home/), [Huzefa Rangwala](https://cs.gmu.edu/~hrangwal/), [Yue Ning](https://yue-ning.github.io/)


## Data
We processed some country based datasets from the ICEWS data. Please find an example dataset (partial events are kept.) in this [Google Drive Link](https://drive.google.com/drive/folders/1qrF1e9I8pnVlCRjb-NPiidZCu5oA0NWL?usp=sharing). The dataset folder (e.g., `AFG-raw-part`) can be placed in the folder `data`. A brief introduction of the data file is as follows:
- `quadruple.txt` includes the structured event information ordered by time.
- `text.txt` event summary file, where each row corresponds to the event in `quadruple.txt`
- `stat.txt` includes the number of entities and event types.
- `entity2id.txt` entity string to index mapping
- `relation2id.txt` event type (i.e., relation) string to index mapping
- `quadruple_id.txt` events represented by the index.

## Prerequisites
The code has been successfully tested in the following environment. (For older dgl versions, you may need to modify the code)
- Python 3.7.7
- PyTorch 1.6.0
- dgl 0.5.0
- Sklearn 0.23.2
- Pandas 1.1.1

Example commands executed to build a conda environment (Note: we use Ubuntu with Cuda 9.2)
```sh
conda create --name glean python=3.7
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
pip install dgl-cu92
pip install tqdm
conda install scikit-learn
pip install pandas
```

## Getting Started
### Prepare your code
Clone this repo.
```bash
git clone https://github.com/amy-deng/glean
cd glean
```
### Prepare your data
Download the dataset from the given link or prepare your own dataset in a similar format. The folder structure is as follows:
```sh
- glean
	- data
		- NGA
		- AFG
		- your own dataset
	- src
	- presrc
```

### Preprocessing
Run the files in the `presrc` folder in the recommended order. Please check the parameters required for each file to run. Here are some brief instructions.
- `0_build_raw_sets.py` split the raw data into training, validation, and testing sets.
- `1_get_digraphs.py` construct the DGL based event graph data 
- `2_get_history.py` get historical data for training the actor predictor
- `3_get_token_for_embedding_training.py`,`4_get_word_embedding.py` Some steps to get word embedding from the event summary. Any other method can be applied instead.
- `5_build_word_graphs.pmi.py` get word graphs
- `6_get_word_entity_map.py` get entity/event type and word mapping
- `7_get_sub_event_dg_from_entity_g.py`,`8_get_sub_word_g_from_entity_g.py`,`9_get_scaled_tr_dataset.py` construct datasets for training the actor predictor

The processed dataset `AFG-example` in [Google Drive Link](https://drive.google.com/drive/folders/1qrF1e9I8pnVlCRjb-NPiidZCu5oA0NWL?usp=sharing) can be used directly. Note that only 20 event types are randomly selected for actor prediction in this dataset.


### Training and testing
Please run following commands for training and testing. We take the dataset `AFG-example` as the example.

**Event prediction**
```python
python train_event_predictor.py --runs 1 --dp ../data/ --gpu 1  -d AFG-example --seq-len 7
```
**Actor prediction**
```python
python train_actor_predictor.py --runs 1 --dp ../data/ --gpu 1  -d AFG-example --num-r 20 --seq-len 7
```

## Cite

Please cite our paper if you find this code useful for your research:

```
@inproceedings{10.1145/3394486.3403209,
author = {Deng, Songgaojun and Rangwala, Huzefa and Ning, Yue},
title = {Dynamic Knowledge Graph Based Multi-Event Forecasting},
year = {2020},
publisher = {Association for Computing Machinery},
booktitle = {Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery &amp; Data Mining},
pages = {1585–1595},
}
```
