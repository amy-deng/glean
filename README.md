# Glean

This is the source code for paper [Dynamic Knowledge Graph based Multi-Event Forecasting](https://yue-ning.github.io/docs/KDD20-glean.pdf) appeared in KDD20 (research track)

[Songgaojun Deng](https://amy-deng.github.io/home/), [Huzefa Rangwala](https://cs.gmu.edu/~hrangwal/), [Yue Ning](https://yue-ning.github.io/)


## Data

We processed five country based datasets from the ICEWS data. Please find the datasets in this [Google Drive Link](https://drive.google.com/drive/folders/1qrF1e9I8pnVlCRjb-NPiidZCu5oA0NWL?usp=sharing). The dataset folder (e.g., `IND`) can be placed in the folder `data`. A brief introduction of the data file is as follows:
- `quadruple.txt` includes the structured event information ordered by time.
- `text.txt` event summary file, where each row corresponds to the event in `quadruple.txt`
- `stat.txt` includes the number of entities and event types.
- `entity2id.txt` entity string to index mapping
- `relation2id.txt` event type (i.e., relation) string to index mapping
- `quadruple_id.txt` events represented by the index.

## Prerequisites

updating
