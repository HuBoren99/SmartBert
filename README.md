# SmartBERT
🔗 This repo is Pytorch implemention of [SmartBERT: A Promotion of Dynamic Early Exiting Mechanism for Accelerating
BERT Inference](https://www.ijcai.org/proceedings/2023/0563.pdf).


<p align="center">
  <img src="https://github.com/HuBoren99/SmartBert/assets/133136668/2c7223d5-f8da-4341-8040-aa7f501d3a41" alt="image" width="450"/>
</p>

# Environment
💻 Recommand you to set up a Python virtual environment with the required dependencies as follows:
```python

conda create -n SmartBert python=3.8
pip install -r requirements.txt
```

# Datasets

📖 We conducted experiments using the GLUE dataset(SST-2, MRPC, RTE, QNLI, QQP, MNLI, CoLA). All datasets can be downloaded from [here](https://gluebenchmark.com/tasks), or by running the following Python script.
```python

python download_glue.py
```
# Usage
📜 Command for training and evaluating model:
```python

sh run.sh
```
Please note that if you need to switch datasets, you'll need to modify both the parameters ```--data_dir``` and ```--task_name```.

# Citation
Please cite our paper if you find the method useful:
```
@inproceedings{ijcai2023p563,
  title     = {SmartBERT: A Promotion of Dynamic Early Exiting Mechanism for Accelerating BERT Inference},
  author    = {Hu, Boren and Zhu, Yun and Li, Jiacheng and Tang, Siliang},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on
               Artificial Intelligence, {IJCAI-23}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Edith Elkind},
  pages     = {5067--5075},
  year      = {2023},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2023/563},
  url       = {https://doi.org/10.24963/ijcai.2023/563},
}
```
