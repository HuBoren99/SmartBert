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
📜 Command for training and evaluating model on RTE dataset
```python

python main.py \
--cuda_id "1" \
--lamada 0.04 \
--eta1 0.01 \
--eta2 0.05 \
--t1 0.5 \
--t2 0.55 \
--special_training_mode True \
--is_hard_weight True \
--skipped_rate 0.5 \
--seed 50 \
--data_dir "./glue-data/RTE" \
--task_name "RTE" \
--output_path "./output/SmartBERT_{}_{}_{}_{}_{}_{}_{}" \
--do_train True \
--do_eval True \
--model_name_or_path "bert-base-uncased" \
--per_gpu_train_batch_size 16 \
--first_stage_train_nums 5 \
--second_stage_train_nums 4 \
--learning_rate 2e-5 \
--weight_decay 0.01 \
--adam_epsilon 1e-8 \
--warmup_steps 10  \
--log_step 100 \
--max_seq_length 128 \
```
