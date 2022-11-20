model=GAT  # [GAT, GCN, GPN, GraphSage]
pipeline=PN # [PN, GPN, Basic]
dataset=email
mkdir -p logs
python -u main.py --model $model --dataset $dataset --pipeline $pipeline > logs/${pipeline}_${model}_$dataset.log 

## for TENT, model specification will be ignored since only the default settings are included. 