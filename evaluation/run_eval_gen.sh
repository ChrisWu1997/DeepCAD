python collect_gen_pc.py --src $1
python evaluate_gen_torch.py --src "${1}_pc" --n_test $2 -g $3
rm -rf "${1}_pc"
