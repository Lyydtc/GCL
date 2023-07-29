import argparse


# general settings
parser = argparse.ArgumentParser()
parser.add_argument("--device", default='cuda:0')
parser.add_argument("--seed", default=2023)
parser.add_argument("--task", default='cls', help='cls, gsl')
parser.add_argument('--data_dir', type=str, help='root directory for the data', default='datasets_new/')
parser.add_argument('--dataset', type=str, default='LINUX',
                    help="IMDB-BINARY, LINUX...")
parser.add_argument("--split", default=0.8)
parser.add_argument("--init_node_encoding", default='OneHot', help='OneHot, RWPE, LapPE')
parser.add_argument("--rwpe_size", default=20)


# ================================= pre train model =====================================

# augment dropout rate
parser.add_argument("--p_node", default=0.2)
parser.add_argument("--p_edge", default=0.2)
parser.add_argument("--p_path", default=0.1)
parser.add_argument("--path_length", default=3)

# embedding_size and dropout
parser.add_argument("--embedding_size", default=32)
parser.add_argument("--dropout", type=float, default=0.2)

# self attention
parser.add_argument('--msa_bias', default=True)
parser.add_argument("--encoder_ffn_size", default=128)

# topk
parser.add_argument("--topk_ratio", default=1)

# SwitchGCN layers
parser.add_argument('--switch', default=False)
parser.add_argument("--nfeat_e", default=8)
parser.add_argument("--n_topk_ratio", default=0.8)
parser.add_argument("--e_topk_ratio", default=0.1)

# align
parser.add_argument("--align", default=False)
parser.add_argument("--align_size", default=12)

# cross attention
parser.add_argument("--n_heads", type=int, default=4)


# ================================= pre train setting =====================================

# pre train
parser.add_argument("--load_pre_model", default=0)
parser.add_argument("--pre_epochs", type=int, default=50)
parser.add_argument('--pre_lr', type=float, default=0.0001)

# ugc loss
parser.add_argument('--T', type=float, default=0.5)
parser.add_argument("--y", default=0.4)


# ================================= downstream model =====================================

# MLP
parser.add_argument("--decoder", default='mlp4', help='mlp4, mlp5')
parser.add_argument("--ds_dropout", type=float, default=0.3)


# ================================= downstream setting =====================================

# training parameters
parser.add_argument("--fine_tuning", default=0)
parser.add_argument("--num_folds", default=10)

parser.add_argument('--epochs', type=int, help='number of training epochs', default=20)
parser.add_argument('--patience', default=100)

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, help="Learning rate.", default=5e-4)
parser.add_argument("--lr_reduce_factor", default=0.5)
parser.add_argument("--lr_schedule_patience", default=800)
parser.add_argument("--min_lr", default=1e-7)
parser.add_argument("--weight_decay", type=float, default=0)


parsed_args = parser.parse_args()
# ================================= cls dataset setting =====================================
if parsed_args.dataset == 'IMDB-BINARY':
    parsed_args.task = 'cls'
    parsed_args.pre_lr = 0.0001
    parsed_args.lr = 0.0001
    parsed_args.pre_epochs = 100
    parsed_args.epochs = 50
    parsed_args.embedding_size = 128
    parsed_args.topk_ratio = 1
    parsed_args.batch_size = 32

if parsed_args.dataset == 'IMDB-MULTI':
    parsed_args.lr = 0.0001
    parsed_args.epochs = 4000
    parsed_args.n_heads = 4
    parsed_args.embedding_size = 64
    parsed_args.batch_size = 128
    parsed_args.pooling_res = 80

if parsed_args.dataset == 'PROTEINS':
    parsed_args.pre_lr = 0.0001
    parsed_args.lr = 0.0001
    parsed_args.pre_epochs = 100
    parsed_args.epochs = 30
    parsed_args.embedding_size = 128
    parsed_args.batch_size = 64

if parsed_args.dataset == 'NCI1':
    parsed_args.lr = 0.0001
    parsed_args.pre_lr = 0.0001
    parsed_args.epochs = 4000
    parsed_args.n_heads = 4
    parsed_args.embedding_size = 64
    parsed_args.pooling_res = 120

if parsed_args.dataset == 'AIDS':
    parsed_args.lr = 0.0001
    parsed_args.n_heads = 2
    parsed_args.GCA_n_heads = 2
    parsed_args.n_channel_transformer_heads = 4


# ================================= gsl dataset setting =====================================
if parsed_args.dataset == 'LINUX':
    parsed_args.task = 'gsl'
    parsed_args.pre_lr = 0.0001
    parsed_args.lr = 0.0001
    parsed_args.lr_schedule_patience = 600
    parsed_args.pre_epochs = 200
    parsed_args.epochs = 200
    parsed_args.embedding_size = 128
    parsed_args.topk_ratio = 1
    parsed_args.batch_size = 32
