import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="MixGCF")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="amazon",
                        help="Choose a dataset:[amazon,yelp2018,ali]")
    parser.add_argument(
        "--data_path", nargs="?", default="/tmp2/syliu/light_mixgcf/data/", help="Input data path."
    )

    # ===== train ===== # 
    parser.add_argument("--gnn", nargs="?", default="lightgcn",
                        help="Choose a recommender:[lightgcn, ngcf]")
    parser.add_argument('--epoch', type=int, default=300, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=2048, help='batch size in evaluation phase')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization weight, 1e-5 for NGCF')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument("--mess_dropout", type=bool, default=False, help="consider mess dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of mess dropout")
    parser.add_argument("--edge_dropout", type=bool, default=False, help="consider edge dropout or not")
    parser.add_argument("--edge_dropout_rate", type=float, default=0.1, help="ratio of edge sampling")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")

    parser.add_argument("--ns", type=str, default='mixgcf', help="rns,mixgcf")
    parser.add_argument("--K", type=int, default=1, help="number of negative in K-pair loss")

    parser.add_argument("--n_negs", type=int, default=32, help="number of candidate negative")
    parser.add_argument("--pool", type=str, default='concat', help="[concat, mean, sum, final]")

    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=2, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60]',
                        help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument("--context_hops", type=int, default=2, help="hop")

    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=False, help="save model or not")
    parser.add_argument(
        "--out_dir", type=str, default="./weights/yelp2018/", help="output directory for model"
    )
    # ====== save running result ====#
    parser.add_argument("--save_running_log", type=str,default="/tmp2/syliu/light_mixgcf/training_log/")

    return parser.parse_args()

def co_training_parse_args():
    parser = argparse.ArgumentParser(description="Co_MixGCF")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="ali",
                        help="Choose a dataset:[amazon,yelp2018,ali]")
    parser.add_argument(
        "--data_path", nargs="?", default="/tmp2/syliu/light_mixgcf/data/", help="Input data path."
    )

    # ===== train ===== # 
    parser.add_argument("--gnn", nargs="?", default="lightgcn",
                        help="Choose a recommender:[lightgcn]")
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs') #300
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=2048, help='batch size in evaluation phase')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization weight, 1e-5 for NGCF')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument("--mess_dropout", type=bool, default=False, help="consider mess dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of mess dropout")
    parser.add_argument("--edge_dropout", type=bool, default=False, help="consider edge dropout or not")
    parser.add_argument("--edge_dropout_rate", type=float, default=0.1, help="ratio of edge sampling")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")

    parser.add_argument("--ns", type=str, default='mixgcf', help="rns,mixgcf")
    parser.add_argument("--K", type=int, default=1, help="number of negative in K-pair loss")

    parser.add_argument("--n_negs", type=int, default=32, help="number of candidate negative")
    parser.add_argument("--pool", type=str, default='mean', help="[concat, mean, sum, final]")

    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=2, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60]',
                        help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument("--context_hops", type=int, default=2, help="hop")

    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=False, help="save model or not")
    parser.add_argument(
        "--out_dir", type=str, default="/tmp2/syliu/light_mixgcf/model_stored/", help="output directory for model"
    )
    # ====== save running result ====#
    parser.add_argument("--save_running_log", type=str,default="/tmp2/syliu/light_mixgcf/different_co_training_log/")
    parser.add_argument("--pre_train", type=str, default='lightgcn')
    parser.add_argument("--pretrain_epoch", type=int, default=150) #100
    parser.add_argument("--classifier_lr", type=float, default=0.01)
    parser.add_argument("--classifier_momentum", type=float, default=0.9)
    parser.add_argument("--classifier_decay", type=float,default = 1e-4)
    parser.add_argument("--classifier_epochs", type=int,default = 300) #300
    parser.add_argument("--classifier_warm_up", type=int, default=40) 
    parser.add_argument("--classifier_lambda_cot_max", type=int, default=5)
    parser.add_argument("--load_pretrain_model",type=bool, default=False)
    parser.add_argument("--save_address_pretrain_model",type=str, default='/tmp2/syliu/light_mixgcf/model_stored/lightgcn_pretrain_model_ali.ckpt')
    return parser.parse_args()
