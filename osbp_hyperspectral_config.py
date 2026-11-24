import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='OSBP Hyperspectral Openset-DA')
    
    # 基本参数
    parser.add_argument('--log_name', type=str, default='OSBP_HS')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seed_list', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # 高光谱数据参数
    parser.add_argument('--source_dataset', choices=['Houston13_7gt', 'Houston18_7gt', 'PaviaU_7gt','Yancheng_ZY','Houston13_7gt'], default='PaviaU_7gt')
    parser.add_argument('--target_dataset', choices=['Houston18_7gt', 'Houston13_7gt','PaviaC_OS','Yancheng_GF','Houston18_7gt'], default='PaviaC_OS')
    parser.add_argument('--source_known_classes', type=list, default=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument('--target_known_classes', type=list, default=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument('--target_unknown_classes', type=list, default=[9])
    
    # 训练参数
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--patch', type=int, default=15)
    parser.add_argument('--train_num', type=int, default=180)
    parser.add_argument('--few_train_num', type=int, default=140)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--warm_up_epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float)
    
    # OSBP特定参数
    parser.add_argument('--th', type=float, default=0.5, help='threshold for unknown class')
    parser.add_argument('--grl-rampup-epochs', default=20, type=int, help='GRL rampup epochs')
    parser.add_argument('--rampup_epochs', default=20, type=int, help='rampup epochs')
    parser.add_argument('--log-interval', type=int, default=10)
    
    # 其他参数
    parser.add_argument('--draw', type=str, default='True')
    
    args = parser.parse_args()
    
    # 根据数据集设置参数
    if args.source_dataset == 'Houston13_7gt':
        args.source_known_classes = [1, 2, 3, 4, 5, 6]
        args.target_known_classes = [1, 2, 3, 4, 5, 6]
        args.target_unknown_classes = [7]
        if hasattr(args, 'seed') and args.seed != 0:
            args.seed_list = [args.seed]
        else:
            args.seed_list = [15, 3407, 1, 2023, 100]   
    elif args.source_dataset == 'playground_morning':
        args.source_known_classes = [1, 2, 3, 4, 5, 6]
        args.target_known_classes = [1, 2, 3, 4, 5, 6]
        args.target_unknown_classes = [7,8]
        #args.seed_list = [1, 4, 41, 47, 80]
       # args.seed_list = [40,55,1111,48,99]
        args.seed_list = [55,99,3407]    
    elif args.source_dataset == 'PaviaU_7gt':
        args.source_known_classes = [1, 2, 3, 4, 5, 6, 7]
        args.target_known_classes = [1, 2, 3, 4, 5, 6, 7]
        args.target_unknown_classes = [9]
        if hasattr(args, 'seed') and args.seed != 0:
            args.seed_list = [args.seed]
        else:
            args.seed_list = [58]
    elif args.source_dataset == 'Yancheng_ZY':
        args.source_known_classes = [1, 6, 7]
        args.target_known_classes = [1, 6, 7]
        args.target_unknown_classes = [2,3,4,5]

        args.seed_list = [15, 18, 19, 24, 37, 50, 51, 62, 77, 80]   
    return args
