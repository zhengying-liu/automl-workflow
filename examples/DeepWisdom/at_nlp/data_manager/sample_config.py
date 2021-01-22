sample_strategy = {
    # 从增量未切分好的数据中切分训练集+采样
    "sample_iter_incremental_with_train_split": {'add_val_to_train': False,
                                                 'update_train': True,
                                                 'use_full': False
                                                 },
    # 从增量未切分好的数据中采样，不切分训练集
    "sample_iter_incremental_no_train_split": {'add_val_to_train': True,
                                               'update_train': True,
                                               'use_full': False},

    # 从全量未切分好的数据中切分训练集+采样
    "sample_from_full_data": {'add_val_to_train': False,
                              'update_train': False,
                              'use_full': True},

    # 从全量的切分好的训练数据中采样
    "sample_from_full_train_data": {'add_val_to_train': False,
                                    'update_train': False,
                                    'use_full': False}
}
