import numpy as np

def get_train_instances(train, num_negatives, num_items, num_neg):
    num = train.nnz * (num_neg + 1)
    user_input, item_input, labels = np.zeros(num), np.zeros(num), np.zeros(num)
    idx = 0
    for (u, i) in train.keys():
        # positive instance
        user_input[idx] = u
        item_input[idx] = i
        labels[idx] = 1
        idx += 1
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while ((u,j) in train.keys()):
                j = np.random.randint(num_items)
            user_input[idx] = u
            item_input[idx] = j
            labels[idx] = 0
            idx += 1
    return user_input, item_input, labels