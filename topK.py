import heapq
import numpy as np
def getListMaxNumIndex(num_list,topk=3):
    max_num_index=map(num_list.index, heapq.nlargest(topk,num_list))
    min_num_index=map(num_list.index, heapq.nsmallest(topk,num_list))
    return set(list(max_num_index)), set(list(min_num_index))

#top bot k match
def topK(a, b, k=5, m=3, num_user=943):
    results_max = np.zeros(num_user) # 943, 6040
    results_min = np.zeros(num_user)
    for i in range(num_user): # 943
        Max1,Min1 = getListMaxNumIndex(list(a[i]),m)
        Max2,Min2 = getListMaxNumIndex(list(b[i]),k)
        results_max[i] = len(Max1&Max2)/m
        results_min[i] = len(Min1&Min2)/m
    return results_max.mean(), results_min.mean()

# #hit ratio @k
# def hrK(a, b, k=5):
#     # a = pred40
#     # b = pref
#     results_max = np.zeros(943)
#     results_min = np.zeros(943)
#     for i in range(943):
#         Max1,Min1 = getListMaxNumIndex(list(a[i]),k)
#         Max2,Min2 = getListMaxNumIndex(list(b[i]),1)
#         results_max[i] = len(Max1&Max2)
#         results_min[i] = len(Min1&Min2)
#     return results_max.mean()
