import torch

n = 10
m = 2

# a = torch.ones((m, 1, 4, 4))
# b = torch.ones((n, 4, 4))
# print((a @ b).size())
# print(a @ b)

# a = torch.ones((m, n, 6))
# b = torch.ones((m, 6, 6))
# print((a @ b).size())


a = torch.ones((m, 1, 6, 6))
b = torch.ones((m, n, 6, 1))
print((a @ b).size())