import torch

y = torch.tensor([
    [
        [1, 2, 3],
        [4, 5, 6]
    ],
    [
        [1, 2, 3],
        [4, 5, 6]
    ],
    [
        [1, 2, 3],
        [4, 5, 6]
    ]
])
print(y)   # torch.Size([3, 2, 3])
b = torch.sum(y, dim=0)
print(b)
b1 = torch.sum(y, dim=0,keepdim=True)
print(b1)

c = torch.sum(y,dim=1)
print(c)
c1 = torch.sum(y,  dim =2)
print(c1)