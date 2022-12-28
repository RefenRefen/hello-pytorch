import torch

scalar = torch.tensor(2)

vector = torch.tensor([1, 2, 4, 5.5])

matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])  # 2D needs 2 []

tensor3d = torch.tensor([[[7, 2, 3],
                          [4., 5, 6]],
                         [[7, 8, 9],
                          [7, 2, 3]]])

print(tensor3d.device)
print(tensor3d.shape)
print(tensor3d.min(1))
print(tensor3d.mean())
print(tensor3d.dtype)

tensor3d - tensor3d.float()
print(tensor3d)

vector = torch.tensor([1, 2, 3])
vector = vector.int()
vector = vector.type(torch.int64)

tensor = torch.ShortTensor([1, 2, 3])

vector = vector.type(torch.float16)
print(vector[0])
print(vector[:2])

print(tensor3d[1, ...])

z1 = torch.zeros(4)
z2 = torch.ones(2, 3)
e2 = torch.eye(4, 5)
r1 = torch.rand(4)
rn2 = torch.randn(8, 3)
rp1 = torch.randperm(10)

c = torch.full((9, 9), 4)
d = torch.zeros_like(c)
e = torch.empty(5, 5)
range1 = torch.arange(0, 4, 0.5)
linspace = torch.linspace(0, 5, 8)

round_down = torch.floor(c)
round_up = torch.ceil(c)
round_to_nearest = torch.round(c)
print(e.floor())
e.floor_()
f = torch.log(e)

a = torch.rand(3)
b = torch.rand(3)
dist = torch.dist(a, b, 2)  # norm 2
