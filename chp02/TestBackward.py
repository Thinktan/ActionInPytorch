import torch

t1 = torch.randn(3, 3, requires_grad=True)
print(t1)

t2 = t1.pow(2).sum()
t2.backward()

print(t1.grad)

t2 = t1.pow(2).sum()
t2.backward()
print(t1.grad)

print(t1.grad.zero_())

del t1, t2

t1 = torch.randn(3, 3, requires_grad=True)
t2 = t1.pow(2).sum()
print(t1)
print(t2)
print(torch.autograd.grad(t2, t1))

del t1, t2
print('----------')
t1 = torch.randn(3, 3, requires_grad=True)
t2 = t1.sum()
print(t2)
with torch.no_grad():
    t3 = t1.sum()

print(t3)
print(t1.sum())
print(t1.sum().detach())
print(t1)