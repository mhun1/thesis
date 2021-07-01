import torch
from torch import nn

from loss.distance.hausdorff import Hausdorff
from loss.distance.surface import Surface
from loss.region.assymetric import Asymmetric
from loss.region.dice import Dice
from loss.region.tversky import Tversky

d = Dice(apply_nonlin=False)
surf = Surface(device="cpu",apply_non_lin=False)
hd_2 = Hausdorff()
#bd = PureSurface()
tv = Tversky(apply_non_lin=False)
asym = Asymmetric(apply_non_lin=False)
bce = nn.BCEWithLogitsLoss()

y = torch.Tensor([[0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,1,1,1,1,0,0,0],
                  [0,0,0,1,1,1,1,0,0,0],
                  [0,0,0,1,1,1,1,0,0,0],
                  [0,0,0,1,1,1,1,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0]
              ])

x = torch.Tensor([[0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,1,1,0,0,0,0],
                  [0,0,0,0,1,1,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0]
              ])

x_2 = torch.Tensor([[0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,1,1,1,1,1,1,0,0],
                  [0,0,1,1,1,1,1,1,0,0],
                  [0,0,1,1,1,1,1,1,0,0],
                  [0,0,1,1,1,1,1,1,0,0],
                  [0,0,1,1,1,1,1,1,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0]
              ])

# y_small = torch.Tensor([[0,0,0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0,0,0],
#               [0,0,0,0,1,1,0,0,0,0],
#               [0,0,0,0,1,1,0,0,0,0],
#               [0,0,0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0,0,0]
#               ])
# x_small = torch.Tensor([[0,0,0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0,0,0],
#               [0,0,0,0,1,1,0,0,0,0],
#               [0,0,0,0,1,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0,0,0]
#               ])
#
# x_small_2 = torch.Tensor([[0,0,0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0,0,0],
#               [0,0,0,0,1,1,0,0,0,0],
#               [0,0,0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0,0,0]
#               ])

#print(1-d(x,y))

print("CE")
print(bce(x,y))
print(bce(x_2,y))
# print(1-bce(x_small,y_small))
# print(1-bce(x_small_2,y_small))

print("Dice")
print(1-d(x,y))
print(1-d(x_2,y))
# print(1-d(x_small,y_small))
# print(1-d(x_small_2,y_small))

print("Tversky--")
print(1-tv(x,y))
print(1-tv(x_2,y))
# print(1-tv(x_small,y_small))
# print(1-tv(x_small_2,y_small))

print("Asym")
print(1-asym(x,y))
print(1-asym(x_2,y))

print("Surf")
print(surf(x,y))
print(surf(x_2,y))

print("HD")
print(hd_2(x,y))
print(hd_2(x_2,y))

# print(1-asym(x_small,y_small))
# print(1-asym(x_small_2,y_small))

