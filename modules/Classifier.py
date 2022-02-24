import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class Classifier(nn.Module):
  def __init__(self, input_dim, number_label, weight_norm = False):
    super(Classifier, self).__init__()
    self.input_dim = input_dim
    self.number_of_label = number_label

    if weight_norm == False:
      self.fc1 = nn.Linear(input_dim, input_dim) #(user dim = 64 and item dim = 64)
      self.fc2 = nn.Linear(input_dim, input_dim//2)
      self.fc3 = nn.Linear(input_dim//2, number_label)
    else:
      self.fc1 = weight_norm(nn.Linear(input_dim, input_dim)) #(user dim = 64 and item dim = 64)
      self.fc2 = weight_norm(nn.Linear(input_dim, input_dim//2))
      self.fc3 = weight_norm(nn.Linear(input_dim//2, number_label))

  def forward(self, x):

    out = F.tanh(self.fc1(x))
    out = F.tanh(self.fc2(out))
    out = self.fc3(out)

    return out