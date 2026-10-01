import torch

m = torch.nn.Dropout(p=0.2)

input_ = torch.ones(10)

match_count = 0
for i in range(100):
    torch.manual_seed(123)
    a = m(input_)
    
    torch.manual_seed(123)
    b = m(input_)
    
    match_count += int(torch.equal(a,b))

print(match_count)