
import torch

class Brain(torch.nn.Module):
    def __init__(self, input_shape, output_shape, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        
        super(Brain, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.device = device
        
        self.linear1 = torch.nn.Linear(self.input_shape, 64)
        self.linear2 = torch.nn.Linear(64, 32)
        self.linear3 = torch.nn.Linear(32, 16)
        self.out = torch.nn.Linear(16, self.output_shape)

        
        torch.nn.init.xavier_uniform_(self.linear1.weight) 
        torch.nn.init.xavier_uniform_(self.linear2.weight) 
        torch.nn.init.xavier_uniform_(self.linear3.weight) 
        torch.nn.init.xavier_uniform_(self.out.weight)
        
        
    def forward(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        x = torch.nn.functional.relu(self.linear1(x)) ##Función de activación RELU
        x = torch.nn.functional.relu(self.linear2(x))
        x = torch.nn.functional.relu(self.linear3(x))
        #x = torch.nn.functional.sigmoid(self.out(x))
        x = self.out(x)
        return x
    
