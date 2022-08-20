class Layer: 
    def __init__(self):
        self.input = None 
        self.output = None 
    
    def forward(self, x):
        pass 

    def backward(self, delta_y, lr):
        pass