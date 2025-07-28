# This is again written here to import in streamlit app file, so this serves as a file package!
# My ANN Model in class implementation

# Basic methodology in pytorch differes from Tensorflow
# Here we have nn.sequential models but still as a sign of progress and ability to tweak little things in pytorch i'm writting a class, this will be super helpfun and handy when working with advanced projects like llm'scalar
import torch
from torch import nn

class binary_nn(nn.Module):
    def __init__(self, input_size):
        super(binary_nn,self).__init__()
        
        # Our NN Architecture!
        # __init__ has all initiallized parameters in it when we define an object of a class binary_nn
        
        self.hidden_layer1 = nn.Linear(input_size,32)  # our first hidden layer which is a Fully connected or FC layer
        self.hidden_layer2 = nn.Linear(32,16)   #our 2nd FC layer 
        self.hidden_layer3 = nn.Linear(16,8)   # our 3rd FC layer (We can just go ahed with 2 layers which is fine but if your data has patters to learn then more layers capture more patters)
        self.output_layer =  nn.Linear(8,1)    # our 4th FC layer 
        
        # Defining Activation Functions!
        self.relu_activation = nn.ReLU()  # x===> f(x)
        self.sigmoid_activation = nn.Sigmoid()  # Applies sigmoid function to input 
        self.dropout = nn.Dropout(0.3)  #Using Dropouts to prevent overfitting!
        
    def forward(self,x):
        x = self.hidden_layer1(x)
        x = self.relu_activation(x)
        x = self.dropout(x)

        
        x = self.hidden_layer2(x)
        x = self.relu_activation(x)
        x = self.dropout(x)

        
        x = self.hidden_layer3(x)
        x = self.relu_activation(x)
        x = self.dropout(x)

        
        x = self.output_layer(x)
        x = self.sigmoid_activation(x)
        
        return x
    
    