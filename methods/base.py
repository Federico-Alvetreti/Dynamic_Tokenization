
import torch.nn as nn

class Model_Wrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        # Build model 
        self.model = model


    # Return the additional loss 
    def get_loss(self):
        return 0

    # Standard forward 
    def forward(self, input_ids, attention_mask, labels):
        return self.model.forward(input_ids = input_ids,
                                  attention_mask = attention_mask,
                                  labels = labels)