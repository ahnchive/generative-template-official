import matplotlib.pyplot as plt
import numpy as np

from torch.distributions import Categorical
from torch.nn.functional import softmax

def get_prob_and_entropy(valuelist, T=0.1, scalelist=None, use_softmax=True):


    valuelist = torch.Tensor(valuelist)
    
    if scalelist is not None:
        raise NotImplementedError
        valuelist = valuelist/scalelist #TODO check this this is elementwise 
    
    if use_softmax:
        problist = softmax((valuelist)/T, dim=-1).numpy()
    else:
        problist = valuelist.numpy()
        
    entropy = Categorical(probs = torch.Tensor(problist)).entropy().cpu().numpy().item()
    return problist, entropy

def get_entropy_class(entropy):
    data = entropy.flatten()
    # data = entropy.flatten()
    plt.hist(data, weights=np.ones(len(data)) / len(data)) 

    #########
    # get batch indices for low and high entropy
    ###########
    value_low = np.percentile(data, 20)
    value_high = np.percentile(data, 80)

    # higher the entropy, lower confidence
    low_entropy = np.unique(np.where(entropy<value_low)[0])
    high_entropy = np.unique(np.where(entropy>value_high)[0]) 
    mid_entropy = np.unique(np.where((entropy>value_low) & (entropy<value_high))[0])
    print('# low ', value_low,  len(low_entropy), '# high ',value_high, len(high_entropy))
    print('average value ', data.mean())
    plt.axvline(x=value_low, color='red')
    plt.axvline(x=value_high, color='red')
    plt.show()

    # entropy class
    entlist_class =[]
    for e in entropy:
        if e < value_low:
            entlist_class.append('low')
        elif value_low<= e < value_high:
            entlist_class.append('mid')
        elif value_high<=e:
            entlist_class.append('high')
        else:
            raise Error
    entlist_class= np.array(entlist_class)
    print(entlist_class.shape)
    
    return entlist_class