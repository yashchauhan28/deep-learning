import numpy as np

data = open('kafka.txt', 'r').read()

chars = list(set(data))
data_size, vocab_size = len(data),len(chars)

char_to_ix = {ch:i for i,ch in enumerate(chars)}
ix_to_char = {ch:i for i,ch in enumerate(chars)}

vector_for_char_a = np.zeros((vocab_size, 1))
vector_for_char_a[char_to_ix['a']] = 1

hidden_size = 100
seq_length = 25
learning_rate = 1e-1

#model parameters
Wxh = np.random.randn(hidden_size,vocab_size)
Whh = np.random.randn(hidden_size,hidden_size)
Why = np.random.randn(vocab_size,hidden_size)
bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))

def lossFunc(inputs, targets, hprev):
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh,xs[t])) + np.dot(Whh, hs[t-1]) + bh
        ys[t] = np.dot(Why,hs[t]) + by
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        loss += -np.log(ps[t][targets[t],0])
    
    dWxh,dWhh,dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(dy,hs[t].T)
        dby += dy
        #back propagate
        dh = np.dot(Why.T, dy) + dhnext # backprop into h                                                                                                                                         
        dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity                                                                                                                     
        dbh += dhraw #derivative of hidden bias
        dWxh += np.dot(dhraw, xs[t].T) #derivative of input to hidden layer weight
        dWhh += np.dot(dhraw, hs[t-1].T) #derivative of hidden layer to hidden layer weight
        dhnext = np.dot(Whh.T, dhraw)
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients                                                                                                                 
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h,seed_ix,n):
    x= np.zeros((vocab_size,1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
        h = np.tanh(np.dot(Wxh,x)) + np.dot(Whh,h)
        y = np.dot(Why,h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size),p = p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    txt = ''.join(ix_to_char[ix] for ix in ixes)
    print ('----\n %s \n -----' % (txt, ))
hprev = np.zeros((hidden_size, 1))
sample(hprev,char_to_ix['a'],200)

       