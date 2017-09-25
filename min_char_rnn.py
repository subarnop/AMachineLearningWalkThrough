"""
Minimal Character Vanilla RNN
"""
import numpy as np
import re

#data I/O
data = open('data/eminem_lyrics.txt','r').read() #read simple text file
#chars = list(set(data))
chars = re.split(' ',data)
data = chars
data_size, vocab_size = len(data), len(chars)
print('data has', data_size, 'charecters, ',  vocab_size, ' unique charecters.')
char_to_ix = {ch:i for i,ch in enumerate(chars)} #mapping from index to charecter
ix_to_char = {i:ch for i,ch in enumerate(chars)} #mapping from charecter to index
print(char_to_ix)
#hyperparmeters
hidden_size = 100 # size of hidden layers of neurons
seq_length = 25 #Number of steps to unroll RNN
learning_rate = 1e-10

#model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 #input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 #hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 #hidden to output
bh  = np.zeros((hidden_size, 1)) #hidden bias
by  = np.zeros((vocab_size, 1)) #output bias

def lossFun(inputs, targets, hprev):
    """
    inputs, targets are both list of integers.
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    #forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size,1)) #encode in 1-of-k representation
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t])) + np.dot(Whh, hs[t-1]+bh)
        ys[t] = np.dot(Why, hs[t]) + by
        ps[t] = np.exp(ys[t])/np.sum(np.exp(ys[t])) #probabilities for next char
        loss += -np.log(ps[t][targets[t],0]) #softmax cross-entropy loss

    #backward pass
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext # backprop into h
        dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Whh.T, dhraw)
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes


n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby =np.zeros_like(bh), np.zeros_like(by) #Memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length #loss at iteration 0

while True:
    #prepare inputs (sweeping from left to right in steps seq_length)
    if p+seq_length+1 >= len(data) or n==0:
        hprev = np.zeros((hidden_size, 1)) #reset RNN Memory
        p = 0 #Goto start of data
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    #sample from model
    if n%100 == 0:
        sample_ix = sample(hprev, inputs[0], 200)
        txt = ' '.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n',txt,'\n----')

    #forward seq_length charecters through the net and fetch gradient
    loss, dWxh, dWhh, dWhy, dbh, dby, hprevs = lossFun(inputs, targets, hprev)
    smooth_loss = smooth_loss*0.999 + loss*0.001
    if n%100 == 0:
        print('iter', n, '\t,loss:', smooth_loss)
        #print progress

    #parameters update with Adagrad
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                  [dWxh, dWhh, dWhy, dbh, dby],
                                  [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)#Adagrad Upadte

    p += seq_length #move data pointer
    n += 1 #iteration counter
