#docker run --privileged --rm -it -e DOCKER_NET_HOST=172.17.0.1 -v /var/run/docker.sock:/var/run/docker.sock -v /Users/pal004/Desktop/MachineLearning/reinforcement-learning/universe/universe:/usr/local/universe universe python
# exec (open ('FromCaroline_TTTPlay.py').read())
import gym
import universe
import numpy as np
import os
import _pickle as pickle
import time
#from universe.wrappers import BlockingReset  # meh, simulation seems a little lessy buggy using this wrapper

os.environ["OPENAI_REMOTE_VERBOSE"] = "0" # get rid of the annoying messages until we need htem
H = 200 #hidden layer neurons
backprop_batch_size = 20 # number of episodes before back prop
rms_batch_size = 200 # number of episodes before applying gradients with rms optimizer
save_every = 200 # save pickle file every 20 episodes
learning_rate = 1e-3
gamma = 0.99 #discount factor for reward
decay_rate = 0.99 #decay factor for RMSProp leaky sum of grad^2
savefile = 'saveTTT.p'
resume = True # resume from previous checkpoint?
render = True

#model initialization
D = 9 #input dimension
O = 9 #number of output neurons

if resume:
  print ("loading model where previously left off training")
  model = pickle.load(open(savefile, 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H,D)/np.sqrt(D)  #xavier initialization, though not sure why Andre left out the '/ 2' part in his Pong example...
  model['W2'] = np.random.randn(O,H)/np.sqrt(H)
#  model['W1'] = np.random.randn(H,D)/np.sqrt(D / 2)  #xavier initialization, though not sure why Andre left out the '/ 2' part in his Pong example...
#  model['W2'] = np.random.randn(O,H)/np.sqrt(H / 2)

grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

def softmax(x):
    """Compute softmax values for each sets of scores in x.
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()"""
    return np.exp(x)/np.sum(np.exp(x), axis=0)

def policy_forward(x):
    h = np.dot(model['W1'], x) #calc z2
    h[h<0] = 0 # ReLU nonlinearity #calc g(z2)?
    outputactivations = np.dot(model['W2'], h)  #calc z3
    p = softmax (outputactivations) 
    return p, h # return probability of taking action 2, and hidden state

def prepro(I):
  smallview = np.zeros (9);

  # ugly, clean up eventually
  smallview[0] = 0 if I['vision'][141][43][0] == 255 else 1
  if (smallview[0] == 0):
     smallview[0] = 0 if I['vision'][159][43][0] == 255 else -1
  smallview[1] = 0 if I['vision'][141][88][0] == 255 else 1
  if (smallview[1] == 0):
     smallview[1] = 0 if I['vision'][159][88][0] == 255 else -1
  smallview[2] = 0 if I['vision'][141][133][0] == 255 else 1
  if (smallview[2] == 0):
     smallview[2] = 0 if I['vision'][159][133][0] == 255 else -1
  smallview[3] = 0 if I['vision'][186][43][0] == 255 else 1
  if (smallview[3] == 0):
     smallview[3] = 0 if I['vision'][204][43][0] == 255 else -1
  smallview[4] = 0 if I['vision'][186][88][0] == 255 else 1
  if (smallview[4] == 0):
     smallview[4] = 0 if I['vision'][204][88][0] == 255 else -1
  smallview[5] = 0 if I['vision'][186][133][0] == 255 else 1
  if (smallview[5] == 0):
     smallview[5] = 0 if I['vision'][204][133][0] == 255 else -1
  smallview[6] = 0 if I['vision'][231][43][0] == 255 else 1
  if (smallview[6] == 0):
     smallview[6] = 0 if I['vision'][249][43][0] == 255 else -1
  smallview[7] = 0 if I['vision'][231][88][0] == 255 else 1
  if (smallview[7] == 0):
     smallview[7] = 0 if I['vision'][249][88][0] == 255 else -1
  smallview[8] = 0 if I['vision'][231][133][0] == 255 else 1
  if (smallview[8] == 0):
     smallview[8] = 0 if I['vision'][249][133][0] == 255 else -1

  print ("Seeing:\n%s | %s | %s\n%s | %s | %s\n%s | %s | %s\n\n" % (tuple (smallview)))
  return smallview

# 's' is the array of softmax proabilities.  Return the index after a random roll
def RollSoftmax (s):
   randomroll = np.random.uniform(0,1)

   cum_s = 0
   for myindex in range (9):
      cum_s += s[myindex] #s has three duplicate values per row
      if (randomroll <= cum_s):
         return myindex;
   print ("warning cum_s is oddly high at %f at %s" % (cum_s, s))
   return 'error';

# global click settings
arrowoffset = 5 # don't click in the way of our secret pixels meh..
clickcoords = np.zeros ((9,2))
clickcoords[0] = [43, 159+arrowoffset];
clickcoords[1] = [88, 159+arrowoffset];
clickcoords[2] = [133, 159+arrowoffset];
clickcoords[3] = [43, 204+arrowoffset];
clickcoords[4] = [88, 204+arrowoffset];
clickcoords[5] = [133, 204+arrowoffset];
clickcoords[6] = [43, 249+arrowoffset];
clickcoords[7] = [88, 249+arrowoffset];
clickcoords[8] = [133, 249+arrowoffset];

# ttcell is the index (in book-reading form) from 0-9 on what cell to make a click action on
# returns: click action for that cell
def fauxclick (ttcell):
#   print("i will click in square", ttcell," at coords ", clickcoords[ttcell][0], clickcoords[cell][1])
   return ttcell, [[universe.spaces.PointerEvent(clickcoords[ttcell][0], clickcoords[ttcell][1], buttonmask=0),
                    universe.spaces.PointerEvent(clickcoords[ttcell][0], clickcoords[ttcell][1], buttonmask=1),
                    universe.spaces.PointerEvent(clickcoords[ttcell][0], clickcoords[ttcell][1], buttonmask=0)]]
   
def discount_rewards(r):
  '''take 1D float array of rewards and compute discounted reward'''
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 #reset sum
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def policy_backward(eph, epdlogp):
    dW2 = np.dot(eph.T, epdlogp)
    dh = np.dot(epdlogp, model['W2'])
    
    dh[eph<=0] = 0 #backprop relu
    dW1 = np.dot(dh.T, epx)
    return {'W1':dW1,'W2':dW2.T}

xs,hs,dlogps,drs,lossarray, rollarray = [],[],[],[],[],[]
batch_xs,batch_hs,batch_dlogps,batch_drs,batch_lossarray,batch_winarray,batch_rollarray  = [],[],[],[],[],[],[] # reset per-backprob batch array memory
episode_number = 0

#env = BlockingReset (gym.make('wob.mini.TicTacToe-v0'))
env = gym.make('wob.mini.TicTacToe-v0')
env.configure(remotes=1)
observation_n = env.reset()[0] # reset env


# ugly universe work-around to help make sure board resets properly
while (observation_n == None):
   time.sleep (0.3)
   stepoutput = env.step ([[]])
   observation_n, reward_n, done_n = [i[0] for i in stepoutput[0:3]] # ugly but allows assignment on one line (assumes one environment)
   if (observation_n == None):
      continue
   workaround = prepro (observation_n)
   if (np.sum (np.abs (workaround)) != 0 and np.sum (np.abs (workaround)) != 1.0):
      print ("Resetting again because observation did not sum to 0 or 1 for some odd reason\n")
      time.sleep (0.3) # artifical sleep to avoid quick double clicks and blue squares.. smh
      observation_n = env.reset()[0] # reset env


while True:
   if render: env.render()

   print ("Prepro that is about to be used before forward pass")
   fewerPixels = prepro(observation_n)
   numberOfXOs = np.sum (np.abs (fewerPixels))
#        fewerPixels = (fewerPixels - np.mean (fewerPixels)) / 256   # mean center and squish
   sm_prob, h = policy_forward(fewerPixels)  # feed forward network, get softmax probability and save off hidden layers for efficiency
   rolled = RollSoftmax(sm_prob) # lets roll the device giving the probabilities in the feed forward pass

   # temp speed hack to force a roll in a non-occupied slot
   maxrolls = 10 # only allow up to 10 retries, as it gets trained a lot the network will become very confident and could hang if this isn't in
   while (fewerPixels[rolled] != 0 and maxrolls > 0):
      rolled = RollSoftmax(sm_prob) # lets roll the device giving the probabilities in the feed forward pass
      maxrolls -= 1

   print ("going to click on %d" % (rolled))
   thisSq, action_n = fauxclick(rolled)
   
   # assume that the action rolled led to winning the game (the reward discounting will invert this if needed later down the road before backprop)
   # and calculate loss (though we don't explicitely use this computation during bck-prop - only the gradient), We should be able to graph this to make sure the loss is falling over time
   y = np.zeros(9)
   y[thisSq] = 1
   loss = np.sum(np.log(sm_prob)*y)*-1; # cross-entropy loss of soft-max

   time.sleep (0.6) # artifical sleep to avoid quick double clicks and blue squares.. smh
   stepoutput = env.step (action_n)
   observation_n, reward_n, done_n = [i[0] for i in stepoutput[0:3]] # ugly but allows assignment on one line (assumes one environment)
   

   # another hack thing to work-around async step + universe bugs
   if (done_n == True and reward_n == 0):
      print ("Going down weird workaround AAA")
      time.sleep (0.6) # artifical sleep to avoid quick double clicks and blue squares.. smh
      stepoutput = env.step ([[]])
      observation_n, reward_n, done_n = [i[0] for i in stepoutput[0:3]] # ugly but allows assignment on one line (assumes one environment)
      done_n = True

   # another hack thing to work-around async step + universe bugs
   if (done_n != True):
      print ("Going down weird workaround BBB")
      time.sleep (0.6) # artifical sleep to avoid quick double clicks and blue squares.. smh
      while True:
         stepoutput = env.step ([[]])
         observation_n, reward_n, done_n = [i[0] for i in stepoutput[0:3]] # ugly but allows assignment on one line (assumes one environment)
         if (done_n != True):
            print ("Printing observation from BBB")
            tmp = prepro(observation_n)
            numberOfXOs_v2 = np.sum (np.abs (tmp))
            if (numberOfXOs_v2 - numberOfXOs == 2):
               print ('Ending game early because %d - %d is 2' % (numberOfXOs_v2, numberOfXOs))
               break;
         else:
            print ("Ending early because done_n is True in BBB")
            break;

   # another hack thing to work-around async step + universe bugs
   if (done_n == True and reward_n == 0):
      print ("Going down weird workaround CCC")
      time.sleep (0.6) # artifical sleep to avoid quick double clicks and blue squares.. smh
      stepoutput = env.step ([])
      observation_n, reward_n, done_n = [i[0] for i in stepoutput[0:3]] # ugly but allows assignment on one line (assumes one environment)
      done_n = True

# For debugging only
   print ("Reward of: %f after moving to %d" % (reward_n, rolled))
   if (done_n == True):
      print ("Finished individual game with reward of: %f\nAfter step seeing is:\n" % (reward_n))
      if (observation_n != None):
         prepro (observation_n) # just for printing/debugging

   # speed hack to speed up training so our player doesn't play more than 5 moves.  this should (hopefully) indirectly discouarge behaviors of clicking on already-occupied squares
   # should be able to remove this once we confirm it's learning
   if (len (drs) >= 5 and done_n != True):
      print ("played too many turns lets end game, speed back")
      reward_n = -1 
      done_n = True
#      raise Exception ("shouldn't ever get here now due to other speed hack to make sure you don't roll in any occupied square")

   dlogps.append(y - sm_prob) # gradient of softmax loss (or probability maximization in this case)
   xs.append(fewerPixels)  # observation before the last step (input neurons)
   hs.append(h)  # hidden state of feedforward before last step
   drs.append(reward_n)  # record reward after step
   lossarray.append (loss) # loss of previous step
   rollarray.append (rolled) # keep track of what square we played (for debugging purposes)
#   print ("loss is: %s" % (loss))


   if done_n == True:  # episode finished
       if (reward_n == 0.0):
          failed = True
          print ("VERY WEIRD SCENARIO2, done but no reward?  unfortunately after manual inspection this doesn't seem to represent a tie but rather a weird simulation thing.. lets ignore this game")
       elif (np.std (drs) == 0): # bug-workaround, only calculate if std is not 0 due to Universe telling us we are done when we're not
          failed = True
          print ("VERY WEIRD SCENARIO, hmm... preventing nan's by workaround:  %s" % (drs))
       else:
          failed = False
          episode_number += 1
          if (reward_n > 0):
             batch_winarray.extend (['win'])
          else:
             batch_winarray.extend(['loss'])
       
          batch_xs.extend (xs)
          batch_hs.extend (hs)
          batch_dlogps.extend (dlogps)
          batch_drs.extend (drs)
          batch_lossarray.extend (lossarray)
          batch_rollarray.extend (rollarray)

       xs,hs,dlogps,drs,lossarray, rollarray = [],[],[],[],[],[] # reset per-game array memory.  This gets cleared even/especially on the weird scenario errors
       
       # consider backprop_batch_size number of games at a time before doing back prop.  this way there is some diversity in the rewards prior to mean normalization of the discounted rewards
       if failed == False and episode_number % backprop_batch_size == 0:
          epx = np.vstack(batch_xs)
          eph = np.vstack(batch_hs)
          epdlogp = np.vstack(batch_dlogps)
          epr = np.vstack(batch_drs)

          # compute discounted reward backwards through time
          discounted_epr = discount_rewards(epr)

          # standardize the reward to unit normal
          discounted_epr -= np.mean(discounted_epr)
          discounted_epr /= np.std(discounted_epr)
          epdlogp *= discounted_epr  # modulate gradient with advantage

          grad = policy_backward(eph, epdlogp)
#          raise Exception ('test44')

          # accumulate grad over batch (gradient ascent versus descent I believe)
          for k in model: grad_buffer[k] += grad[k]
          if (np.sum(np.isnan(grad_buffer['W1'])) > 0):
             raise Exception ('some reason grad_buffer has nans, exception....should never get here so just making sure')

          # print frequency of wins/losses after the batch of games
          uniq, counts = np.unique (batch_winarray, return_counts=True)
          print (np.asarray((uniq, counts)).T) 

          batch_xs,batch_hs,batch_dlogps,batch_drs,batch_lossarray,batch_winarray,batch_rollarray  = [],[],[],[],[],[],[] # reset per-backprob batch array memory

       if failed == False and episode_number % rms_batch_size == 0:
           for k, v in model.items():
               g = grad_buffer[k]  # gradient
               rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
               model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
               grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

       if failed == False and episode_number % save_every == 0: pickle.dump(model, open(savefile, 'wb'))

       print ("Resetting board")
       observation_n = env.reset()[0] # reset env

       # ugly universe work-around to help make sure board resets properly
       while (observation_n == None):
          time.sleep (.3)
          stepoutput = env.step ([[]])
          observation_n, reward_n, done_n = [i[0] for i in stepoutput[0:3]] # ugly but allows assignment on one line (assumes one environment)
          if (observation_n == None):
             continue
          workaround = prepro (observation_n)
          if (np.sum (np.abs (workaround)) != 0 and np.sum (np.abs (workaround)) != 1.0):
             print ("Resetting again because observation did not sum to 0 or 1 for some odd reason\n")
             time.sleep (.3) # artifical sleep to avoid quick double clicks and blue squares.. smh
             observation_n = env.reset()[0] # reset env
