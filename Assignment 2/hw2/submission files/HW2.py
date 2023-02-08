#!/usr/bin/env python
# coding: utf-8

# Task 1: Vocabulary Creation 
# 
# - For the vocabulary creation task we first read the train dataset file and create a dataframe of it and replace all words with unknown tag for which the frequency of the particular word is below the threshold
# - Secondly we use the frequency as the parameter to sort the words in the dataframe and then we extract all the unique words in tha ascending order of the dataframe and then export it to the csv file having columns 'name', 'position of the word', 'frequency'

# In[38]:


import pandas as pd
import json

df = pd.read_csv('./data/train', sep='\t', names=['idx', 'name', 'tag'])
df['frequency'] = df['name'].map(df['name'].value_counts())
def replace_unk(entry):
    if entry['frequency']<=2:
        return '<unk>'
    else:
        return entry['name']

df['name'] = df.apply(lambda entry: replace_unk(entry), axis=1)

# Sort by descending freq
df = df.sort_values(by=['frequency'], ascending=False)
# Return a Series containing counts of unique values.
df_counted = df['name'].value_counts().reset_index()
df_counted.columns = [''] * len(df_counted.columns)

df_counted.columns = ['name', 'frequency']
df_unknown = df_counted[df_counted['name']=='<unk>']

unk_idx = df_counted[df_counted['name']=='<unk>'].index

df_counted = df_counted.drop(index=unk_idx)
df_counted = pd.concat([df_unknown, df_counted])
df_counted = df_counted.reset_index()
df_counted['index'] = df_counted.index+1
columns_titles = ["name","index", "frequency"]
df_counted=df_counted.reindex(columns=columns_titles)
df_counted.to_csv("vocab.txt", sep="\t", header=None, index=False)

print("What is the selected threshold for unknown words replacement?", 2)
print("What is the total size of your vocabulary?", len(df_counted))
print("What is the total occurrences of the special token '<unk>' after replacement?", int(df_unknown['frequency']
                                                                                           [df_unknown['name']=='<unk>']))



# Task 2: Model Learning
# 
# - In this task we create transition matrix and emmission matrix and after creating both the matrices we then convert the matrix to a python dictionary for faster retrieval of data.
# - transition matrix is of size: length of all_tags * length of all_tags
# - row represents the tag at the previous state and the column represents the tag at the current state for which we want to calculate the transition probability
# - Emmission matrix is of size:  length of all_tags * length of vocabualry
# - row represents the current tag at the previous state and the column represents the vocab word for which we want to calculate the emmission probability

# In[12]:


df = pd.read_csv('./data/train', sep='\t', names=['idx', 'name', 'tag'])
df['frequency'] = df['name'].map(df['name'].value_counts())
df['name'] = df.apply(lambda entry: replace_unk(entry), axis=1)
df_pos = pd.DataFrame(df['tag'].value_counts()).reset_index()
df_pos.columns = [''] * len(df_pos.columns)
df_pos.columns = ['tag', 'count']
all_tags = list(df_pos['tag'])

all_sentences = []
temp_sentence = []
for i in range(len(df)):
    if df.loc[i]['idx']==1 and i!=0:
        all_sentences.append(temp_sentence)
        temp_sentence =[]
    temp_sentence.append((df.loc[i]['name'], df.loc[i]['tag']))
    


# Transition Matrix code
transition_matrix = [[0 for j in range(len(all_tags))] for i in range(len(all_tags))]
tag_freq = {} # format: key = <TAG>, value = <tag_freq>
def generate_transition_matrix():
    # Calculate tag frequency
    for sentence in all_sentences:
        for i in range(len(sentence)):
            curr_tag = sentence[i][1]
            if curr_tag not in tag_freq:
                tag_freq[curr_tag]=1
            else:
                tag_freq[curr_tag]+=1
    
    # 1. Calculate the number of transitions from one tag to another for each sentence
    for sentence in all_sentences:
        for i in range(1, len(sentence)):
            curr_tag_index = all_tags.index(sentence[i][1])
            prev_tag_index = all_tags.index(sentence[i-1][1])
            transition_matrix[prev_tag_index][curr_tag_index]+=1
    
    # 2. Calculate the transition probabilities for each transition
    for i in range(len(transition_matrix)):
        for j in range(len(transition_matrix[0])):
            prev_tag_index = i
            prev_tag_count = tag_freq[all_tags[i]]
            transition_matrix[i][j]/=prev_tag_count
            
# Emmission Matrix code
vocabulary = list(df_counted['name'])
emmission_matrix = [[0 for j in range(len(vocabulary))] for i in range(len(all_tags))]
def generate_emmission_matrix():    
    # 1. Calculate the number of transitions from one tag to another for each sentence
    for sentence in all_sentences:
        for i in range(len(sentence)):
            curr_word_index = vocabulary.index(sentence[i][0])
            prev_tag_index = all_tags.index(sentence[i][1])
            emmission_matrix[prev_tag_index][curr_word_index]+=1
    
    # 2. Calculate the transition probabilities for each transition
    for i in range(len(emmission_matrix)):
        for j in range(len(emmission_matrix[0])):
            prev_tag_index = i
            prev_tag_count = tag_freq[all_tags[i]]
            emmission_matrix[i][j]/=prev_tag_count


generate_transition_matrix()
generate_emmission_matrix()
    


# - Below is the Code for conversion of transition and emmission matrix to respective dictionary
# - For the transition dictionary each key represents a tuple in which  first value is the previous tag and the second value of the tuple is the current tag and the value of the key represents the transition probability from the previous tag to the current tag
# - Likewise for the emmission dictionary each key represents a tuple in which  first value is the current tag and the second value of the tuple is the vocab word from which the current tag is pointed to, and the value of the key represents the emmission probability from the current tag to the current vocab word
# 

# In[13]:


# Code for conversion of transition and emmission matrix to respective dictionary


start_tags = {} # maintains the initial tag frequency of all tag.
def starting_transition_prob():
    
    start_tags_total = 0
    start_tags_prob = {}
    
    for i in range(len(all_tags)):
        start_tags[all_tags[i]]=0
        
    for i in range(len(df)):
        if df.loc[i]['idx']==1:
            start_tags_total+=1
            start_tags[df.loc[i]['tag']]+=1
    
    for tag in start_tags:
        start_tags_prob[tag] = start_tags[tag]/start_tags_total
    
    return start_tags_prob

# transition matrix is of dimension: len of all_tags * len of all_tags
def calculate_trans_prob():
    trans_prob_dict = {}
    for i in range(len(transition_matrix)):
        for j in range(len(transition_matrix[0])):
            tag_at_i = all_tags[i]
            tag_at_j = all_tags[j]
            trans_prob_dict['(' + tag_at_i + ', ' + tag_at_j + ')'] = transition_matrix[i][j]
    return trans_prob_dict

# emmission matrix is of dimension: len of all_tags * len of vocabulary
def calculate_emmission_prob():
    emmission_prob_dict = {}
    for i in range(len(emmission_matrix)):
        for j in range(len(vocabulary)):
            tag_at_i = all_tags[i]
            vocab_at_j = vocabulary[j]
            emmission_prob_dict['(' + tag_at_i + ', ' + vocab_at_j + ')'] = emmission_matrix[i][j]
    
    return emmission_prob_dict

# Probability Matrices
trans_prob_dict = calculate_trans_prob()
emmission_prob_dict = calculate_emmission_prob()
start_tags_prob = starting_transition_prob()

total_transition_prob = {}
# Add both transition probs in to the final transition dictionary
for key in start_tags_prob:
    total_transition_prob['(' + '<s>' + ', ' + key + ')'] = start_tags_prob[key]
    
for key in trans_prob_dict:
    total_transition_prob[key] = trans_prob_dict[key]

print("Total transition and emission parameters in the HMM model: ", len(total_transition_prob), ',', len(emmission_prob_dict))
    


# In[14]:


# Dump the dictionaries to the json file
with open('hmm.json', 'w') as f:
    json.dump({"transition": total_transition_prob, "emission": emmission_prob_dict}, f, indent = 4)


# Task 3: Greedy Decoding with HMM
# 
# - In greedy decoding for each transition in the states we calculate the net probability  score till the current tag every time and store the tag which is giving maximum score and the stored tag will help us get the final sequence and accuracy of the greedy decoding algorithm.
# 
# - To handle a situation where a vocab word is not present in the emmission matrix then we use the probability for the unknown words considering the not found word as unknown

# In[15]:


df_dev = pd.read_csv('./data/dev', sep='\t', names=['idx', 'name', 'tag'])
df_dev['frequency'] = df_dev['name'].map(df_dev['name'].value_counts())


all_sentences_dev = []
temp_sentence_dev = []
for i in range(len(df_dev)):
    if df_dev.loc[i]['idx']==1 and i!=0:
        all_sentences_dev.append(temp_sentence_dev)
        temp_sentence_dev =[]
    temp_sentence_dev.append((df_dev.loc[i]['name'], df_dev.loc[i]['tag']))
    
from_tag = None
tag_sequence = []
sentence_scores = [] # score for each tag to word score for each sentence

for sentence in all_sentences_dev:
    curr_sentence_scores = []
    curr_sequence = []
    for i in range(len(sentence)):
        max_score = float('-inf') # initialize max score
        for j in range(len(all_tags)):
            curr_score = 1
            if i==0:
                curr_score *= start_tags_prob[all_tags[j]]
            else:
                curr_score *= trans_prob_dict['(' + from_tag + ', ' + all_tags[j] + ')']
            
            if str('(' + all_tags[j] + ', ' + sentence[i][0] + ')') not in emmission_prob_dict:
                curr_score *= emmission_prob_dict['(' + all_tags[j] + ', ' + '<unk>' + ')']
            else:
                curr_score *= emmission_prob_dict['(' + all_tags[j] + ', ' + sentence[i][0] + ')']
            
            if curr_score>max_score:
                max_score = curr_score
                highest_score_tag = all_tags[j]
        
        from_tag = highest_score_tag
        curr_sequence.append(highest_score_tag)
        curr_sentence_scores.append(max_score)
    sentence_scores.append(curr_sentence_scores)
    tag_sequence.append(curr_sequence)

def accuracy_finder():
    frequency = 0
    cur_tag_freq = 0
    
    for i in range(len(all_sentences_dev)):
        for j in range(len(all_sentences_dev[i])):
            if tag_sequence[i][j]==all_sentences_dev[i][j][1]:
                cur_tag_freq+=1
            frequency+=1
    return cur_tag_freq/frequency

print("Accuracy on the dev data for Greedy Decoding in percent: ", accuracy_finder()*100)


# In[16]:


# Greedy Decoding Test Data
df_test = pd.read_csv('./data/test', sep='\t', names=['idx', 'name', 'tag'])
df_test['frequency'] = df_test['name'].map(df_test['name'].value_counts())


all_sentences_test = []
temp_sentence_test = []
for i in range(len(df_test)):
    if df_test.loc[i]['idx']==1 and i!=0:
        all_sentences_test.append(temp_sentence_test)
        temp_sentence_test =[]
    temp_sentence_test.append(df_test.loc[i]['name'])
    
from_tag_test = None
tag_sequence_test = []
sentence_scores_test = [] # score for each tag to word score for each sentence

for sentence in all_sentences_test:
    curr_sentence_scores = []
    curr_sequence = []
    for i in range(len(sentence)):
        max_score = float('-inf') # initialize max score
        for j in range(len(all_tags)):
            curr_score = 1
            if i==0:
                curr_score *= start_tags_prob[all_tags[j]]
            else:
                curr_score *= trans_prob_dict['(' + from_tag_test + ', ' + all_tags[j] + ')']
            
            if str('(' + all_tags[j] + ', ' + sentence[i] + ')') not in emmission_prob_dict:
                curr_score *= emmission_prob_dict['(' + all_tags[j] + ', ' + '<unk>' + ')']
            else:
                curr_score *= emmission_prob_dict['(' + all_tags[j] + ', ' + sentence[i] + ')']
            
            if curr_score>max_score:
                max_score = curr_score
                highest_score_tag = all_tags[j]
        
        from_tag_test = highest_score_tag
        curr_sequence.append(highest_score_tag)
        curr_sentence_scores.append(max_score)
    sentence_scores_test.append(curr_sentence_scores)
    tag_sequence_test.append(curr_sequence)

# Creating the final output format
final_output = []

for i in range(len(all_sentences_test)):
    idx = 1
    curr_sentence = []
    for j in range(len(all_sentences_test[i])):
        curr_sentence.append([idx, all_sentences_test[i][j], tag_sequence_test[i][j]])
        idx+=1
    final_output.append(curr_sentence)


# In[17]:


# Generating the final output: greedy.out
with open("greedy.out", 'w') as f:
    for item in final_output:
        for element in item:
            f.write("\n".join([str(element[0])+"\t"+element[1]+"\t"+element[2]]))
            f.write("\n")
        f.write("\n")


# Task 4: Viterbi Decoding with HMM
# 
# - For Viterbi Decoding , we maintain a list/array representing all the scores for the previous state/pos tags.
# - We maintain a Memo or cache memory as a dictionary for storing all indices and pos tags as the key and value as the score or the probability, the cache will continue updating until we find a better transition mapping from one tag and to a vocab word.

# In[25]:


# viterbi decoding for each sentence:

def viterbi_decoding_algo(sentence):
    tag_score_probs = []
    memo={}
    for i in range(len(all_tags)):
        if str('(' + all_tags[i] + ', ' + sentence[0][0] + ')') not in emmission_prob_dict:
            tag_score_probs.append(start_tags_prob[all_tags[i]]*
                                   emmission_prob_dict['(' + all_tags[i] + ', ' + '<unk>' + ')'])
        else:
            tag_score_probs.append(start_tags_prob[all_tags[i]]*
                                   emmission_prob_dict['(' + all_tags[i] + ', ' + sentence[0][0] + ')'])
            
    for i in range(1, len(sentence)):
        temp_tag_score=[float('-inf')]*len(all_tags)
        for j in range(len(all_tags)): # CURRENT
            best_score = float('-inf')
            curr_score = 1
            for k in range(len(tag_score_probs)): # PREVIOUS
                if str('(' + all_tags[j] + ', ' + sentence[i][0] + ')') not in emmission_prob_dict:
                    curr_score = tag_score_probs[k] * trans_prob_dict['(' + all_tags[k] + ', ' + all_tags[j] + ')']* emmission_prob_dict['(' + all_tags[j] + ', ' + '<unk>' + ')']
                
                else:
                    curr_score = tag_score_probs[k] * trans_prob_dict['(' + all_tags[k] + ', ' + all_tags[j] + ')'] * emmission_prob_dict['(' + all_tags[j] + ', ' + sentence[i][0] + ')']
                    
                if best_score<curr_score:
                    best_score=curr_score
                    memo[i, all_tags[j]] = (all_tags[k], curr_score)
            temp_tag_score[j] = best_score
        tag_score_probs = temp_tag_score
    
    return tag_score_probs, memo
                    
                        
                        
each_sentence_cache = []
each_sentence_scores = []

for sentence in all_sentences_dev:
    op = viterbi_decoding_algo(sentence)
    each_sentence_cache.append(op[1])
    each_sentence_scores.append(op[0])
                        
                        


# - We calculated all the best scores for each sentence and stored it to a memo/cache using viterbi decoding, now we propagate backward to find the best pos tag sequence for the corresponding sentence.
# - While propagating backwords we maintain the tag which gives the max score until now and find the next best tag (using the cache/memo) at the previous position, thats how we find the final sequence for that sentence.

# In[46]:


#  Back propagate to find the best sequence

def propagate_bacwards(i):
    final_seq = []
    final_seq_scores = []
    sent_scores = each_sentence_scores[i]
    max_tag = all_tags[sent_scores.index(max(sent_scores))]
    final_seq.append(max_tag)
    sent_memo = each_sentence_cache[i]
    
    for j in range(int(len(each_sentence_cache[i])/len(all_tags)), 0, -1):
        score, max_tag = sent_memo[(j, max_tag)][1], sent_memo[(j, max_tag)][0]
        temp_max = [max_tag]
        temp_max.extend(final_seq)
        final_seq = temp_max
        temp_score = [score]
        temp_score.extend(final_seq_scores)
        final_seq_scores = temp_score
    
    return final_seq, final_seq_scores

final_seq = []
final_seq_scores = []
for i in range(len(each_sentence_scores)):
    op_dev = propagate_bacwards(i)
    final_seq.append(op_dev[0])
    final_seq_scores.append(op_dev[1])

# Accuracy finder
def accuracy_finder_viterbi():
    frequency = 0
    cur_tag_freq = 0

    for i in range(len(all_sentences_dev)):
        for j in range(len(all_sentences_dev[i])):
            if final_seq[i][j]==all_sentences_dev[i][j][1]:
                cur_tag_freq+=1
            frequency+=1
    return cur_tag_freq/frequency

print("Accuracy on the dev data for Viterbi Decoding in percent: ", accuracy_finder_viterbi()*100)

# Testing data Viterbi Algorithm                        
each_sentence_cache_test = []
each_sentence_scores_test = []

for sentence in all_sentences_test:
    op_test = viterbi_decoding_algo(sentence)
    each_sentence_cache_test.append(op_test[1])
    each_sentence_scores_test.append(op_test[0])
    
def propagate_bacwards_testing(i):
    final_seq = []
    final_seq_scores = []
    sent_scores = each_sentence_scores_test[i]
    sent_memo = each_sentence_cache_test[i]
    max_tag = all_tags[sent_scores.index(max(sent_scores))]
    final_seq.append(max_tag)
    
    for j in range(int(len(each_sentence_cache_test[i])/len(all_tags)), 0, -1):
        score, max_tag = sent_memo[(j, max_tag)][1], sent_memo[(j, max_tag)][0]
        temp_max = [max_tag]
        temp_max.extend(final_seq)
        final_seq = temp_max
        temp_score = [score]
        temp_score.extend(final_seq_scores)
        final_seq_scores = temp_score
    
    return final_seq, final_seq_scores

final_seq_test = []
final_seq_scores_test = []
for i in range(len(each_sentence_scores_test)):
    op_test = propagate_bacwards_testing(i)
    final_seq_test.append(op_test[0])
    final_seq_scores_test.append(op_test[1])


# In[47]:


# Generating the final output: viterbi.out
# Creating the final output format
final_output_test = []
for i in range(len(all_sentences_test)):
    idx = 1
    curr_sentence = []
    for j in range(len(all_sentences_test[i])):
        curr_sentence.append([idx, all_sentences_test[i][j], final_seq_test[i][j]])
        idx+=1
    final_output_test.append(curr_sentence)
    
with open("viterbi.out", 'w') as f:
    for item in final_output_test:
        for element in item:
            f.write("\n".join([str(element[0])+"\t"+element[1]+"\t"+element[2]]))
            f.write("\n")
        f.write("\n")


# In[ ]:




