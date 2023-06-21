import numpy as np
import torch
from collections import Counter
import pickle as pkl
from torch.autograd import Variable

UNK = 0  # unknow word-id
PAD = 1  # padding word-id

def seq_padding(X, padding=0):
    """
    add padding to a batch data 
    """
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])    
    
class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg, if_inverse = 'Forward', pad=0):

        # convert words id to long format.  
        src = torch.from_numpy(src).long()
        trg = torch.tensor(trg).float()
        if if_inverse == 'Forward':
            self.src = src # source is structure
            self.trg = trg # target is spectrum
            # # get the padding postion binary mask
            # # change the matrix shape to  1×seq.length
            self.src_mask = (src != pad).unsqueeze(-2)
            self.ntokens = (self.src != pad).data.sum()
        elif if_inverse == 'Inverse':
            self.trg = src[:, :-1] # target is structure
            # decoder target from trg 
            self.trg_y = src[:, 1:]
            self.src = trg.unsqueeze(-2) # source is spectrum
            # # get the padding postion binary mask
            # # change the matrix shape to  1×seq.length
            self.src_mask = None
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg != pad).data.sum()
        else:
            raise NotImplementedError
    # Mask 
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask # subsequent_mask is defined in 'decoder' section.

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class PrepareData:
    def __init__(self, train_file, train_spec_file, train_ratio, dev_file, dev_spec_file, BATCH_SIZE=128, spec_type = 'R_T', if_inverse = 'Forward'):

        # 01. Read the data and tokenize
        self.train_struc, self.train_spec = self.load_data(train_file, train_spec_file, train_ratio)
        self.dev_struc, self.dev_spec = self.load_data(dev_file, dev_spec_file)

        dims = self.train_spec.shape[1]//2
        if spec_type == 'R':
            self.train_spec = self.train_spec[:, :dims]
            self.dev_spec = self.dev_spec[:, :dims]
        elif spec_type == 'T':
            self.train_spec = self.train_spec[:, dims:]
            self.dev_spec = self.dev_spec[:, dims:]

        # 02. build dictionary: structure
        self.struc_word_dict, self.struc_total_words, self.struc_index_dict = self.build_dict(self.train_struc)

        # 03. word to id by dictionary 
        self.train_struc = self.wordToID(self.train_struc, self.struc_word_dict)
        self.dev_struc = self.wordToID(self.dev_struc, self.struc_word_dict)

        # 04. batch + padding + mask
        self.train_data = self.splitBatch(self.train_struc, self.train_spec, BATCH_SIZE, if_inverse)
        self.dev_data   = self.splitBatch(self.dev_struc, self.dev_spec, BATCH_SIZE, if_inverse)

    def load_data(self, path, spec_path, ratio = 100):

        """
        Read structure and spec Data 
        tokenize the structure and add start/end marks(Begin of Sentence; End of Sentence)
        """

        struc = []
        all_struc = []

        with open (path, 'rb') as fp:
            all_struc = pkl.load(fp)

        for ele in all_struc:
            struc.append(["BOS"] + ele + ["EOS"])

        with open (spec_path, 'rb') as fp:
            spec = pkl.load(fp)       

        if ratio <=0 or ratio > 100:
            raise NameError('Wrong training dataset ratio. Make sure it is (0, 100]. ') 
        
        lengs = len(struc)
        struc = struc[:int(ratio*lengs/100)]
        spec = spec[:int(ratio*lengs/100)]

        return struc, spec
    
    def build_dict(self, sentences, max_words = 1000):
        """
        sentences: list of word list 
        build dictonary as {key(word): value(id)}
        """
        word_count = Counter()
        for sentence in sentences:
            for s in sentence:
                word_count[s] += 1

        ls = word_count.most_common(max_words)
        total_words = len(ls) + 2
        word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
        word_dict['UNK'] = UNK
        word_dict['PAD'] = PAD
        index_dict = {v: k for k, v in word_dict.items()}
        return word_dict, total_words, index_dict

    def wordToID(self, en, en_dict, sort=False):
        """
        convert input/output word lists to id lists. 
        Use input word list length to sort, reduce padding.
        """

        out_en_ids = [[en_dict.get(w, 0) for w in sent] for sent in en]
        # out_cn_ids = [[cn_dict.get(w, 0) for w in sent] for sent in cn]

        def len_argsort(seq):
            """
            get sorted index w.r.t length.
            """
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        if sort: # update index
            sorted_index = len_argsort(out_en_ids)
            out_en_ids = [out_en_ids[id] for id in sorted_index]
        return out_en_ids

    def splitBatch(self, struc, spec, batch_size, if_inverse = 'Forward', shuffle=False):
        """
        get data into batches
        """
        idx_list = np.arange(0, len(struc), batch_size)
        if shuffle:
            np.random.shuffle(idx_list)

        batch_indexs = []
        for idx in idx_list:
            batch_indexs.append(np.arange(idx, min(idx + batch_size, len(struc))))
        
        # print(batch_indexs, len(struc))
        batches = []
        for batch_index in batch_indexs:
            batch_struc = [struc[index] for index in batch_index]  
            batch_spec = [spec[index] for index in batch_index]

            batch_struc = seq_padding(batch_struc)  # pad the structure sequence 
            batches.append(Batch(batch_struc, np.array(batch_spec), if_inverse)) 

        return batches
