import sys
sys.path.append('../optogpt')
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
from core.datasets.sim import load_materials, spectrum
from core.models.transformer import make_model_I, subsequent_mask
from core.trains.train import Variable
from generate_dev_data import generate_dev_data
import multiprocessing as mp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATABASE = '../optogpt/nk'
illuminant = SDS_ILLUMINANTS['D65']
cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']

mats = ['Al', 'Al2O3', 'AlN', 'Ge', 'HfO2', 'ITO', 'MgF2', 'MgO', 'Si', 'Si3N4', 'SiO2', 'Ta2O5', 'TiN', 'TiO2', 'ZnO', 'ZnS', 'ZnSe', 'Glass_Substrate']
thicks = [str(i) for i in range(5, 255, 5)]

lamda_low = 0.4
lamda_high = 1.1
wavelengths = np.arange(lamda_low, lamda_high+1e-3, 0.01)

class Prepare_Augment_Data:

    def __init__(self, model_path, decoding_method = "Greedy Decode",error_type = "MAE",top_k = 10, top_p = 0.9, kp_num = 50, keep_num = 20):
        print("Initializing...")
        # load the model
        self.model,self.args = self.load_model(model_path)

        self.error_type = error_type

        # load data
        self.load_data_file(model_path)
        
        # get the id_word_dict
        self.id_word_dict = self.get_id_word_dict()
        
        print("Calculating original spec...")
        self.original_spec = self.get_original_spec()  

        self.top_k = top_k
        self.top_p = top_p
        self.kp_num = kp_num  
        
        print("Calculating designed structure...")
        if decoding_method == "Greedy Decode":
            self.all_mat, self.all_thick, self.designed_struct = self.get_designed_struct(self.greedy_decode)
        elif decoding_method == "TOP-KP Decode":
            # repeat the decoding process for kp_num times
            self.all_mat, self.all_thick, self.designed_struct = [],[],[]
            for i in range(self.kp_num):
                print("Decoding process: {}/{}".format(i+1,self.kp_num))
                mat, thick, struct = self.get_designed_struct(self.top_kp_decode)
                self.all_mat.append(mat)
                self.all_thick.append(thick)
                self.designed_struct.append(struct)
            # flatten the list
            self.all_mat = [inner for outer in self.all_mat for inner in outer]
            self.all_thick = [inner for outer in self.all_thick for inner in outer]
            self.designed_struct = [inner for outer in self.designed_struct for inner in outer]
            #copy original_spec kp_num times
            all_original_spec = []
            for i in range(self.kp_num):
                all_original_spec.append(self.original_spec)
            self.original_spec = [inner for outer in all_original_spec for inner in outer]
        elif decoding_method == "TOP-KP Decode_v2":
            # repeat the decoding process for kp_num times
            self.all_mat, self.all_thick, self.designed_struct = [],[],[]
            all_original_spec = []
            cnt = 0
            for spec in self.original_spec:
                tmp_mat, tmp_thick, tmp_struct = [], [], []
                for i in range(self.kp_num):
                    with torch.no_grad():
                        struct = self.top_kp_decode(spec, 10, 'BOS')
                        mat, thick = self.return_mat_thick(struct)
                        tmp_mat.append(mat)
                        tmp_thick.append(thick)
                        tmp_struct.append(struct)
                # calculate the error for each decoding
                tmp_error = []
                for i in range(self.kp_num):
                    int_list = [int(item) for item in tmp_thick[i]]
                    tmp_error.append(self.calc_single_error(spec, spectrum(tmp_mat[i],int_list)))
                # keep only the keep_num best decoding
                best_idx = np.argsort(tmp_error)[:keep_num]
                self.all_mat.append([tmp_mat[i] for i in best_idx])
                self.all_thick.append([tmp_thick[i] for i in best_idx])
                self.designed_struct.append([tmp_struct[i] for i in best_idx])
                all_original_spec.append([spec]*keep_num)
                cnt += 1
                if cnt % 100 == 0:
                    print("Decoding process: {}/{}".format(cnt,len(self.original_spec)))
            # flatten the list
            self.all_mat = [inner for outer in self.all_mat for inner in outer]
            self.all_thick = [inner for outer in self.all_thick for inner in outer]
            self.designed_struct = [inner for outer in self.designed_struct for inner in outer]
            self.original_spec = all_original_spec
            self.original_spec = [inner for outer in all_original_spec for inner in outer]
        elif decoding_method == 'Beam Search':
            self.all_mat, self.all_thick, self.designed_struct = [],[],[]
            cnt = 0
            for spec in self.original_spec:
                with torch.no_grad():
                    if cnt % 100 == 0:
                        print("Decoding process: {}/{}".format(cnt,len(self.original_spec)))
                    cnt += 1
                    struct = self.beam_search_decode(spec, 10, 'BOS',keep_num)
                    for i in range(keep_num):
                        mat, thick = self.return_mat_thick(struct[i])
                        self.all_mat.append(mat)
                        self.all_thick.append(thick)
                        self.designed_struct.append(struct[i])
            # duplicate self.original_spec keep_num times like [spec1,spec1,spec1,spec2,spec2,spec2,...]
            all_original_spec = []
            for spec in self.original_spec:
                for i in range(keep_num):
                    all_original_spec.append(spec)
            self.original_spec = all_original_spec


        print("Calculating designed spec...")
        self.designed_spec = self.simulate_spec()
        
        print("Calculating Error...")
        self.error = self.calc_error()
        

    # helper function
    def return_mat_thick(self, struc_list):
        materials = []
        thickness = []
        for struc_ in struc_list:
            materials.append(struc_.split('_')[0])
            thickness.append(struc_.split('_')[1])

        return materials, thickness

    def get_id_word_dict(self):
        # translate the encoding back to structure
        word_id_dict = self.struc_word_dict
        # make a reverse dictionary
        id_word_dict = {}
        for key, value in word_id_dict.items():
            id_word_dict[value] = key
        return id_word_dict
            
    #translate an array of word ids to words
    def translate(self, word_ids, id_word_dict):
        word_ids = word_ids.to('cpu').numpy()
        words = []
        for i in word_ids:
            words.append(id_word_dict[i])
        return words

    def load_model(self, model_path):
        #load the model
        a = torch.load(model_path)
        args = a['configs']
        torch.manual_seed(args.seeds)
        np.random.seed(args.seeds)
        model = make_model_I(
                        args.spec_dim, 
                        args.struc_dim,
                        args.layers, 
                        args.d_model, 
                        args.d_ff,
                        args.head_num,
                        args.dropout
                    ).to(DEVICE)

        model.load_state_dict(a['model_state_dict'])
        return model, args

    def load_data_file(self, model_path):
        # load the training and spec data
        a = torch.load(model_path)
        self.train_spec = generate_dev_data()
        self.struc_word_dict,  self.struc_index_dict = a['configs'].struc_word_dict, a['configs'].struc_index_dict
        
        return

    def get_original_spec(self):
        return self.train_spec

    def get_designed_struct(self, decoding_method):
        all_mat = []
        all_thick = []
        designed_struc = []

        with torch.no_grad():
            for src in self.original_spec:
                crt_res = decoding_method(list(src), 10, 'BOS')
                material, thickness =  self.return_mat_thick(crt_res)
                all_mat.append(material)
                all_thick.append(thickness)
                designed_struc.append(crt_res)
        return all_mat, all_thick, designed_struc
                    
    def simulate_spec(self):
        NUM_CORES = min(mp.cpu_count(), 16)  # Reasonable default
        DATABASE = './nk'
        nk_dict = load_materials(all_mats=mats, wavelengths=wavelengths, DATABASE=DATABASE)
        
        from multiprocessing import Pool
        args_for_starmap = [(mat, thick, 's', 0, wavelengths, nk_dict, 'Glass_Substrate', 500000) 
                            for mat, thick in zip(self.all_mat, self.all_thick)]

        # Create a pool and use starmap
        with Pool(NUM_CORES) as pool:
            spec_res = pool.starmap(spectrum, args_for_starmap)
        pool.close()
        
        return np.array(spec_res)

    def calc_error(self):
        error = []
        for i in range(len(self.original_spec)):
            if self.error_type == "MAE":
                error.append(np.mean(np.abs(self.original_spec[i] - self.designed_spec[i])))
            elif self.error_type == "MSE":
                error.append(np.mean(np.square(self.original_spec[i] - self.designed_spec[i])))
        return error
    
    def calc_single_error(self, spec, designed_spec):
        if self.error_type == "MAE":
            return np.mean(np.abs(spec - designed_spec))
        elif self.error_type == "MSE":
            return np.mean(np.square(spec - designed_spec))

    def greedy_decode(self, spec_target, max_len, start_symbol, start_mat = None):
        """
        use greedy decode to generate text 
        """
        # init 1×1 tensor as prediction，fill in ('BOS')id, type: (LongTensor)
        start_symbol = self.struc_word_dict[start_symbol]
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.LongTensor).to(DEVICE)

        if start_mat:
            start_mat = self.struc_word_dict[start_mat]
            ys = torch.tensor([[start_symbol, start_mat]]).type(torch.LongTensor).to(DEVICE)
            struc_design = [start_mat]
        else:
            struc_design = []

        # process src
        src = torch.tensor([spec_target]).unsqueeze(0).float().to(DEVICE)
        src_mask = None

        struc_design = []
        probs = []
        for i in range(max_len-1):
            # decode one by one
            trg_mask = Variable(subsequent_mask(ys.size(1)).type_as(src.data))
            
            out = self.model(src.to(DEVICE), Variable(ys), src_mask, trg_mask.to(DEVICE))

            #  out to log_softmax 
            prob = self.model.generator(out[:, -1])
            probs.append(prob[0, :].to('cpu').tolist())

            #  get the max-prob id
            _, next_word = torch.max(prob, dim = 1)
            next_word = next_word.data[0]
            #  concatnate with early predictions
            ys = torch.cat([ys,torch.ones(1, 1).type(torch.LongTensor).fill_(next_word).to(DEVICE)], dim=1)
            sym = self.struc_index_dict[next_word.to('cpu').item()]
            if sym != 'EOS':
                struc_design.append(sym)
            else:
                break

        return struc_design
    
    def top_kp_decode(self, spec_target, max_len, start_symbol, start_mat=None):
        """
        Use top-k and top-p (top-kp) decode to generate text
        """
        # process src
        src = torch.tensor([spec_target]).unsqueeze(0).float().to(DEVICE)
        src_mask = None

        probs = []
        start_symbol = self.struc_word_dict[start_symbol]
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.LongTensor).to(DEVICE)

        if start_mat:
            start_mat = self.struc_word_dict[start_mat]
            ys = torch.tensor([[start_symbol, start_mat]]).type(torch.LongTensor).to(DEVICE)
            struc_design = [start_mat]
        else:
            struc_design = []

        src = torch.tensor([spec_target]).unsqueeze(0).float().to(DEVICE)
        src_mask = None

        for i in range(max_len - 1):
            trg_mask = Variable(subsequent_mask(ys.size(1)).type_as(src.data))
        
            out = self.model(src.to(DEVICE), Variable(ys), src_mask, trg_mask.to(DEVICE))

            #  out to log_softmax 
            prob = self.model.generator(out[:, -1]).exp().to('cpu')

            prob_sort = torch.argsort(prob, descending=True)

            prob_item_select = []
            prob_total = 0
            i = 0
            while prob_total < self.top_p and len(prob_item_select) < min(int(self.top_k), prob.size(1)):
                mat_design = self.struc_index_dict[prob_sort[0, i].item()].split('_')[0]
                prob_item_select.append(prob_sort[0, i].item())
                prob_total += prob[0, prob_sort[0, i]].item()
                i += 1
                prob_select = [prob[0, i].item() for i in prob_item_select]
            probs.append(prob_item_select + prob_select)

            temp_sum = sum(prob_select)
            prob_select = [i/temp_sum for i in prob_select]

            next_word = np.random.choice(prob_item_select, p=prob_select)       
            
            #  concatnate with early predictions
            ys = torch.cat([ys,torch.ones(1, 1).type(torch.LongTensor).fill_(next_word).to(DEVICE)], dim=1)
            sym = self.struc_index_dict[next_word]

            if sym != 'EOS':
                struc_design.append(sym)
            else:
                break

        return struc_design
    
    def beam_search_decode(self, spec_target, max_len, start_symbol, beam_width=5, start_mat=None):
        """
        Use beam search decode to generate text
        """
        src = torch.tensor([spec_target]).unsqueeze(0).float().to(DEVICE)
        src_mask = None

        start_symbol = self.struc_word_dict[start_symbol]
        initial_beam = {
            'ys': torch.ones(1, 1).fill_(start_symbol).type(torch.LongTensor).to(DEVICE),
            'score': 0,
            'struc_design': []
        }

        if start_mat:
            start_mat = self.struc_word_dict[start_mat]
            initial_beam['ys'] = torch.tensor([[start_symbol, start_mat]]).type(torch.LongTensor).to(DEVICE)
            initial_beam['struc_design'] = [start_mat]
        
        beams = [initial_beam]

        for _ in range(max_len - 1):
            all_candidates = []
            for beam in beams:
                trg_mask = Variable(subsequent_mask(beam['ys'].size(1)).type_as(src.data))
                out = self.model(src, Variable(beam['ys']), src_mask, trg_mask)
                prob = self.model.generator(out[:, -1]).softmax(dim=-1)

                # Get top beam_width candidates for this beam
                top_probs, top_idxs = prob.topk(beam_width)

                for i in range(beam_width):
                    next_word = top_idxs[0][i].item()
                    score = beam['score'] - torch.log(top_probs[0][i])  # Use log prob for numerical stability
                    ys = torch.cat([beam['ys'], torch.tensor([[next_word]], device=DEVICE)], dim=1)
                    sym = self.struc_index_dict[next_word]
                    struc_design = beam['struc_design'] + [sym] if sym != 'EOS' else beam['struc_design']

                    candidate = {
                        'ys': ys,
                        'score': score,
                        'struc_design': struc_design
                    }
                    all_candidates.append(candidate)

            # Sort all candidates by score and select top beam_width
            beams = sorted(all_candidates, key=lambda x: x['score'])[:beam_width]

            # Check if all beams ended with EOS
            if all(self.struc_index_dict[beam['ys'][0, -1].item()] == 'EOS' for beam in beams):
                break

        return [beam['struc_design'] for beam in beams]
