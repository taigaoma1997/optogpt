import sys
sys.path.append('../optogpt')
import os
import math
import numpy as np
import torch
import random
import pandas as pd
from core.datasets.sim import load_materials, spectrum
from multiprocessing import Pool
import multiprocessing
import multiprocessing as mp
import pyswarms as ps

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATABASE = '../optogpt/nk'
illuminant = SDS_ILLUMINANTS['D65']
cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']

mats = ['Al', 'Ag', 'Al2O3', 'AlN', 'Ge', 'HfO2', 'ITO', 'MgF2', 'MgO', 'Si', 'Si3N4', 'SiO2', 'Ta2O5', 'TiN', 'TiO2', 'ZnO', 'ZnS', 'ZnSe', 'Glass_Substrate']
thicks = [str(i) for i in range(10, 505, 10)]
high_index_mats = ['TiO2','ZnS','ZnSe','Ta2O5','HfO2']
medium_index_mats = ['SiO2','Al2O3','MgF2','Si3N4']
low_index_mats = ['MgO','ITO']
metal_mats = ['Al','Ag']
semi_mats = ['Ge','Si']
 
lamda_low = 0.4
lamda_high = 1.1
wavelengths = np.arange(lamda_low, lamda_high+1e-3, 0.01)

nk_dict = load_materials(all_mats = mats, wavelengths = wavelengths, DATABASE = DATABASE)


def return_mat_thick(struc_list):
        materials = []
        thickness = []
        for struc_ in struc_list:
            materials.append(struc_.split('_')[0])
            thickness.append(struc_.split('_')[1])

        return materials, thickness

POPULATION_SIZE = 100
  
# Valid genes: materials
  
# Target string to be generated 
def mutate_genes(individual): 
    ''' 
    gene mutation with specified probabilities
    '''
    mat_to_mutate = np.random.randint(0, len(individual.chromosome))
    material = individual.chromosome[mat_to_mutate]
    
    # Determine mutation type based on specified probabilities
    mutation_type = np.random.choice(['same_class', 'unchanged', 'other_class'], p=[0.8, 0.1, 0.1])
    
    if mutation_type == 'same_class':
        # mutate within the same category
        if material in high_index_mats:
            new_material = random.choice([m for m in high_index_mats if m != material])  # Avoid picking the same material
        elif material in medium_index_mats:
            new_material = random.choice([m for m in medium_index_mats if m != material])
        elif material in low_index_mats:
            new_material = random.choice([m for m in low_index_mats if m != material])
        elif material in metal_mats:
            new_material = random.choice([m for m in metal_mats if m != material])
        else:
            new_material = random.choice([m for m in semi_mats if m != material])
    elif mutation_type == 'unchanged':
        # Keep the material unchanged
        new_material = material
    elif mutation_type == 'other_class':
        # mutate to a different category
        all_mats = []
        if material in high_index_mats:
            all_mats = medium_index_mats + low_index_mats + metal_mats + semi_mats
        elif material in medium_index_mats:
            all_mats = high_index_mats + low_index_mats + metal_mats + semi_mats
        elif material in low_index_mats:
            all_mats = high_index_mats + medium_index_mats + metal_mats + semi_mats
        elif material in metal_mats:
            all_mats = high_index_mats + medium_index_mats + low_index_mats + semi_mats
        else:
            all_mats = high_index_mats + medium_index_mats + low_index_mats + metal_mats
        new_material = random.choice([m for m in all_mats if m != material])  # Ensure different material is chosen
        
    individual.chromosome[mat_to_mutate] = new_material
    individual.cal_fitness()
    return individual

def mate(par1, par2): 
    ''' 
    Perform mating and produce new offspring 
    '''

    # chromosome for offspring 
    child_chromosome = [] 
    child_thickness = []
    cut_off_point = np.random.randint(0, min(len(par1.chromosome), len(par2.chromosome)))
    dice = np.random.random()
    if dice < 0.5:
        child_chromosome = par1.chromosome[:cut_off_point] + par2.chromosome[cut_off_point:]
        child_thickness = par1.thickness[:cut_off_point] + par2.thickness[cut_off_point:]
    else:
        child_chromosome = par2.chromosome[:cut_off_point] + par1.chromosome[cut_off_point:]
        child_thickness = par2.thickness[:cut_off_point] + par1.thickness[cut_off_point:]
    
    # create new Individual(offspring) using  
    child_individual  = [str(child_chrom) + '_' + str(child_thick) for child_chrom, child_thick in zip(child_chromosome, child_thickness)]
    # generated chromosome for offspring 
    return Individual(child_individual, par1.target) 

  
class Individual(): 
    ''' 
    Class representing individual in population 
    '''
    def __init__(self, chromosome, target): 
        self.chromosome, self.thickness = return_mat_thick(chromosome)  
        self.target = target
        self.fitness = self.cal_fitness() 

    def cal_fitness(self): 
        ''' 
        Calculate fitness score, it is the number of 
        characters in string which differ from target 
        string. 
        '''
        designed_spec = spectrum(self.chromosome, self.thickness, wavelengths=wavelengths, nk_dict=nk_dict, substrate='Glass_Substrate', substrate_thick=500000)
        fitness = np.mean(np.square(np.array(designed_spec) - np.array(self.target)))
        
        return fitness 

def GA_perturb(mat_thicks,target,size):
    ''' 
    Create initial population 
    '''
    population = [] 
    for i in range(size): 
        crt = mat_thicks[i]
        population.append(Individual(crt, target)) 
    generation = 1
    # stopping criteria: mean fitness score of the population doesn't improve for 10 generations
    early_stopping_patience = 20
    early_stopping = 0
    crt_best_fitness = np.mean([ind.fitness for ind in population])
    crt_best_population = mat_thicks
    crt_best_err = []
    print("Original mean fitness: ", np.mean([ind.fitness for ind in population]))
    while True:     
        ''' 
        Sort the population in increasing order of fitness score 
        '''
        population = sorted(population, key = lambda x:x.fitness) 
        # if the mean fitness score doesn't improve for 10 generations, stop
        
        #calculate the mean fitness score
        mean_fitness = np.mean([population[i].fitness for i in range(20)])
        if mean_fitness < crt_best_fitness:
            crt_best_fitness = mean_fitness
            early_stopping = 0
            crt_best_population = []
            for i in range(20):
                crt_struct = []
                for j in range(len(population[i].chromosome)):
                    crt_struct.append(str(population[i].chromosome[j]) + '_' + str(population[i].thickness[j]))
                crt_best_population.append(crt_struct)
            crt_best_err = []
            for i in range(20):
                crt_best_err.append(population[i].fitness)
        else:
            early_stopping += 1
        if early_stopping == early_stopping_patience:
            break
        
        ''' 
        Otherwise generate new offsprings for new generation 
        '''
        new_generation = [] 
        ''' 
        Perform Elitism, that mean 10% of fittest population 
        goes to the next generation 
        '''
        s = int((10*POPULATION_SIZE)/100) 
        new_generation.extend(population[:s]) 
        ''' 
        From 50% of fittest population, Individuals 
        will mate to produce offspring 
        '''
        s = int((90*POPULATION_SIZE)/100) 
        for _ in range(s): 
            parent1 = random.choice(population[:10]) 
            parent2 = random.choice(population[:10]) 
            child = mate(parent1, parent2) 
            mutated_child = mutate_genes(child)
            new_generation.append(mutated_child) 

        
        population = new_generation 
        print("Generation: ", generation, "Fitness: ", np.mean([ind.fitness for ind in population]) )
 #       print("Best struc: ", population[0].fitness)
        generation += 1
    return crt_best_population    


def PSO_perturb(mat_thicks, target, type='MAE'):
    
    materials, thickness = return_mat_thick(mat_thicks)

    M_t = len(thickness)
    x = [float(ii) for ii in thickness]

    def objective_func(x):
        # x is a matrix of shape (n_particles, dimensions)
        # Each row represents a particle (potential solution)
        n_particles = x.shape[0]
        j = [0] * n_particles

        for i in range(n_particles):
            R_T = spectrum(materials, list(x[i]), wavelengths=wavelengths, nk_dict=nk_dict, substrate='Glass_Substrate', substrate_thick=500000)
            if type == 'MAE':
                j[i] = np.mean(np.abs(np.array(R_T) - np.array(target)))
            elif type == 'MSE':
                j[i] = np.mean(np.square(np.array(R_T) - np.array(target)))
            else:
                raise NotImplementedError
        return np.array(j)
    # PSO hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

    # Bounds for each dimension in the search space
    bounds = (np.array([10] * M_t), np.array([500] * M_t))

    init_pos = np.full((5, M_t), x)

    # Initialize the optimizer
    optimizer = ps.single.GlobalBestPSO(n_particles=5, dimensions=M_t, options=options, bounds=bounds, init_pos = init_pos)
    
    # Create the final structure
    
    all_structures = []
    for _ in range(5):  # Number of iterations
        # Perform one step of optimization
        optimizer.optimize(objective_func, iters=1)

        # Record the current structures
        current_positions = optimizer.swarm.position
        for particle_pos in current_positions:
            temp_struc = [materials[i] + '_' + str(int(pos)) for i, pos in enumerate(particle_pos)]
            all_structures.append(temp_struc)

    # Return the list of all structures
    return all_structures
        

def apply_func_to_chunk(chunk_df):
    return chunk_df.apply(lambda x: PSO_perturb(x['designed_struct'], x['original_spec'], type='MSE'), axis=1)

def apply_func_to_chunk2(chunk_df):
    return chunk_df.apply(lambda x: PSO_perturb(x['perturb_struct1'], x['original_spec'], type='MSE'), axis=1)


def get_to_be_perturbed_data(original_data, give_up_threshold=3):
    origin_df = pd.DataFrame()
    origin_df['original_spec'] = original_data.original_spec
    origin_df['designed_spec'] = original_data.designed_spec.tolist()
    origin_df['designed_struct'] = original_data.designed_struct
    origin_df['error'] = original_data.error

    # get the data with mae smaller than threshold
    origin_df = origin_df[origin_df['error'] < give_up_threshold].reset_index(drop=True)

    return origin_df

def random_perturb(crt_struct):
        if len(crt_struct) == 0:
            return crt_struct
        idx = np.random.randint(0, len(crt_struct))
        while (crt_struct[idx] in ['EOS','BOS','UNK']):
            idx = np.random.randint(0, len(crt_struct))
        # separate the structure into material and thickness
        material, thickness = crt_struct[idx].split('_')
        # disturb the thickness randomly by 10/20/30/40/50 nm, normal distribution
        # Define the numbers and their corresponding probabilities
        numbers = [10, 20, 30, 40, 50]
        probabilities = [0.30, 0.25, 0.20, 0.15, 0.10]

        # Sample a number based on the defined probabilities
        sampled_number = random.choices(numbers, probabilities, k=1)[0]

        # Randomly choose whether the number is positive or negative
        sampled_number *= random.choice([-1, 1])    
        thickness = int(thickness) + sampled_number
        if (thickness > 500):
            thickness = 500
        if (thickness < 10):
            thickness = 10
        # combine the material and thickness
        crt_struct[idx] = str(material)+'_'+str(thickness)
        # perturb the structure
        return crt_struct

def apply_func_to_chunk_GA(chunk_df):
    return chunk_df.apply(lambda x: GA_perturb(x['designed_struct'], x['original_spec'], 20), axis=1)

def perturb_data(df, method="random", output_dir=None):

    if method == "random":
        # random perturb
        perturb_df = df.copy()
        perturb_df['perturb_struct'] = perturb_df['designed_struct'].apply(lambda x:random_perturb(x))
    elif method == "PSO":
        # PSO perturb
        perturb_df = df.copy()

        def parallelize_dataframe(df, func):
            num_cores = mp.cpu_count()  # Number of CPU cores
            df_split = np.array_split(df, min(100, num_cores))  # Split DataFrame into chunks
            pool = mp.Pool(num_cores)
            results = pool.map(func, df_split)  # Process each chunk in parallel
            pool.close()
            pool.join()
            # Concatenate results into a single Series
            concatenated_series = pd.Series([item for sublist in results for item in sublist])
            return concatenated_series
        perturb_df['perturb_struct'] = parallelize_dataframe(perturb_df, apply_func_to_chunk)
        perturb_df = expand_perturbed_data(perturb_df)
        perturb_df['perturb_struct'] = perturb_df.apply(lambda x:round_to_nearest_length(x),axis=1)

    elif method == "GA_PSO":
        # first use GA to perturb the structure, then use PSO to optimize the perturbed structure

        def parallelize_dataframe(df, func):
            num_cores = mp.cpu_count()  # Number of CPU cores
            df_split = np.array_split(df, min(100, num_cores))  # Split DataFrame into chunks
            pool = mp.Pool(num_cores)
            results = pool.map(func, df_split)  # Process each chunk in parallel
            pool.close()
            pool.join()
            # Concatenate results into a single Series
            concatenated_series = pd.Series([item for sublist in results for item in sublist])
            return concatenated_series
        new_df = df.copy()
        original_error = new_df['error'][19::20].reset_index(drop=True)
        designed_spec = new_df['designed_spec'][19::20].reset_index(drop=True)
        original_spec = new_df['original_spec'][19::20].reset_index(drop=True)

        tmp = []
        for i in range(0, len(new_df), 20):
            tmp.append(new_df['designed_struct'][i:i+20].to_list())
        chunks = pd.Series(tmp)
        designed_structs = pd.DataFrame({'designed_struct':chunks})
        designed_structs['original_spec'] = original_spec
        designed_structs['designed_spec'] = designed_spec
        designed_structs['error'] = original_error
        designed_structs.reset_index()
        print(designed_structs.iloc[0])


        designed_structs['perturb_struct1'] = parallelize_dataframe(designed_structs, apply_func_to_chunk_GA)
        designed_structs = designed_structs.explode('perturb_struct1').reset_index(drop=True)

        # Save intermediate results if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            designed_structs.to_pickle(os.path.join(output_dir, "designed_structs.pkl"))

        # apply PSO to the perturbed structure
        perturb_df = designed_structs.copy()
        perturb_df['perturb_struct'] = parallelize_dataframe(perturb_df, apply_func_to_chunk2)
        perturb_df = expand_perturbed_data(perturb_df)
        perturb_df['perturb_struct'] = perturb_df.apply(lambda x:round_to_nearest_length(x),axis=1)    
        
        # Save final results if output_dir is provided
        if output_dir:
            perturb_df.to_pickle(os.path.join(output_dir, "perturb_df.pkl"))

    else:
        raise NotImplementedError
    return perturb_df

def simulate_perturbed_struct(perturb_df, error_type="MAE"):
    def return_mat_thick(struc_list):
        materials = []
        thickness = []
        for struc_ in struc_list:
            if (struc_ != 'EOS' and struc_ != 'BOS' and struc_ != 'UNK' and struc_ != 'PAD'):
                materials.append(struc_.split('_')[0])
                thickness.append(struc_.split('_')[1])
        return materials, thickness
    
    def simulate_spec(all_mat, all_thick):
        NUM_CORES = min(mp.cpu_count(), 16)  # Reasonable default
        DATABASE = './nk'
        mats = ['Al', 'Ag', 'Al2O3', 'AlN', 'Ge', 'HfO2', 'ITO', 'MgF2', 'MgO', 'Si', 'Si3N4', 'SiO2', 'Ta2O5', 'TiN', 'TiO2', 'ZnO', 'ZnS', 'ZnSe', 'Glass_Substrate']
        nk_dict = load_materials(all_mats = mats, wavelengths = wavelengths, DATABASE = DATABASE )
        args_for_starmap = [(mat, thick, 's', 0, wavelengths, nk_dict,'Glass_Substrate',  500000) 
                            for mat, thick in zip(all_mat, all_thick)]

        # Create a pool and use starmap
        with Pool(NUM_CORES) as pool:
            spec_res = pool.starmap(spectrum, args_for_starmap)
        pool.close()
        
        return spec_res
    
    # separate the structure into material and thickness for perturbed
    all_mat = []
    all_thick = []
    for i in range(len(perturb_df)):
        material, thickness =  return_mat_thick(perturb_df.iloc[i,:]['perturb_struct'])
        all_mat.append(material)
        all_thick.append(thickness)

    # simulate the perturbed spectrum
    perturb_df['perturb_spec'] = simulate_spec(all_mat, all_thick)

    # calculate the new error
    if error_type == "MAE":
        perturb_df['new_error'] = perturb_df.apply(lambda x:np.mean(np.abs(np.array(x['original_spec']) - np.array(x['perturb_spec']))),axis=1)
    else:
        perturb_df['new_error'] = perturb_df.apply(lambda x:np.mean(np.square(np.array(x['original_spec']) - np.array(x['perturb_spec']))),axis=1)
    return perturb_df

def get_perturbed_better_data(perturb_df):
    # get the data with new mae smaller than original mae
    perturb_df = perturb_df[perturb_df['new_error'] < perturb_df['error']]
    return perturb_df

def expand_perturbed_data(perturb_df):
    # Create a new DataFrame
    new_df = pd.DataFrame()

    # Copy the relevant columns
    for col in ['original_spec', 'designed_spec', 'designed_struct', 'perturb_struct', 'error']:
        new_df[col] = perturb_df[col]

    # Explode the perturb_struct column
    new_df = new_df.explode('perturb_struct')

    # Reset the index
    new_df.reset_index(drop=True, inplace=True)

    return new_df

def round_to_nearest_length(perturb_df_slice):
    perturb_struct = perturb_df_slice['perturb_struct']
    for i in range(len(perturb_struct)):
        material, thickness = perturb_struct[i].split('_')
        thickness = int(thickness)
        # round the thickness to the nearest 10
        thickness = int(round(thickness / 10.0)) * 10
        # if the thickness is smaller than 10, set it to 10
        if (thickness < 10):
            thickness = 10
        # if the thickness is larger than 500, set it to 500
        if (thickness > 500):
            thickness = 500
        # combine the material and thickness
        perturb_struct[i] = material + '_' + str(thickness)
    return perturb_struct









    
