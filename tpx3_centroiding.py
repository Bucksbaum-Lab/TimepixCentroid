import numpy as np
import time
import torch
from scipy.ndimage import center_of_mass as com
from scipy.optimize import curve_fit

from tqdm.notebook import tqdm
from pathlib import Path

def centroid_block(block):
    '''
    Produces centroids of hits within a single trigger
    Parameters
    ~~~~~~~~~~
    block : array of shape (N,4) listing all (ToA, ToT, y, x) pixel values for a single trigger
    Returns
    ~~~~~~~~~~
    centroids_block : 2-tuple, list of x and y centroid values
    '''
    if block.size==0:
        return 0,0
    tots, toas, ys, xs = block.T
    centroids_block = (torch.sum(xs*tots)/torch.sum(tots)),(torch.sum(ys*tots)/torch.sum(tots))
    return centroids_block

def get_neighbors(block,size=1,tsep=5.0e-7):
    '''
    Takes a list of TimePix3 pixels from a single trigger and finds which ones are in the neighborhood of each other. Here, pixels are considered to be neighbors of one another if they light up within a certain distance in each of (X,Y,ToF). This allows us to group pixels that came from the same hit on the MCP/phosphor.
    This function finds neighbors quickly using array operations on the GPU. For each of (X,Y,ToF), it repeats the array N times to make it a square matrix, then subtracts the transpose of this square matrix from itself. This results in a distance matrix between pixels i and j in the (i,j)th row and column. The neighborhood adjacency matrix is then built by taking a boolean condition on the distance matrix, i.e. abs(distance)<max_distance. Pixels must satisfy the neighbor condition in each of (X,Y,ToF) to be considered neighbors.
    Parameters
    ~~~~~~~~~~
    block : PyTorch tensor of shape (N,4) OR (N,4,N_batch) containing all pixel data from a single trigger OR from a batch of triggers. 
    size : int (default 1), the +/- size of the neighborhood to consider in X and Y (i.e. size=1 implies 3x3 neighborhoods, size=2 implies 5x5 neighborhoods)
    tsep : float (default 5.0e-7), ToF neighborhood size in seconds (i.e. tsep=5.0e-7 means pixels within half a microsecond in ToF are considered neighbors)
    Returns
    ~~~~~~~
    neighbors : boolean PyTorch tensor of shape (N,N) OR (N,N,N_batch). The (i,j)th entry indicates whether pixel i and pixel j in block are considered neighbors.
    '''
    block_rep = block.expand(block.shape[0],*[-1 for _ in range(len(block.shape))])
    block_sub = block_rep - block_rep.transpose(1,0)
    neighbors = (torch.abs(block_sub[:,:,3])<=size)&(torch.abs(block_sub[:,:,2])<=size)&(torch.abs(block_sub[:,:,0])<=tsep)
    return neighbors

def get_local_maxima(block,neighbors,min_size=3,**kwargs):
    '''
    Finds the brightest pixel in each hit and uses that pixel to centroid around. This uses the pixel Time-over-Threshold (ToT) as the measure of brightness, and uses the neighbors matrix from get_neighbors() to identify separate hits. 
    The function weights the neighbors matrix by each pixel's ToT value along a column. It then subtracts the value along the diagonal from each row. This produces columns representing each neighborhood where the ToT of the pixel in question along the diagonal has been subtracted from the ToTs of all its neighbors in the column, including itself. The columns where the maximum value in the column is 0 are those that correspond to the local maxima, because all of that pixel's neighbors' ToTs are less than its own. The function returns the indices of these local maximum columns.
    Parameters
    ~~~~~~~~~~
    block : PyTorch tensor of shape (N,4) OR (N,4,N_batch) containing all pixel data from a single trigger OR from a batch of triggers. 
    neighbors : boolean PyTorch tensor of shape (N,N) OR (N,N,N_batch). The (i,j)th entry indicates whether pixel i and pixel j in block are considered neighbors.
    min_size : int (default 3). The minimum number of pixels that must be in a neighborhood in order for the maximum pixel in that neighborhood to be considered a true local maximum. Setting min_size>1 eliminates single pixels that happened to light up because these will not give a good centroid.
    Returns
    ~~~~~~~
    local_maxima : boolean PyTorch tensor of shape (N,N_batch) indicating which pixels in block correspond to local maxima.
    '''
    try:
        block_rep = block.expand(block.shape[0],*[-1 for _ in range(len(block.shape))])
        tot_neighbors = block_rep[:,:,1]*neighbors
        tot_neighbors_sub = tot_neighbors.transpose(1,0) - torch.diagonal(tot_neighbors).T
        local_maxima = (torch.amax(tot_neighbors_sub,dim=0)==0)&(torch.amin(tot_neighbors_sub,dim=0)<0)&(torch.sum(neighbors,dim=0)>=min_size)
        return local_maxima
    except Exception as e:
        return torch.tensor([[]],dtype=int).to(block.device)
    
def tottof_fit_func(x, a, b, c, d):
    '''
    Used to correct for tot-tof correlation effect ('timewalk'). Fitted values are predetermined for specific VMI conditions and loaded in. Can also be run post-centroiding but is more precise this way
    Parameters
    ~~~~~~~~~~
    x : float, ToT values of pre-centroided data.
    a,b,c,d : values defined by isolating a single ion peak, finding the center of the ToF distribution for each ToT by fitting a gaussian, and fitting the resulting ToF(ToT) data to this function. Saved in the hard-coded variable 'fitty' in read_file_batched, changes in voltage/timing may affect this
    Returns
    ~~~~~~~~~~
    corrected ToF for input ToT value (float)
    '''
    return a / ((x + b) ** d) + c

def read_file_batched(filename,read_line_num = 100000000,batch_size=1,start_trigger_num=0,
                      skiprows=0,tottofcorr=True,show_bar=True,centroid_area_size=2,centroid_time_size=5e-7):
    '''
    Centroids a single TimePix3 file. Uses PyTorch, a Python-based library written for machine learning, to access the GPU and drastically speed up computations. Employs array-based operations that are equivalent to looping but execute in parallel on the GPU. Also uses batch processing of laser triggers (i.e. camera "frames") to further speed up processing.
    
    The centroids are computed in the neighborhood of local maxima, using Time-over-Threshold (ToT) as the measure of pixel brightness. The neighborhood of a local maxumim is defined both in X-Y space and in Time-of-Arrival (ToA) space since TimePix is sensitive to both. 
    Parameters
    ~~~~~~~~~~
    filename : str, path to .txt file to be processed
    read_line_num : int (default 100000000), number of lines in .txt file to read. Set this to a smaller number to read a subset of the data.
    batch_size : int (default 1), number of triggers to process in parallel. The maximum allowable value will depend on the amount of memory your GPU has, as well as the number of electron/ion hits per shot (more hits means more data per laser trigger). We have fount that with the NVIDIA Titan RTX (24 GB memory) and count rates below 100/shot, a batch size of 10 is a reasonable choice. If you get a CUDA Out of Memory error, try restarting the kernel, clearing the GPU cache with torch.cuda.empty_cache(), and then setting the batch size smaller.
    start_trigger_num : int (default 0), the trigger number to start at. Defaults to 0 (the first trigger in the file).
    skiprows : int (default 0), skip the first number of lines in the .txt file
    tottofcorr : (default True) Defines whether or not to do the tot/tof correction. Is generally helpful for higher precision ToFs but should be turned off if VMI conditions changed and tot/tof fit parameters need to be refitted to uncorrected data.
    show_bar : (default True) Defines whether to print time information and progress bar
    centroid_area_size : int (default 2), number of pixels in any direction of the local maximum taken into account when computing the centroid. For the default value of 2, the centroids will be computed in a 5x5-pixel square with the local maximum at the center. This parameter can be adjusted based on your MCP/phosphor voltage depending on the size of the hits you observe.
    centroid_time_size : float (default 5.0e-7), time window in seconds that defines the neighborhood of a local maximum. Since all the pixels in a single hit should light up within 500 ns of each other, the default is set to this. Setting this to a longer value risks including the pixels from two distinct hits into one centroid.
    Returns
    ~~~~~~~~~~
    centroids : Nx6 array of the form [x,y,tot,tof,trigger,parameter] containing all of the centroid information for each hit.
    '''
    t1 = time.time()
    if read_line_num is None:
        data_array = np.loadtxt(filename)
    else:
        data_array = np.loadtxt(filename,max_rows = read_line_num,skiprows=skiprows)
    
    t0a = time.time()
    if show_bar:
        print(f'readin time: {t0a-t1:.2f} sec')

    param_number = 0   
    
    trigger_lines = np.argwhere(np.sum(data_array[:,1:],axis=1)<-1).flatten()
    
    block_sizes = np.diff(trigger_lines)-1
    
    #### deals with offset by 260 pixels in y in a backwards compatible way #############
    i = 0
    while i < data_array.shape[0]:
        if i not in trigger_lines:
            first_non_trigger_row = i
            break
        i += 1
    
    if data_array[i,2] > 259:
        data_array -= np.array([0,0,260,0]) # fix the fact that y is off by 260
        
    #### end offset by 260 in y backwards compatibility section ##########################

    
    #### ToT-ToF correction section, change these parameters if VMI conditions change ####
    if tottofcorr:
        fix_y = np.argwhere((data_array[:,2]>=193.5) & (data_array[:,2]<=203.5) | (data_array[:,2]>=181.5) & (data_array[:,2]<=183.5)).flatten()
        data_array[fix_y,0] -= 25*1e-9

        fitty = np.array([11.9665668, 61.44000884, 4.16635999, 0.92730988])
        offset = 4.183831122463326

        mask = np.ones(data_array.shape[0], dtype=bool)
        mask[trigger_lines] = False

        cent_mod_val = np.zeros_like(data_array[:,1])
        cent_mod_val[mask] = tottof_fit_func(data_array[mask,1], *fitty)*1e-6
        cent_tof_mod = data_array[:,0] - cent_mod_val + offset*1e-6

        data_array[mask, 0] = cent_tof_mod[mask]
        t0 = time.time()
        if show_bar:
            print(f'correction time: {t0-t0a:.2f} sec')
        t0a = t0
    
    #############################################

    data_array = data_array[data_array[:, 0].argsort()]
    t0 = time.time()
    if show_bar:
        print(f'sort time: {t0-t0a:.2f} sec')
    
    
    
    
    #### This section does the batched GPU centroiding #################
    
    data_array = torch.tensor(data_array,device='cuda:0')

    centroids = torch.zeros((int(data_array.shape[0]/5),6),device='cuda:0')
    num_triggers = len(trigger_lines)
    if show_bar:
        print(f'Number of triggers : {num_triggers}')
    neighbors_times,centroiding_times,concatenation_times = [],[],[]
    if show_bar:
        progress = tqdm(total=100)
    num_centroids = 0
    current_percentage = 0
    temp = 0
    for trigger_num,trigger_index in zip(np.arange(start_trigger_num,start_trigger_num+len(trigger_lines[:-1]),batch_size),trigger_lines[:-1]):

        if int(100*(trigger_num-start_trigger_num)/num_triggers)>current_percentage and show_bar:
            progress.update(1)
            progress.set_description(f"Trigger {trigger_num + 1}/{num_triggers}")
            progress.set_postfix({'Centroids so far' : num_centroids})
            current_percentage += 1
        
        max_size = np.max(block_sizes[trigger_num-start_trigger_num:trigger_num-start_trigger_num+batch_size])
        block_batch = torch.zeros((max_size,4,batch_size),dtype=torch.float64,device='cuda:0')
        
        tot_offset = 0.001*torch.arange(max_size).cuda()
        tot_offset2 = [tot_offset * (1000*tot_offset<block_size) for block_size in block_sizes[trigger_num-start_trigger_num:trigger_num-start_trigger_num+batch_size]]
        tot_offset2 = torch.stack(tot_offset2).transpose(1,0)
        
        for i in range(batch_size):
            try:
                block_size = block_sizes[trigger_num-start_trigger_num+i]
                block_batch[:block_size,:,i] = data_array[trigger_lines[trigger_num-start_trigger_num+i]+1:trigger_lines[trigger_num-start_trigger_num+i+1]]
            except Exception as e:
                print(e)
                block_batch = block_batch[:,:,:i]
        block_batch[:,1,:] += tot_offset2

        try:
            neighbors_batch = get_neighbors(block_batch,size=centroid_area_size,tsep=centroid_time_size)
        except:
            continue
        
        local_maxima_filter_batch = get_local_maxima(block_batch,neighbors_batch)
        block_batch[:,1,:] -= tot_offset2
        
        num_local_maxima_batch = torch.sum(local_maxima_filter_batch,dim=0)
        lmfb = torch.nonzero(local_maxima_filter_batch,as_tuple=True)

        centroids_x_batch = (torch.sum(block_batch[:,3]*neighbors_batch*block_batch[:,1],axis=1)/torch.sum(block_batch[:,1]*neighbors_batch,axis=1))
        centroids_y_batch = (torch.sum(block_batch[:,2]*neighbors_batch*block_batch[:,1],axis=1)/torch.sum(block_batch[:,1]*neighbors_batch,axis=1))
        centroids_x_batch = centroids_x_batch[lmfb]
        centroids_y_batch = centroids_y_batch[lmfb]

        trigger_vals_batch = torch.tensor([data_array[trigger_lines[trigger_num-start_trigger_num+batch_trigger_num],0] for batch_trigger_num in range(local_maxima_filter_batch.shape[-1])],device=local_maxima_filter_batch.device,dtype=torch.float64)
        trigger_nums_batch = (local_maxima_filter_batch*torch.arange(trigger_num,trigger_num+local_maxima_filter_batch.shape[-1],1,device=local_maxima_filter_batch.device,dtype=torch.float64))[lmfb]
        triggers_batch = (local_maxima_filter_batch*trigger_vals_batch)[lmfb]
        
        tot_hits_batch = block_batch[:,1][lmfb]
        toa_hits_batch = block_batch[:,0][lmfb]-triggers_batch.cuda()

        param_nums = torch.ones(trigger_nums_batch.shape[0]).cuda()*(param_number-1)
        
        centroids_trigger = torch.stack((centroids_x_batch,centroids_y_batch,tot_hits_batch,toa_hits_batch,trigger_nums_batch.cuda(),param_nums),dim=-1)
        centroids_trigger = centroids_trigger[centroids_trigger[:, 4].argsort()]

        try:
            centroids[num_centroids:num_centroids+len(centroids_trigger)] = centroids_trigger
            num_centroids += len(centroids_trigger)
        except Exception as e:
            centroids = torch.cat((centroids,centroids_trigger),dim=0)
    
    #### end batched GPU centroiding section ##################

    centroids[:,3] = centroids[:,3]*1e6
    centroids = centroids[:num_centroids].cpu().numpy()
    if show_bar:
        progress.update(1)
        progress.set_description(f"Trigger {num_triggers}/{num_triggers}")
        progress.set_postfix({'Centroids so far' : num_centroids})
        current_percentage += 1
        progress.close()
        print('Done!')

    return centroids

def centroid_multi_scan(foldy,batch_size=1,tottofcorr=True,cent_filename='all_centroids',
                        checkpoint_interval=10,spec_delays=True,skip_last=False,max_gb=3):
    '''
    Runs get_file_batched in a loop, assuming a particular file structure that comes from our parameter scan code - specifically, that the .txt filenames are like 'file{filenumber}_param_{parameternumber}_000000.txt'. Adjusts the trigger numbers of each file so they are consistent with the previous ones. Reads the parameter number from the file name and saves it in centroids. Additionally creates (optional, but standard) 2 new columns in centroids for the file number and delay, which is mapped from the file number after measuring spectral interference fringes. Can be run with or without existing centroids.npy file, will load if it exists and append, or create one if it doesn't. 
    Parameters
    ~~~~~~~~~~
    foldy : str, directory name where .tpx3 and .txt files are stored
    batch_size : int (default 1), number of triggers to process at once within a single batch on the GPU. Usually works well if this value is set to 10 or so.
    tottofcorr : bool (default True), whether to apply the ToT-ToF correction to individual pixels before centroiding.
    cent_filename : str (default 'all_centroids'), name of the .npy file where the centroids will be saved. This function appends to the array as new data comes in.
    checkpoint_interval : int (default 10), number of files to centroid at once before saving the data to all_centroids.npy
    spec_delays : bool (default True), whether to make 2 extra columns to store true pulse pair delays from spectrometer fringes in a different file. This is a special case to the Bucksbaum lab and should be treated carefully for other users.
    skip_last : bool (default False), whether to attempt to centroid the file that is currently being collected. This should be set to True for live data processing, and then set to False and rerun one more time once collection is finished.
    max_gb : int or float (default 3), maximum file size of all_centroids.npy in GB before starting a new file (i.e. all_centroids_2.npy will be started once all_centroids_1.npy exceeds this file size)
    Returns
    ~~~~~~~~~~
    centroids : numpy array containing all centroids processed by the function when it is run
    '''
    data_folder = Path(foldy)
        
#     file_pattern = f"{foldy}/{cent_filename}_*.npy"
    existing_files = sorted(Path(foldy).glob(f"{cent_filename}_*.npy"))

    if existing_files:
        latest_file = existing_files[-1]
        latest_num = int(latest_file.stem.split("_")[-1])
    else:
        latest_file = None
        latest_num = 1

    if latest_file and latest_file.stat().st_size < max_gb * 10**9:
        centroids = np.load(latest_file)
        last_trig_num = int(centroids[-1, 4])
    else:
        if spec_delays:
            centroids = np.zeros((1, 8))
        else:
            centroids = np.zeros((1, 6))
        last_trig_num = 0
        
    if latest_file and latest_file.stat().st_size >= max_gb * 10**9:
        latest_num += 1
        
    trigs_file_count_path = data_folder / 'trigs_file_count.npy'
    if trigs_file_count_path.exists():
        trigs_file_count = np.load(trigs_file_count_path)
    else:
        trigs_file_count = np.zeros((0, 4))

    txt_files = list(data_folder.glob('file*.txt'))
    txt_files.sort()
    if skip_last:
        txt_files.pop()
        
    num_files = len(txt_files)

    progress = tqdm(total=num_files)
    last_file_deleted = -1
    for jj, txt_file in enumerate(txt_files):
        param_num_s = str(txt_file).split('_')[-2]
        param_num = int(param_num_s)
        filly = (str(txt_file).split('/')[-1])
        progress.set_description(filly)
        
        file_num_s = str(txt_file).split('_')[-4]
        file_num = int(file_num_s[-6:])

        cents1 = read_file_batched(txt_file,batch_size=batch_size,tottofcorr=tottofcorr,show_bar=False)
        cents1[:,4] += last_trig_num
        cents1[:,5] = param_num
        if np.shape(cents1)[0] > 0:
            new_last_trig_num = int(cents1[-1,4])
            trigs_in_file = new_last_trig_num - last_trig_num
            last_trig_num = new_last_trig_num
        else:
            trigs_in_file = 0

        ## want to make this nicer but need to change read_file_batched to output 8 cols then
        if spec_delays:
            cents1a = np.full_like(cents1[:,0:2],np.nan)
            cents1b = np.hstack((cents1,cents1a))
            cents1 = cents1b.copy()
            cents1[:,6] = file_num
            
        centroids = np.vstack((centroids,cents1))

        progress.update(1)
        num_centroids = np.shape(centroids)[0]
        progress.set_postfix({'Centroids so far' : num_centroids})
        
        trigs_file_count = np.vstack((trigs_file_count, [param_num, file_num, 0, trigs_in_file]))
        
        if (not jj%checkpoint_interval) or (jj == len(txt_files)-1):
            files_to_del = txt_files[last_file_deleted+1:jj+1]
            np.save(f'{data_folder}/{cent_filename}_{latest_num}.npy',centroids)
            np.save(trigs_file_count_path, trigs_file_count)
            for ff in files_to_del:
                ff.unlink()
            last_file_deleted = jj

    progress.close()
    np.save(f'{data_folder}/{cent_filename}_{latest_num}.npy',centroids)
    np.save(trigs_file_count_path, trigs_file_count)
    
    return centroids