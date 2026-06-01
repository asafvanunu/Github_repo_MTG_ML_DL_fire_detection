# %%
import rioxarray
import segmentation_models_pytorch as smp
import torch
import numpy as np
import warnings
from pathlib import Path
import pandas as pd
import joblib
import os
import math
import torch.nn as nn

# %%
def get_my_neighbores(array, row_i, col_j, distance=1, value_or_index="value"):
    """This function returns the neighbors of a pixel in a raster image

    Args:
        array (numpy array): the raster image
        row_i (int): the row index of the pixel
        col_j (int): the column index of the pixel 
        distance (int, optional): the distance of the neighbors 1 is 3x3 and 2 is 5x5. Defaults to 1.
        value_or_index (str, optional): return the value of the neighbors or the index of them. Defaults to "value".
    """
    if (0>distance) or (distance>2): ## check if the distance is between 0 and 2
        raise ValueError("The distance should be between 0 and 2")
    if value_or_index not in ["value", "index"]: ## check if the value_or_index is either value or index
        raise ValueError("The value_or_index should be either value or index")
    if isinstance(array, np.ndarray) == False: ## check if the array is a numpy array
        raise ValueError("The array should be a numpy array")
    
    array_shape = array.shape ## get the shape of the array
    if (row_i < 0) or (row_i >= array_shape[0]): ## check if the row index is within the range of the array
        raise ValueError("The row index is out of range")
    if (col_j < 0) or (col_j >= array_shape[1]): ## check if the column index is within the range of the array
        raise ValueError("The column index is out of range")
    
    pixel_loc = [row_i, col_j] ## get the location of the pixel
    if distance == 1: ## check if the distance is 1 (3x3)
        neighbors = [[row_i-1, col_j-1],
                     [row_i-1, col_j],
                     [row_i-1, col_j+1],
                     [row_i, col_j-1],
                     [row_i, col_j],
                     [row_i, col_j+1],
                     [row_i+1, col_j-1],
                     [row_i+1, col_j],
                     [row_i+1, col_j+1]]
        replace_list = [] ## create an empty list
        for i in range(len(neighbors)): ## loop through the neighbors
            ## check if the neighbors are out of the range of the array
            if (neighbors[i][0] < 0) or (neighbors[i][0] >= array_shape[0]) or (neighbors[i][1] < 0) or (neighbors[i][1] >= array_shape[1]):
                replace_list.append(neighbors[i]) ## add the neighbors to the replace_list
                
            
        if value_or_index == "value": ## check if the value_or_index is value
            list_of_values = [] ## create an empty list
            for i in range(len(neighbors)): ## loop through the neighbors
                if neighbors[i] in replace_list: ## check if the neighbors are out of the range of the array
                    list_of_values.append(np.nan) ## add nan to the list_of_values
                else: ## if the neighbors are in the range of the array
                    pixel_value = array[neighbors[i][0], neighbors[i][1]] ## get the value of the pixel
                    list_of_values.append(pixel_value) ## add the value to the list_of_values
            return list_of_values ## return the list_of_values
        elif value_or_index == "index": ## check if the value_or_index is index
            return neighbors
        
    elif distance == 2: ## check if the distance is 2 (5x5)
        neighbors = [[row_i-2, col_j-2],
                     [row_i-2, col_j-1],
                     [row_i-2, col_j],
                     [row_i-2, col_j+1],
                     [row_i-2, col_j+2],
                     [row_i-1, col_j-2],
                     [row_i-1, col_j-1],
                     [row_i-1, col_j],
                     [row_i-1, col_j+1],
                     [row_i-1, col_j+2],
                     [row_i, col_j-2],
                     [row_i, col_j-1],
                     [row_i, col_j],
                     [row_i, col_j+1],
                     [row_i, col_j+2],
                     [row_i+1, col_j-2],
                     [row_i+1, col_j-1],
                     [row_i+1, col_j],
                     [row_i+1, col_j+1],
                     [row_i+1, col_j+2],
                     [row_i+2, col_j-2],
                     [row_i+2, col_j-1],
                     [row_i+2, col_j],
                     [row_i+2, col_j+1],
                     [row_i+2, col_j+2]]
        replace_list = [] ## create an empty list
        for i in range(len(neighbors)): ## loop through the neighbors
            ## check if the neighbors are out of the range of the array
            if (neighbors[i][0] < 0) or (neighbors[i][0] >= array_shape[0]) or (neighbors[i][1] < 0) or (neighbors[i][1] >= array_shape[1]):
                replace_list.append(neighbors[i])    ## add the neighbors to the replace_list
        
            
        if value_or_index == "value": ## check if the value_or_index is value
            list_of_values = [] ## create an empty list
            for i in range(len(neighbors)): ## loop through the neighbors
                if neighbors[i] in replace_list: ## check if the neighbors are out of the range of the array
                    list_of_values.append(np.nan) ## add nan to the list_of_values
                else: ## if the neighbors are in the range of the array
                    pixel_value = array[neighbors[i][0], neighbors[i][1]] ## get the value of the pixel
                    list_of_values.append(pixel_value) ## add the value to the list_of_values
            return list_of_values ## return the list_of_values
        elif value_or_index == "index": ## check if the value_or_index is index
            return neighbors ## return the neighbors

# %%
def MTG_all_pixel_location_list(MTG_Fire_Index_array):
    """This function get MTG fire index array and return all the pixel locations in a list for example [[0,0], [0,1], ...]

    Args:
        MTG_Fire_Index_array (array): MTG Fire Index array
    """
    
    if isinstance(MTG_Fire_Index_array, np.ndarray) == False:
        raise ValueError("MTG_Fire_Index_array should be a numpy array")
    
    MTG_all_pixel_location_list = [] ## list of all the pixel locations
    image_shape = MTG_Fire_Index_array.shape ## get the shape of the MTG Fire Index array
    for i in range(image_shape[0]): ## loop through the rows
        for j in range(image_shape[1]): ## loop through the columns
            MTG_all_pixel_location_list.append([i,j]) ## append the location to the MTG_all_pixel_location_list
    return MTG_all_pixel_location_list ## return the MTG_all_pixel_location_list

# %%
def remove_cloud_neighbores(band_array, cloud_mask_array, row_i, col_j, distance,cloud_probability_list, statistic):
    """This function gets the band array, cloud mask array, row index, column index, distance, and statistic and return a statistic of the pixel neighbore withot clouds and without the pixel itself

    Args:
        band_array (array): The band array
        cloud_mask_array (array): ACM array
        row_i (int): fire pixel row index
        col_j (int): fire pixel column index
        distance (int): the buffer distance 1 for 3x3 and 2 for 5x5
        cloud_probability_list (list): list of cloud probabilities of CLM to be excluded for example [2,3]
        statistic (string): the statistic to calculate for example "mean". Avilable statistics are "mean", "median, "std, "max", "min"
    """
    if isinstance(band_array, np.ndarray) == False:
        raise ValueError("band_array should be a numpy array")
    if isinstance(cloud_mask_array, np.ndarray) == False:
        raise ValueError("cloud_mask_array should be a numpy array")

    all_clouds = -999
    all_nan = -888
    if statistic == "value": ## check if the statistic is value
        return band_array[row_i, col_j] ## return the value of the pixel
    
    else: ## if the statistic is not value
    
        band_values = get_my_neighbores(array=band_array, row_i=row_i, col_j=col_j, distance=distance, value_or_index="value") ## get the band values
        cloud_values = get_my_neighbores(array=cloud_mask_array, row_i=row_i, col_j=col_j, distance=distance, value_or_index="value") ## get the cloud values
    
        if distance == 1:
            band_values.pop(4) ## remove the center pixel
            cloud_values.pop(4) ## remove the center pixel
        elif distance == 2:
            band_values.pop(12) ## remove the center pixel
            cloud_values.pop(12) ## remove the center pixel
        
        band_values = np.array(band_values) ## convert the band values to a numpy array
        cloud_values = np.array(cloud_values) ## convert the cloud values to a numpy array
    
        is_cloud = np.isin(cloud_values, cloud_probability_list) ## check if the cloud values are in the cloud_probability_list
        filter_band_values = band_values[~is_cloud] ## filter the band values. Take only the values that are not clouds
    
        if len(filter_band_values) == 0: ## check if the filter_band_values is empty
            #print(f"in pixel {row_i}, {col_j} all of the neighbors are clouds") ## print a message
            return all_clouds ## return -999
    
        else: ## if the filter_band_values is not empty
            with warnings.catch_warnings(): ## catch the warnings
                warnings.simplefilter("ignore", category=RuntimeWarning) ## ignore the runtime warnings of nan
                if statistic == "mean": ## check if the statistic is mean
                    mean = np.nanmean(filter_band_values) ## calculate the mean of the filter_band_values
                    if np.isnan(mean): ## check if the mean is nan
                        return all_nan ## return -888
                    else: ## if the mean is not nan
                        return mean ## return the mean
                elif statistic == "median": ## check if the statistic is median
                    median = np.nanmedian(filter_band_values)
                    if np.isnan(median):
                        return all_nan
                    else:
                        return median
                elif statistic == "std": ## check if the statistic is std
                    std = np.nanstd(filter_band_values)
                    if np.isnan(std):
                        return all_nan
                    else:
                        return std
                elif statistic == "max": ## check if the statistic is max
                    max_value = np.nanmax(filter_band_values)
                    if np.isnan(max_value):
                        return all_nan
                    else:
                        return max_value
                elif statistic == "min": ## check if the statistic is min
                    min_value = np.nanmin(filter_band_values)
                    if np.isnan(min_value):
                        return all_nan
                    else:
                        return min_value

# %%
def get_fire_pixel_values_in_all_bands_for_fire_event(pixel_location_list, multispectral_path, CLM_path,
                                                            MTG_date_time, cloud_probability_list=[3]):
    r"""This function gets the full pixel location list and the multispectral and AOI paths and return a df with pixel values

    Args:
        pixel_location_list (list): full pixel location list for all of the image. for example [[1,2], [3,4]]
        multispectral_path (str): cropped multispectral tif full path for example "cloud/2025_07_25_01_10/2025_07_25_01_10.tif"
        CLM_path (str): cropped CLM tif full path for example "cloud/2025_07_25_01_10/2025_07_25_01_10.tif"
        MTG_date_time (str): MTG date time for example '2023-01-01 07:51'
    """
    if isinstance(pixel_location_list, list) == False:
        raise ValueError("pixel_location_list should be a list")
    if isinstance(multispectral_path, str) == False:
        raise ValueError("multispectral_path should be a string")
    if isinstance(CLM_path, str) == False:
        raise ValueError("CLM_path should be a string")
    if isinstance(MTG_date_time, str) == False:
        raise ValueError("MTG_date_time should be a string")
    
    
    CLM = rioxarray.open_rasterio(CLM_path) ## crop the GOES image using the VIIRS image
    CLM_values = CLM.values[0] ## get the values of the CLM
    multispectral = rioxarray.open_rasterio(multispectral_path) ## crop the GOES image using the VIIRS image
    SWIR = multispectral[2] ## get the values of the SWIR band
    TIR = multispectral[3] ## get the values of the TIR band
    FI = (SWIR.values - TIR.values) / (SWIR.values + TIR.values) ## calculate the fire index 
    band_list = ["VIS_06","NIR_2.2", "IR_3.8", "IR_10.5"] ## Get the multispectral band names
    indices_list = ["FI"] ## list of indices for example fire index (FI)
    band_iteration_list = band_list + indices_list ## combine the band_list and indices_list
    statistics_list = ["value", "mean", "median", "std","min","max"] ## list of statistics for example value, mean, median, std
    
    row_list = [] ## list to store the row values
    col_list = [] ## list to store the column values
    CLM_list = [] ## list to store the CLM values

    
    for loc in pixel_location_list: ## loop through the pixel location list
        row = loc[0] ## get the row location
        col = loc[1] ## get the column location
        row_list.append(row) ## append the row to the row_list
        col_list.append(col) ## append the column to the col_list
        if np.isnan(FI[row, col]): ## check if the FI value is NaN
            CLM_list.append(-888) ## append NaN to the CLM_list
        else: ## if the FI value is not NaN    
            CLM_value = CLM_values[row, col] ## get the value of the CLM
            CLM_list.append(CLM_value) ## append the CLM value to the CLM_list
        
    d = {} ## dictionary to store the values
    d["row"] = row_list ## add the row_list to the dictionary
    d["col"] = col_list ## add the col_list to the dictionary
    for band in band_iteration_list: ## loop through the band_iteration_list
        if band == "FI": ## check if the band is FI
            FI_value_list = [] ## list to store the fire index values
            FI_n_mean_list = [] ## list to store the fire index mean values
            FI_n_median_list = [] ## list to store the fire index median values
            FI_n_std_list = [] ## list to store the fire index std values
            FI_n_min_list = [] ## list to store the fire index min values
            FI_n_max_list = [] ## list to store the fire index max values
            for loc in pixel_location_list: ## loop through the pixel location list
                row = loc[0] ## get the row location
                col = loc[1] ## get the column location
                if np.isnan(FI[row, col]): ## check if the FI value is NaN
                    FI_value_list.append(-888) ## append NaN to the FI_value_list
                    FI_n_mean_list.append(-888) ## append NaN to the FI_n_mean_list
                    FI_n_median_list.append(-888) ## append NaN to the FI_n_median_list
                    FI_n_std_list.append(-888) ## append NaN to the FI_n_std_list
                    FI_n_min_list.append(-888) ## append NaN to the FI_n_min_list
                    FI_n_max_list.append(-888) ## append NaN to the FI_n_max_list
                else: ## if the FI value is not NaN
                    for stat in statistics_list: ## loop through the statistics_list
                    ## get the neighbors of the pixel including the pixel itself
                        stat_value = remove_cloud_neighbores(band_array=FI, 
                                                              cloud_mask_array=CLM_values,
                                                              row_i=row,
                                                              col_j=col,
                                                              distance=1,
                                                              cloud_probability_list=cloud_probability_list,
                                                              statistic=stat) ## get the neighbors of the pixel
                        if stat == "value":
                            FI_value_list.append(stat_value) ## append the value to the FI_value_list
                        elif stat == "mean": ## check if the stat is mean
                            FI_n_mean_list.append(stat_value) ## append the mean to the FI_n_mean_list
                        elif stat == "median": ## check if the stat is median
                            FI_n_median_list.append(stat_value) ## append the median to the FI_n_median_list
                        elif stat == "std": ## check if the stat is std
                            FI_n_std_list.append(stat_value) ## append the std to the FI_n_std_list
                        elif stat == "min": ## check if the stat is min
                            FI_n_min_list.append(stat_value) ## append the min to the FI_n_min_list
                        elif stat == "max": ## check if the stat is max
                            FI_n_max_list.append(stat_value) ## append the max to the FI_n_max_list
            d[f"FI_value"] = FI_value_list ## add the FI_value_list to the dictionary
            d[f"FI_mean"] = FI_n_mean_list ## add the FI_n_mean_list to the dictionary
            d[f"FI_median"] = FI_n_median_list ## add the FI_n_median_list to the dictionary
            d[f"FI_std"] = FI_n_std_list ## add the FI_n_std_list to the dictionary
            d[f"FI_min"] = FI_n_min_list ## add the FI_n_min_list to the dictionary
            d[f"FI_max"] = FI_n_max_list ## add the FI_n_max_list to the dictionary
        else: ## if the band is not FI
            band_number = band ## get the band name for example VIS_06
            band_index = band_list.index(band) ## get the band index
            ## crop the MTG image using the VIIRS image
            B = multispectral[band_index] ## get the band   
            band_array = B.values ## get the values of the band
            band_value_list = [] ## list to store the band values
            band_n_mean_list = [] ## list to store the band mean values
            band_n_median_list = [] ## list to store the band median values
            band_n_std_list = [] ## list to store the band std values
            band_n_min_list = [] ## list to store the band min values
            band_n_max_list = [] ## list to store the band max values
            for loc in pixel_location_list: ## loop through the pixel location list
                row = loc[0] ## get the row location
                col = loc[1] ## get the column location
                if np.isnan(FI[row, col]): ## check if the FI value is NaN
                    band_value_list.append(-888) ## append NaN to the band_value_list
                    band_n_mean_list.append(-888) ## append NaN to the band_n_mean_list
                    band_n_median_list.append(-888) ## append NaN to the band_n_median_list
                    band_n_std_list.append(-888) ## append NaN to the band_n_std_list
                    band_n_min_list.append(-888) ## append NaN to the band_n_min_list
                    band_n_max_list.append(-888) ## append NaN to the band_n_max_list
                else: ## if the FI value is not NaN
                    for stat in statistics_list: ## loop through the statistics_list
                    ## get the neighbors of the pixel including the pixel itself
                        stat_value = remove_cloud_neighbores(band_array=band_array, 
                                                              cloud_mask_array=CLM_values,
                                                              row_i=row,
                                                              col_j=col,
                                                              distance=1,
                                                              cloud_probability_list=cloud_probability_list,
                                                              statistic=stat) ## get the neighbors of the pixel
                        if stat == "value": ## check if the stat is value
                            band_value_list.append(stat_value) ## append the value to the band_value_list
                        elif stat == "mean": ## check if the stat is mean
                            band_n_mean_list.append(stat_value) ## append the mean to the band_n_mean_list
                        elif stat == "median": ## check if the stat is median
                            band_n_median_list.append(stat_value) ## append the median to the band_n_median_list
                        elif stat == "std": ## check if the stat is std
                            band_n_std_list.append(stat_value) ## append the std to the band_n_std_list
                        elif stat == "min": ## check if the stat is min
                            band_n_min_list.append(stat_value)
                        elif stat == "max": ## check if the stat is max
                            band_n_max_list.append(stat_value)
            d[f"{band_number}_value"] = band_value_list ## add the band_value_list to the dictionary
            d[f"{band_number}_mean"] = band_n_mean_list ## add the band_n_mean_list to the dictionary
            d[f"{band_number}_median"] = band_n_median_list ## add the band_n_median_list to the dictionary
            d[f"{band_number}_std"] = band_n_std_list ## add the band_n_std_list to the dictionary
            d[f"{band_number}_min"] = band_n_min_list ## add the band_n_min_list to the dictionary
            d[f"{band_number}_max"] = band_n_max_list ## add the band_n_max_list to the dictionary
        
    df = pd.DataFrame(d) ## create a DataFrame from the dictionary
    df[f"CLM_value"] = CLM_list ## add the CLM_list to the DataFrame
    file_name = os.path.basename(multispectral_path).split("\\")[-1] ## get the base name of the MCMI file 
    file_name_list = np.repeat(file_name, len(df)) ## repeat the file name for the length of the DataFrame
    date_time_list = np.repeat(MTG_date_time, len(df)) ## repeat the date time for the length of the DataFrame
    df.insert(0, f"multispectral_file", file_name_list) ## insert the file name to the first column
    df.insert(1, f"MTG_date_time", date_time_list) ## insert the date time to the second column
    return df ## return the DataFrame

# %%
def create_fire_event_df(multispectral_path, 
                          CLM_path ,
                          MTG_date_time, 
                          cloud_probability_list=[3]):
    r"""This function gets the multispectral, CAP, CLM, and return the validation image DataFrame

    Args:
        multispectral_path (str): path to the cropped tif multispectral file for example "G:\MTG_project\Germany\MTG_clipped_for_ML\1km\multispectral\2025_07_25_01_10\2025_07_25_01_10.tif"
        CLM_path (str): path of the cropped tif CLM file for example "G:\MTG_project\Germany\MTG_clipped_for_ML\1km\cloud\2025_07_25_01_10\2025_07_25_01_10.tif"
        MTG_date_time (str): MTG date time for example '2023-01-01 07:51'
        cloud_probability_list (list): list of cloud probabilities of CLM to be excluded for example [3]
    """
    ## check the input types
    if isinstance(multispectral_path, str) == False:
        raise ValueError("multispectral_path should be a string")
    if isinstance(CLM_path, str) == False:
        raise ValueError("CLM_path should be a string")
    if isinstance(MTG_date_time, str) == False:
        raise ValueError("MTG_date_time should be a string")
    
    
    print(f"Now working of MTG time stamp: {MTG_date_time}") ## print the MTG date time
    ## Open MTG bands
    multispectral = rioxarray.open_rasterio(multispectral_path) ## Open MCMI
    SWIR = multispectral[2] ## get the values of the SWIR band
    TIR = multispectral[3] ## get the values of the TIR band
    FI = (SWIR.values - TIR.values) / (SWIR.values + TIR.values) ## calculate the fire index 
    
    
    MTG_pixel_list = MTG_all_pixel_location_list(MTG_Fire_Index_array=FI)
    print(f"MTG pixel list is done for MTG time stamp: {MTG_date_time}") ## print a message
    
    print(f"staring to genrate fire pixel values for MTG time stamp: {MTG_date_time}")
    ## Get the fire pixel values
    print(f"Starting to get the fire pixel values for MTG time stamp: {MTG_date_time}")
    df_pixels = get_fire_pixel_values_in_all_bands_for_fire_event(pixel_location_list=MTG_pixel_list,
                                        multispectral_path=multispectral_path,
                                        CLM_path=CLM_path,
                                        MTG_date_time=MTG_date_time,
                                        cloud_probability_list=cloud_probability_list)
       
    print(f"done. df is ready for MTG time stamp: {MTG_date_time}")
    return df_pixels ## return the train_df
        
    

# %%
def predict_fire_using_ML(multispectral_path, CLM_path, MTG_date_time, ML_model_path, save_predicted_image=True, save_predicted_image_path=None):
    """This function gets multispectral_path, CLM_path, MTG_date_time, ML_model_path, and return the predicted fire pixels using the ML model
    Args:
        multispectral_path (str): path to the cropped tif multispectral file for example "multispectral\2025_07_25_01_10\2025_07_25_01_10.tif"
        CLM_path (str): path of the cropped tif CLM file for example "cloud\2025_07_25_01_10\2025_07_25_01_10.tif"
        MTG_date_time (str): MTG date time for example '2023-01-01 07:51'
        ML_model_path (str): path to the ML model
        save_predicted_image (bool): whether to save the predicted image
        save_predicted_image_path (str): path to save the predicted image
    """
    if isinstance(multispectral_path, str) == False:
        raise ValueError("multispectral_path should be a string")
    if isinstance(CLM_path, str) == False:
        raise ValueError("CLM_path should be a string")
    if isinstance(MTG_date_time, str) == False:
        raise ValueError("MTG_date_time should be a string")
    if isinstance(ML_model_path, str) == False:
        raise ValueError("ML_model_path should be a string")
    if isinstance(save_predicted_image, bool) == False:
        raise ValueError("save_predicted_image should be a boolean")
    if save_predicted_image == True:
        if not save_predicted_image_path is None:
            if isinstance(save_predicted_image_path, str) == False:
                raise ValueError("save_predicted_image_path should be a string")
    
    cloud_probability_list=[3] ## list of cloud probabilities of CLM to be excluded for example [3]
    MTG_date_time_string_out = MTG_date_time.replace(" ", "_").replace(":", "_") ## replace the space and colon in the MTG date time string to underscore for example '2023-01-01_07_51' 
    print(f"Now working of MTG time stamp: {MTG_date_time}") ## print the MTG date time
    ## create the fire event DataFrame
    df_pixels = create_fire_event_df(multispectral_path=multispectral_path,
                                    CLM_path=CLM_path,
                                    MTG_date_time=MTG_date_time,
                                    cloud_probability_list=cloud_probability_list)
    
    grid_search = joblib.load(ML_model_path) ## Load the grid search object
    ML_model = grid_search.best_estimator_ ## Get the best estimator from the grid search
    name_filter_list = [] ## create a list to keep the names of the columns
    
    for band in ["VIS_06","NIR_2.2","IR_3.8","IR_10.5","FI"]:
        for stat in ["value", "mean", "median", "std", "min", "max"]: ## loop over statistics
            if band == "FI": ## if the band is FI, add the name to the list
                name_filter_list.append(f"FI_{stat}") ## add the name to the list
            else: ## if the band is not FI, add the name to the list
                name_filter_list.append(f"{band}_{stat}") ## add the name to the list
    
    threshold = 0.95 ## set the threshold for fire pixel classification
    ML_df = df_pixels[name_filter_list] ## Filter the ML DataFrame to keep only the columns in the name filter list
    y_prob = ML_model.predict_proba(ML_df)[:, 1] ## Get the predicted probabilities for the positive class
    y_pred = (y_prob >= threshold).astype(int) ## Apply the threshold to the predicted probabilities to get the predicted labels
    CLM = rioxarray.open_rasterio(CLM_path) ## Open the CLM file
    array_shape = CLM.shape ## Get the shape of the CLM array
    y_pred_reshaped = y_pred.reshape(array_shape) ## Reshape the predicted labels to the shape of the CAP array
    pred_array_out = CLM.copy() ## Create a copy of the CAP array to store the predicted labels
    pred_array_out.values = y_pred_reshaped ## Replace the values in the output array
    pred_array_out.values = pred_array_out.values.astype(np.float32) ## Convert the values in the output array to float32
    pred_array_out.values[np.isnan(CLM.values)] = np.nan ## Set the values in the output array to NaN where the CLM values are NaN
    if save_predicted_image == False: ## check if the save_predicted_image is False
        print(f"Predicted raster is ready for MTG time stamp: {MTG_date_time}") ## print a message
        return pred_array_out ## return the predicted array
    if save_predicted_image == True: ## check if the save_predicted_image is True
        if save_predicted_image_path is None: ## check if the save_predicted_image_path is None
            pred_array_out.rio.to_raster(f"predicted_fire_{MTG_date_time_string_out}.tif") ## save the predicted array to a raster file
            print(f"Predicted image saved at {os.getcwd()}") ## print a message
        else: ## if the save_predicted_image_path is not None
            posix = Path(save_predicted_image_path).as_posix() ## convert the save_predicted_image_path to posix format 
            output_path = f"{posix}" ## create the output path
            if not os.path.exists(save_predicted_image_path): ## check if the save_predicted_image_path exists
                os.makedirs(save_predicted_image_path) ## create the directory if it does not exist
                print(f"Directory {save_predicted_image_path} created") ## print a message
            pred_array_out.rio.to_raster(f"{output_path}/predicted_fire_{MTG_date_time_string_out}.tif") ## save the predicted array to a raster file
            print(f"Predicted image saved to {output_path}") ## print a message
            return pred_array_out ## return the predicted array

# %%
def pad_to_multiple(tensor: torch.Tensor, multiple: int = 128):
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padded = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)
    return padded, h, w

# %%
def predict_fire_using_DL(multispectral_path, MTG_date_time, DL_model_path, save_predicted_image=True, save_predicted_image_path=None):
    """This function gets multispectral_path, MTG_date_time, DL_model_path, and cloud_probability_list and return the predicted fire pixels using the DL model
    Args:
        multispectral_path (str): path to the cropped tif multispectral file for example "multispectral\2025_07_25_01_10\2025_07_25_01_10.tif"
        MTG_date_time (str): MTG date time for example '2023-01-01 07:51'
        DL_model_path (str): path to the DL model
        save_predicted_image (bool): whether to save the predicted image
        save_predicted_image_path (str): path to save the predicted image
    """
    if isinstance(multispectral_path, str) == False:
        raise ValueError("multispectral_path should be a string")
    if isinstance(MTG_date_time, str) == False:
        raise ValueError("MTG_date_time should be a string")
    if isinstance(DL_model_path, str) == False:
        raise ValueError("DL_model_path should be a string")
    if isinstance(save_predicted_image, bool) == False:
        raise ValueError("save_predicted_image should be a boolean")
    if save_predicted_image == True:
        if not save_predicted_image_path is None:
            if isinstance(save_predicted_image_path, str) == False:
                raise ValueError("save_predicted_image_path should be a string")
    
    MTG_date_time_string_out = MTG_date_time.replace(" ", "_").replace(":", "_") ## replace the space and colon in the MTG date time string to underscore for example '2023-01-01_07_51' 
    print(f"Now working of MTG time stamp: {MTG_date_time}") ## print the MTG date time
    ## create the fire event DataFrame
    GLOBAL_MINS = np.array([ -0.351, -1.875, 139.341, 176.073], dtype=np.float32)  ## replace with real values
    GLOBAL_MAXS = np.array([130.927, 113.637, 459.553, 332.823], dtype=np.float32)  ## replace with real values
    mins = GLOBAL_MINS[:, np.newaxis, np.newaxis]        ## (4,1,1) for broadcasting
    maxs = GLOBAL_MAXS[:, np.newaxis, np.newaxis]        ## (4,1,1) for broadcasting
    threshold = 0.6 ## set the threshold for fire pixel classification
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ## Set the device to GPU if available, otherwise use CPU
    model  = smp.MAnet(encoder_name="mit_b2", encoder_weights=None, in_channels=4, classes=1) ## Create the model architecture
    model.load_state_dict(torch.load(DL_model_path, map_location=device, weights_only=True)) ## Load the trained model weights
    model = model.to(device) ## Move the model to the specified device
    model.eval() ## Set the model to evaluation mode
    
    tile = rioxarray.open_rasterio(multispectral_path) ## Open the multispectral image using rioxarray
    reference = tile[0].copy()
    img = tile.astype(np.float32).values                 # (4, H, W) numpy
    img = (img - mins) / (maxs - mins + 1e-8)
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    img = np.clip(img, 0.0, 1.0) 
    tensor                        = torch.tensor(img).unsqueeze(0).to(device)  # (1, 4, H, W)
    tensor_padded, orig_h, orig_w = pad_to_multiple(tensor, multiple=128) ## Pad the tensor to be a multiple of 128 for the model input
    with torch.no_grad(): ## Disable gradient calculation for inference
        output = model(tensor_padded) ## Get the model output
        prob   = torch.sigmoid(output) # Apply sigmoid to get probabilities
        pred   = (prob > threshold).squeeze().cpu().numpy().astype(np.float32) # (H_pad, W_pad)

    pred_cropped = pred[:orig_h, :orig_w]                                      # strip padding
    print(f"Prediction shape   : {pred_cropped.shape}")

    # ── Restore NaN mask + write into reference DataArray ────────────────────────
    nan_locs               = np.isnan(reference.values)
    reference.values       = pred_cropped
    reference.values[nan_locs] = np.nan

    # ── Save GeoTIFF ──────────────────────────────────────────────────────────────
    if save_predicted_image == False: ## check if the save_predicted_image is False
        print(f"Predicted raster is ready for MTG time stamp: {MTG_date_time}") ## print a message
        return reference ## return the predicted array
    
    if save_predicted_image == True: ## check if the save_predicted_image is True
        if save_predicted_image_path is None: ## check if the save_predicted_image_path is None
            reference.rio.to_raster(f"predicted_fire_{MTG_date_time_string_out}.tif", dtype="float32") ## save the predicted array to a raster file
            print(f"Predicted image saved at {os.getcwd()}") ## print a message
            return reference ## return the predicted array
        else: ## if the save_predicted_image_path is not None
            posix = Path(save_predicted_image_path).as_posix() ## convert the save_predicted_image_path to posix format 
            output_path = f"{posix}/predicted_fire_{MTG_date_time_string_out}.tif" ## create the output path
            if not os.path.exists(save_predicted_image_path): ## check if the save_predicted_image_path exists
                os.makedirs(save_predicted_image_path) ## create the directory if it does not exist
                print(f"Directory {save_predicted_image_path} created") ## print a message
            reference.rio.to_raster(output_path, dtype="float32")
            print(f"Saved prediction to: {output_path}")
            return reference ## return the predicted array

# %%



