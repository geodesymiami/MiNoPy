### Brief description of the steps: ###

minopyApp.py runs 9 steps sequentially from reading data to time series analysis. It uses [ISCE](https://github.com/isce-framework/isce2), [MintPy](https://github.com/insarlab/MintPy) and [PyAPS](https://github.com/AngeliqueBenoit/pyaps3) as extrenal modules and for correction steps, it uses MintPy.

You need to have a configuration text file with the options for each step like the [sample](https://github.com/geodesymiami/MiNoPy/blob/main/sample_input/PichinchaSenDT142.txt). For a complete list of options, run `minopyApp.py -H`

Run `minopyApp.py -h` for a quick help on steps.
For more details refer to the [example](https://nbviewer.jupyter.org/github/geodesymiami/MiNoPy/blob/main/tutorial/minopyApp.ipynb) tutorial.

1. The first step is to read/load the coregistered SLC data and geometry files in full resolution. For that, 
it is recommended to set the options with `minopy.load.*` in your template. The ones related to interferograms 
are not required at this step. You only need SLC and geometry files. If your directory is set up following ISCE 
convention, you may set `minopy.load.autoPath` to `yes` and it will automatically read the data. 
Also you need to set subset area by specifying bounding box in `minopy.subset.lalo`. 
Processing time would be a matter if large subset is selected. 
After setting up your template file, run following command to load data. It will call the `load_slc.py` script. 
```
minopyApp.py $PWD/PichinchaSenDT142.template --dostep load_data --dir $PWD/minopy
```

2. Second step would be the phase linking. 
In this step full network of wrapped phase series will be inverted using non-linear 
phase linking methods including EVD, EMI, PTA, sequential_EVD, sequential_EMI (default) and 
sequential_PTA. It will process the data in parallel by dividing the subset into patches. 
You may set the number of workers in configuration file `minopy.multiprocessing.numProcessor` depending on 
your processing system availability which will be the number of parallel jobs. 
All options begining with `minopy.inversion.*` are used in this step. Patch size is the dimension
of your patches, for example 200 by 200 as default. ministack size is the number of images used for inverting 
each mini stack. Range and Azimuth window are the size of searching window to find SHPs. 
Statistical test to find SHPs can be selected among KS (default), AD and ttest. Following command will call `phase_inversion.py` script

```
minopyApp.py $PWD/PichinchaSenDT142.template --dostep phase_linking --dir $PWD/minopy
```

3. Third step is to concatenate the patches created in previous step. 

```
minopyApp.py $PWD/PichinchaSenDT142.template --dostep concatenate_patch --dir $PWD/minopy
```

4. After phase linking you have the single reference interferograms in a stack called `phase_series.h5`. You need to unwrap the interferograms for time series analysis but unwrapping is not easy specially when you have large temporal baselines. We like to unwrap minimum number of interferograms but the most correlated ones. In MiNoPy you can select which pairs to unwrap and for that you write selected pairs from the stack to separate ifgram directories. Use options starting with `minopy.interferograms.*` in template to select your network of interferograms to unwrap. The available options are: single reference, mini_stacks and sequential. You may also write your own selected list in a text file and set the path to `minopy.interferograms.list`. For sequential pairs (more than 2 connections), you can later perform both `bridging` and `phase_closure` unwrap error corrections of MintPy but for other pair networks you may only run `bridging`. Following command will call `generate_ifgram.py` script.

```
minopyApp.py $PWD/PichinchaSenDT142.template --dostep generate_ifgram --dir $PWD/minopy
```

5. The next step would be to unwrap the selected pairs. We use [SNAPHU](https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/) for unwrapping and you can set some options starting with `minopy.unwrap.*` in template. Following command will call `unwrap_ifgram.py` script.

```
minopyApp.py $PWD/PichinchaSenDT142.template --dostep unwrap_ifgram --dir $PWD/minopy
```

6. After unwrapping, The interferograms will be loaded to a stack in HDF5 format to be ready for mintpy time series analysis and correction steps.
You can now use `minopy.load.*` options in template specified for interferograms or set `minopy.load.autoPath = yes` to read them automatically. Following command will call `load_ifgram.py` script.

```
minopyApp.py $PWD/PichinchaSenDT142.template --dostep load_ifgram --dir $PWD/minopy
```

7. At this step you will run the modify network, reference point selection and correct unwrap error using MintPy. Use the corresponding mintpy template options `mintpy.unwrapError.*`. Following command will call `smallbaselineApp.py` script from MintPy.

```
minopyApp.py $PWD/PichinchaSenDT142.template --dostep ifgram_correction --dir $PWD/minopy
```

8. Now you need to convert phase to range change (time series). The temporal coherence threshold can be set for this step using `minopy.timeseries.minTempCoh` and you can use water mask by setting `minopy.timeseries.waterMask`. Following command will call `network_inversion.py` script.

```
minopyApp.py $PWD/PichinchaSenDT142.template --dostep invert_network --dir $PWD/minopy
```

9. Finally, the time series is ready for different corrections including, tropospheric and topographic corrections. At this step you can use MintPy starting `correct_LOD` or run the following. It will call `smallbaselineApp.py` script from MintPy.


```
minopyApp.py $PWD/PichinchaSenDT142.template --dostep timeseries_correction --dir $PWD/minopy
```


#### Post processing (Optional) ####
You can correct geolocation by running the following command but you need to do it after topographic residual correction step in MintPy. Also geocoding must be done after this post processing.

```
correct_geolocation.py -g ./minopy/inputs/geometryRadar.h5 -d ./minopy/demErr.h5
```

#### Geocoding (resampling) ####
It is worth noting that MiNoPy products are in full resolution and the storage is usually a matter. 
The geocoding step of MintPy helps to reduce the size of final geocoded products by resampling the 
data to a grid of lower resolution. You can choose one of 'linear' or 'nearest' options for interpolation, 
you can change the output resolution in degrees and also there is an option to limit the subset area: 
Try playing with the following options to match your needs.

```
########## 11.1 geocode (post-processing)
# for input dataset in radar coordinates only
# commonly used resolution in meters and in degrees (on equator)
# 100,         60,          50,          30,          20,          10
# 0.000925926, 0.000555556, 0.000462963, 0.000277778, 0.000185185, 0.000092593
mintpy.geocode              = auto  #[yes / no], auto for yes
mintpy.geocode.SNWE         = auto  #[-1.2,0.5,-92,-91 / none ], auto for none, output extent in degree
mintpy.geocode.laloStep     = auto  #[-0.000555556,0.000555556 / None], auto for None, output resolution in degree
mintpy.geocode.interpMethod = auto  #[nearest], auto for nearest, interpolation method
mintpy.geocode.fillValue    = auto  #[np.nan, 0, ...], auto for np.nan, fill value for outliers.
```
