Kaggle Belkin competition
=========================

WARNING:Don't add raw data to the git repo, it's too big and github
won't like it. (Also, it will make everything really slow).

BD had some thoughts:
---------------------

* Not quite sure which frequencies the HF data represents. The action seems to be happening in the bins near the end 3500-4000. Perhaps this corresponds to frequencies c. 50kHz-550kHz, which would fit with the plots on the competition description. I don't think this is particularly important.

* Big devices should show up in the power consumption. Small devices with high frequency switching, such as CFLs, should create HF EMI. I can't see any means of predicting any remaining devices.

* Test data probably looks very different from training.

* Example iPython session. Shows what's going on for high frequency, and phase 1 of the power line between 13:00 and 13:30:


```
In [86]: import mat_loading as ml

In [87]: d = ml.load_data(file='H4/Tagged_Training_07_26_1343286001')
start: 
2012-07-26 08:00:01
end : 
2012-07-27 07:59:58

In [88]: ml.smart_plot(d, d.start + 60*60*13, 60 * 60 * 0.5, HF = True, L1_real = True, L1_imaginary = True, L1_factor = True)
Out[88]: <matplotlib.figure.Figure at 0xe119dec>

In [89]: ml.device_sample_all(d)
```

(This throws up about 20 plots...)

* Appliances always off benchmark: 0.08
* Total hamming loss: 0.32
* Timepoints: 219579
* Approximate devices per house: 38
* Approximate ON time (public data): Total hamming loss * timepoints * approx. devices per house = 2670080
* Percentage equivalent ON time: Approximate_on_time / timepoints = 0.32 * 38 / 2 = 6.08

MV's thoughts:
--------------
As a 1st order approx we should:

* Assume a divice is off if error margin seems big (6.08% on-time
  implies this is reasonable)
* Assume everything is off at night (unless an obvious reason not to?)
* Error function such that divices which are not on very much (e.g
  Washing machines, toasters) should just be set to permanently off.
  This is particularly true because looking at the forums it seems as
  though there are issues with tagging of washing machines.
  
