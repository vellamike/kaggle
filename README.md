Kaggle Belkin competition
=========================

WARNING:Don't add raw data to the git repo, it's too big and github
won't like it. (Also, it will make everything really slow).

BD Thoughts:
---------------------

* On the leaderboard: All but 2 teams have made relatively little progress over the baseline. For a long time slow but steady was first with ~0.05, and second place was ~0.07. Titan dropped suddenly from ~0.07 to 0.05. What this means for us: Since we can assume all the top teams are trying hard, there is likely some *significant and unexpected* feature to find. We haven't found it yet.

* The HF data represents frequencies 0-1MHz. Each bin contains power across a 244Hz range. The action seems to be happening in the bins near the end 3500-4000, corresponding to the ~100kHz range, fitting with the competition description.

* Big devices should show up in the power consumption. Small devices with high frequency switching, such as CFLs, should create HF EMI. I can't see any means of predicting any remaining devices.

* Test data probably looks very different from training, since test data is based on simulation of a real house. Training data just shows a couple of minutes for each appliance.

*  [Splitphase power](http://en.wikipedia.org/wiki/Split-phase_electric_power) is used, so appliances are on a multiwire branch circuit. Some will get phase 1, some phase 2, some (big appliances) both. Unless the devices move, this should be reflected in the power usage on each phase. 

* Example iPython session. Shows what's going on for high frequency, and phase 1 of the power line between 13:00 and 13:30 (NB houses are in California: GMT -8):


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
* smart_plot: You'll need to make the plotting window large enough for the HF EMI plot to be as wide as the LF plots. Then the time axes will line up, so you can see simultaneous impact of a device on power and HF EMI.
You might want to play with smart_plot's min_bin and max_bin parameters to look at other parts of the frequency range.


* Appliances always off benchmark: 0.08
* Total hamming loss: 0.32
* Timepoints: 219579
* Approximate devices per house: 38
* Approximate ON time (public data): Total hamming loss * timepoints * approx. devices per house = 2670080
* Percentage equivalent ON time: Approximate_on_time / timepoints = 0.32 * 38 / 2 = 6.08

MV Thoughts:
------------
As a 1st order approx we should:

* Assume a divice is off if error margin seems big (6.08% on-time
  implies this is reasonable)
* Assume everything is off at night (unless an obvious reason not to?)
* Error function such that divices which are not on very much (e.g
  Washing machines, toasters) should just be set to permanently off.
  This is particularly true because looking at the forums it seems as
  though there are issues with tagging of washing machines. BD: As a result of this forum whinging, 
  admins [decided to correct](http://www.kaggle.com/c/belkin-energy-disaggregation-competition/forums/t/5933/when-will-the-back-end-changes-happen) parts of the data set, with changes happening by Oct 15. 
  Let's have a bool flag for the washing machine, so we can test with and w\o it. 
  If they've tagged it, a 2 hour runtime is worth predicting.
  
Naming notes:
-------------

* 1 = Phase 1
* 2 = Phase 2
* LF = Low freq
* HF = High freq
* V = Voltage
* I = current
