Usage
-----
Manual noise learn workflow:
1) First select a portion of noise in your track of at least 1 second and loop it.
2) Turn on learn noise profile for a bit (at least 4 or 5 loops).
3) Once that's done turn it off and tweak parameters as you like.

Adaptive noise learn workflow:
1) Turn on Adaptive mode and keep it on (preferably)
2) Tweak parameters

What's the difference between Audacity's noise reduction tool?
-----
The main difference is that this plugin is coded in order to apply the effect in real time. That sounds fancy but really is a limitation. When you apply any processing offline (not in real time) you have access to all the data at once and that is a big advantage in order to use much more complex schemes for a cleaner reduction. In real time you only have access small blocks of sound at a time. So what is gained using real time processing? You have instant feedback of what you are doing (that is not the case in Audacity Noise Reduction).

Ok, so in terms of interaction noise-repellent is much better but what about the internals and better yet results? Audacity uses a combination of a "second-greatest" filter to apply over-subtraction and time and frequency smoothing in order to get cleaner results. Noise-repellent uses a psycho-acoustic model based on human hearing to adaptively apply over-subtraction and time smoothing with a release envelope to avoid artifacts. Plus is has an onset detector for transient protection in percussive sounds. Alright, but how does it sounds? You'll have to try it for yourself but I haven't had the chance to encounter a case where it sounded worse yet.

Parameters explained
-----
* Amount of reduction: Determines how much the noise floor will be reduced.
* Thresholds offset: Scales the noise profile learned. Greater values will reduce more noise at the expense of removing low level detail of the signal. Lower values will preserve the signal better but noise might appear.
* Release: Timing of the reduction applied. Larger values will reduce artifacts but might blur nearby transients (only for spectral gating).
* Masking: This enables a psycho-acoustic model that gets the masking thresholds of an estimated clean signal and adaptively scales the noise spectrum in order to avoid distortion or musical noise. Higher values will reduce more musical noise but can distort lower details of the signal. Lower values will preserve the signal more but musical noise might be heard. Value of 1 turn off this feature.
* Transient Protection: This parameter dictates the scaling applied to the onset detection thresholding function. This means lower values will only preserve louder transients and higher values will preserve more but can reintroduce musical noise if is configured too high. The onset detector is used to detect transients and use a non smoothed suppression rule to preserve them better. Value of 1 turn off this feature.
* Whitening: Modifies the residual noise to be more like white noise. This takes into account that our ears do well discriminating sounds in white noise versus colored noise. Higher values will brighten the residual noise and will mask high frequency artifacts.
* Learn noise profile: To manually learn the noise profile.
* Adaptive noise learn: To change the noise profile dynamically in time. This enables the automatic estimation of noise thresholds.
* Reset noise profile: Removes the noise profile previously learned.
* Residual listen: To only hear the residual of the reduction.

Advice for better reduction
-----
General noise reduction advice
* Try to reduce and not remove entirely the noise. It will sound better.
* Gentler settings with multiple instance of the plug-in will probably sound better than too much reduction with one instance. Of course for every instance the noise should be re-learned again.
* If the noise varies to much from one section to other apply different reduction for each part.
* Always remember to listen to the residual signal to make sure that you are not distorting the signal too much.
* Start with the reduction at 0 dB and then decrease it until you hear artifacts then tune the parameters to get rid of them without distorting the signal.
* It might help using an spectrogram analysis to help you notice what you are doing. This can be easily done by using a-Inline Spectrogram in Ardour or the spectrogram view in audacity or sonic visualizer

For adaptive Reduction:
* Adaptive mode should be used only with voice tracks because the algorithm for noise estimation is tuned for that use.
* It's recommended to play with thresholds offset because those are estimated continuously and the algorithm used for that tends sometimes to overestimate them.
* Release won't work at all in this version. To reduce more or less use threshold offset
* If the track you are processing does not have a long section of noise before the wanted signal starts cut some noise from an in-between section and extend the beginning a bit. This is to take into account the time that takes the algorithm to learn the noise. Alternatively you can learn the noise profile by using one section of the track and then turning off the adaptive mode so a fixed noise profile is used. This will not adapt in time but will give you something to work with.

For manual reduction:
* You can use adaptive mode to estimate noise thresholds when there is no section in the track that contains only noise. It should be used the same way you would use the manual learn.
* If noise floor changes a bit over time it might be useful to use higher thresholds offset.
* Make sure that the section you select to learn the noise profile is noise only (no breaths or sustained notes or anything but noise)
* The longer the section you select to learn the noise the better the reduction will sound.
* The best way to reduce artifacts is to use a combination of masking, release and transient protection controls. Masking will reduce musical noise and at the same time it will preserve the wanted signal but too much will start to distort it. Release can reduce musical noise significantly but too much can blur transients away. To avoid this you can increase transient preservation until you find those transients less distorted. If you are not perceiving what you are doing, listen to the residual signal and increase to a higher value. Each of this controls work independently be sure to try the smallest amount of each one just to avoid artifacts. This will lead to a better reduction.