# Climate Learning (WORK-IN-PROGRESS)

A work-in-progress machine learning project to predict monthly temperature/precipitation from coastline data and elevation, which can be aggregated into a Koppen climate classification, as shown below: 

<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/K%C3%B6ppen-Geiger_Climate_Classification_Map.png/1280px-K%C3%B6ppen-Geiger_Climate_Classification_Map.png" width="800px">
</p>

Libraries used are numpy for most data processing, pytorch for the neural network, matplotlib.pyplot for plots, and imageio for processing plots into gifs

The model is naïve as it does not take into account water depth, vegetation, thermohaline circulation, or the chemical makeup of the atmosphere/water.

The model has no practical scientific use, but it could be used as a fast, light-weight, high-resolution way (compared to real physics-based climate simulations) to predict data for hypothetical Earth-like land masses.

This project is not complete and will not run as I have not uploaded the necessary raw data to Github (it is 12 gigabytes). I will upload saved copies of data in the future, when my model is complete.

## About

Predicting climate is a task that doesn't neatly fit into most machine learning tasks because there is only one Earth (ie. one example). If you break up individual pixels into input and target sets, there's a real risk of overfitting.

Luckily, I've been aware of this scientific article (https://esd.copernicus.org/articles/9/1191/2018/) about the climate of a retrograde rotating Earth (ie. if Earth rotated backwards), which is a real physics-based simulation that I trust is likely to be accurate. So, I can sort of judge based on that by whether or not my model overfits.

I have some knowledge of climatology, but only to the level of a hobbyist.

## So Far...

I am focused on predicting monthly temperature for now. I am mostly finished with the data collection, going back and forth between the data processing phase and data analysis phase to reduce loss and converge quicker in my models (a horrendously long and perserverance-needing process).

Here is a generated gif of real monthly temperatures across Earth:

<p align="center">
<img src="img/real.gif" width="800px">
</p>

Here is my first attempt, using basic multivariate linear regression. There're clearly flaws in how high elevations in the northern hemisphere seems to get colder in the summer.

<p align="center">
<img src="img/lin.gif" width="800px">
</p>

Here's another attempt, using extra latitude features. I hoped it wouldn't be prominent, but you can really see three unnatural horizontal bars (the latitude inputs). At least it solved the elevation issue.

<p align="center">
<img src="img/lin2.gif" width="800px">
</p>

Here's another attempt, where I treated latitude completely differently and modelled it as a single input to constantly move with time. This took a lot of time and looked more natural, but the elevation issue came back.

<p align="center">
<img src="img/new_lat.gif" width="800px">
</p>

Here is a gradient descent attempt by a 3-layer neural network using relu(x), followed by 70 * tanh(x / 140), learning rate of 0.001, and momentum:

<p align="center">
<img src="img/neural.gif" width="800px">
</p>

And here's its prediction on a retrograde Earth:

<p align="center">
<img src="img/neural_retrograde.gif" width="800px">
</p>

I do worry that the network has overfitted, as I feel West Siberia (as in, East Siberia in the normal Earth) should be much warmer and that this retrograde model predicts it to be almost as cold as it was before.

I actually attempted neural networks first, but my inputs were not well-processed for the task, so I ended up with super-long convergence times that overfitted a lot. But this attempt with better features worked out pretty well. There's still many rough edges though:
- Patagonia should get warmer in the summer and colder in the winter
- Northern Africa is too cold during the winter
- Northern coasts of Australia is warmer than the interior during January
- Northern Antarctica is too warm
