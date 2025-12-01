# Photoresist Stochastic Simulation

A 2D particle-based stochastic simulation of a photoresist film, modeling the reaction-diffusion kinetics of Acid and Base particles during the Post-Exposure Bake (PEB) process.

## Features

- **Stochastic Simulation**: Uses Brownian motion (Random Walk) to simulate particle diffusion.
- **Reaction Kinetics**: Models the neutralization reaction ($Acid + Base \rightarrow Inert$) based on a proximity interaction radius.
- **Initialization**:
  - **Acids**: Generated in a checkerboard pattern (simulating UV exposure) with 1000nm feature size.
  - **Bases**: Uniformly distributed across the 6000nm x 6000nm domain.
- **Interactive Control Panel**:
  - **Sliders**: Adjust Diffusion Coefficients ($D_{acid}$, $D_{base}$), Reaction Radius, and Initial Concentrations in real-time.
  - **Reset**: Apply new parameters and restart the simulation.
  - **View Modes**: Toggle between "Particles" (scatter plot) and "Density" (heatmap) views.

## Requirements

- Python 3.x
- `numpy`
- `matplotlib`
- `scipy`

## Usage

Run the simulation script:

```bash
python3 photoresist_sim.py
```

## Controls

- **D_acid / D_base**: Control the diffusion speed of Acid and Base particles.
- **Radius**: Set the interaction radius for the neutralization reaction.
- **Acid Conc / Base Conc**: Set the initial density of particles.
- **Reset**: Resets the simulation with the currently selected slider values.
- **View Mode (Radio Buttons)**:
  - **Particles**: Shows individual Acid (Red) and Base (Blue) particles.
  - **Density**: Shows a 2D histogram/heatmap of Acid particle density.
