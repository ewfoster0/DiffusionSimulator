import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import cKDTree

class PhotoresistSimulation:
    def __init__(self, width=2000.0, height=2000.0, 
                 acid_conc=0.002, base_conc=0.002, 
                 D_acid=50.0, D_base=10.0, 
                 reaction_radius=5.0, dt=0.1):
        """
        Initialize the simulation parameters and particles.
        
        Args:
            width (float): Width of the simulation domain (nm).
            height (float): Height of the simulation domain (nm).
            acid_conc (float): Approximate concentration/density of acid particles (particles/nm^2).
            base_conc (float): Approximate concentration/density of base particles.
            D_acid (float): Diffusion coefficient for Acid (nm^2/s).
            D_base (float): Diffusion coefficient for Base (nm^2/s).
            reaction_radius (float): Distance within which Acid and Base react (nm).
            dt (float): Time step (s).
        """
        self.width = width
        self.height = height
        self.D_acid = D_acid
        self.D_base = D_base
        self.reaction_radius = reaction_radius
        self.dt = dt
        self.acid_conc = acid_conc
        self.base_conc = base_conc
        
        self.reset()

    def reset(self):
        """
        Reset the simulation to initial state.
        """
        self.acids = self._initialize_acids(self.acid_conc)
        self.bases = self._initialize_bases(self.base_conc)
        self.time = 0.0

    def _initialize_acids(self, concentration):
        """
        Initialize Acid particles in a checkerboard pattern.
        """
        # Create a grid of potential particle positions or regions
        # For a checkerboard, we can divide the domain into squares.
        # Let's say 1000nm x 1000nm squares.
        square_size = 1000.0
        num_x = int(self.width / square_size)
        num_y = int(self.height / square_size)
        
        particles = []
        
        # Calculate number of particles per square based on concentration
        particles_per_square = int(concentration * (square_size ** 2))
        
        for i in range(num_x):
            for j in range(num_y):
                # Checkerboard logic: populate if i+j is even (or odd)
                if (i + j) % 2 == 0:
                    # Generate random positions within this square
                    x_min = i * square_size
                    x_max = (i + 1) * square_size
                    y_min = j * square_size
                    y_max = (j + 1) * square_size
                    
                    x_pos = np.random.uniform(x_min, x_max, particles_per_square)
                    y_pos = np.random.uniform(y_min, y_max, particles_per_square)
                    
                    particles.append(np.column_stack((x_pos, y_pos)))
                    
        if particles:
            return np.vstack(particles)
        return np.zeros((0, 2))

    def _initialize_bases(self, concentration):
        """
        Initialize Base particles uniformly across the domain.
        """
        total_area = self.width * self.height
        num_particles = int(concentration * total_area)
        
        x_pos = np.random.uniform(0, self.width, num_particles)
        y_pos = np.random.uniform(0, self.height, num_particles)
        
        return np.column_stack((x_pos, y_pos))

    def update(self):
        """
        Advance the simulation by one time step.
        """
        # 1. Move particles (Brownian Motion)
        # Displacement = sqrt(2 * D * dt) * N(0, 1)
        
        if len(self.acids) > 0:
            disp_acid = np.sqrt(2 * self.D_acid * self.dt) * np.random.normal(0, 1, self.acids.shape)
            self.acids += disp_acid
            self._apply_boundary_conditions(self.acids)
            
        if len(self.bases) > 0:
            disp_base = np.sqrt(2 * self.D_base * self.dt) * np.random.normal(0, 1, self.bases.shape)
            self.bases += disp_base
            self._apply_boundary_conditions(self.bases)
            
        # 2. Interaction (Reaction)
        self._react()
        
        self.time += self.dt

    def _apply_boundary_conditions(self, particles):
        """
        Apply reflective boundary conditions.
        """
        # Reflect off 0
        # If x < 0, x = -x
        np.abs(particles, out=particles)
        
        # Reflect off Width/Height
        # If x > W, x = 2W - x
        # We handle x and y separately for the upper bound
        
        # X dimension
        mask_x = particles[:, 0] > self.width
        particles[mask_x, 0] = 2 * self.width - particles[mask_x, 0]
        
        # Y dimension
        mask_y = particles[:, 1] > self.height
        particles[mask_y, 1] = 2 * self.height - particles[mask_y, 1]
        
        # Handle corner cases where reflection pushes it back past 0 (rare but possible with large steps)
        # A simple second pass of abs fixes the < 0 case again if it happened.
        np.abs(particles, out=particles)

    def _react(self):
        """
        Remove Acid and Base pairs that are within the reaction radius.
        """
        if len(self.acids) == 0 or len(self.bases) == 0:
            return

        # Use KDTree for efficient neighbor search
        tree_acid = cKDTree(self.acids)
        tree_base = cKDTree(self.bases)
        
        # query_ball_tree returns list of lists of indices
        # We want to find pairs. 
        # Strategy: For each acid, find bases within radius.
        # This gives us potential pairs. We need to handle 1-to-1 mapping if multiple match.
        # A simple greedy approach: shuffle and pick.
        
        # Using query_ball_tree to find all bases near each acid
        # results is a list where i-th element is list of base indices near acid i
        results = tree_acid.query_ball_tree(tree_base, r=self.reaction_radius)
        
        acid_indices_to_remove = set()
        base_indices_to_remove = set()
        
        # Process results. To avoid bias, we could shuffle, but iterating in order is usually fine for stochastic sims
        # unless density is extremely high.
        # We need to ensure one base only reacts with one acid in a single step.
        
        for acid_idx, nearby_bases in enumerate(results):
            if acid_idx in acid_indices_to_remove:
                continue
                
            for base_idx in nearby_bases:
                if base_idx not in base_indices_to_remove:
                    # Reaction occurs
                    acid_indices_to_remove.add(acid_idx)
                    base_indices_to_remove.add(base_idx)
                    break # This acid is now used
        
        # Remove particles
        if acid_indices_to_remove:
            keep_acid = np.array([i for i in range(len(self.acids)) if i not in acid_indices_to_remove])
            if len(keep_acid) < len(self.acids):
                 self.acids = self.acids[keep_acid] if len(keep_acid) > 0 else np.zeros((0, 2))
        
        if base_indices_to_remove:
            keep_base = np.array([i for i in range(len(self.bases)) if i not in base_indices_to_remove])
            if len(keep_base) < len(self.bases):
                self.bases = self.bases[keep_base] if len(keep_base) > 0 else np.zeros((0, 2))

from matplotlib.widgets import Slider, Button, RadioButtons

def run_animation():
    # Simulation Parameters
    D_acid = 200.0
    D_base = 50.0
    reaction_radius = 15.0
    
    sim = PhotoresistSimulation(
        width=6000, height=6000,
        acid_conc=0.0005, base_conc=0.0005, 
        D_acid=D_acid, D_base=D_base,
        reaction_radius=reaction_radius,
        dt=0.1
    )
    
    # Create figure and adjust layout to make room for controls
    fig, ax = plt.subplots(figsize=(8, 10))
    plt.subplots_adjust(bottom=0.45, right=0.8) # Make room on right for radio buttons
    
    ax.set_xlim(0, sim.width)
    ax.set_ylim(0, sim.height)
    ax.set_aspect('equal')
    ax.set_title(f"Photoresist PEB Simulation\nTime: 0.0s | Acids: {len(sim.acids)} | Bases: {len(sim.bases)}")
    
    # Scatter plots (Particles View)
    scat_acid = ax.scatter([], [], c='red', s=2, label='Acid', alpha=0.6)
    scat_base = ax.scatter([], [], c='blue', s=2, label='Base', alpha=0.6)
    
    # Density plot (Density View) - Initially hidden
    # Use a fixed bin size for density
    bins = 100
    density_img = ax.imshow(np.zeros((bins, bins)), origin='lower', extent=[0, sim.width, 0, sim.height], 
                            cmap='coolwarm', alpha=0.8, animated=True, vmin=0, vmax=10) # vmin/vmax might need dynamic adjustment
    density_img.set_visible(False)
    
    # Legend - only for scatter
    legend = ax.legend(loc='upper right')
    
    # State for view mode
    view_mode = {'mode': 'Particles'}

    # Controls
    axcolor = 'lightgoldenrodyellow'
    
    # Sliders
    ax_D_acid = plt.axes([0.25, 0.3, 0.65, 0.03], facecolor=axcolor)
    ax_D_base = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
    ax_radius = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
    ax_acid_conc = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_base_conc = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    
    s_D_acid = Slider(ax_D_acid, 'D_acid', 0.0, 500.0, valinit=D_acid)
    s_D_base = Slider(ax_D_base, 'D_base', 0.0, 500.0, valinit=D_base)
    s_radius = Slider(ax_radius, 'Radius', 1.0, 50.0, valinit=reaction_radius)
    s_acid_conc = Slider(ax_acid_conc, 'Acid Conc', 0.0001, 0.002, valinit=0.0005, valfmt='%1.4f')
    s_base_conc = Slider(ax_base_conc, 'Base Conc', 0.0001, 0.002, valinit=0.0005, valfmt='%1.4f')
    
    def update_params(val):
        sim.D_acid = s_D_acid.val
        sim.D_base = s_D_base.val
        sim.reaction_radius = s_radius.val
        sim.acid_conc = s_acid_conc.val
        sim.base_conc = s_base_conc.val
        
    s_D_acid.on_changed(update_params)
    s_D_base.on_changed(update_params)
    s_radius.on_changed(update_params)
    s_acid_conc.on_changed(update_params)
    s_base_conc.on_changed(update_params)
    
    # Reset Button
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    
    def reset(event):
        sim.reset()
        # Reset view logic handled in animate, but we can force an update
        ax.set_title(f"Photoresist PEB Simulation\nTime: {sim.time:.1f}s | Acids: {len(sim.acids)} | Bases: {len(sim.bases)}")
        
    button.on_clicked(reset)
    
    # Radio Buttons for View Mode
    rax = plt.axes([0.82, 0.5, 0.15, 0.15], facecolor=axcolor)
    radio = RadioButtons(rax, ('Particles', 'Density'))
    
    def change_view(label):
        view_mode['mode'] = label
        if label == 'Particles':
            scat_acid.set_visible(True)
            scat_base.set_visible(True)
            density_img.set_visible(False)
            legend.set_visible(True)
        else:
            scat_acid.set_visible(False)
            scat_base.set_visible(False)
            density_img.set_visible(True)
            legend.set_visible(False)
        fig.canvas.draw_idle()

    radio.on_clicked(change_view)
    
    def init():
        scat_acid.set_offsets(np.zeros((0, 2)))
        scat_base.set_offsets(np.zeros((0, 2)))
        density_img.set_data(np.zeros((bins, bins)))
        return scat_acid, scat_base, density_img
    
    def animate(frame):
        # Multiple simulation steps per frame
        steps_per_frame = 5
        for _ in range(steps_per_frame):
            sim.update()
            
        ax.set_title(f"Photoresist PEB Simulation\nTime: {sim.time:.1f}s | Acids: {len(sim.acids)} | Bases: {len(sim.bases)}")
        
        if view_mode['mode'] == 'Particles':
            scat_acid.set_offsets(sim.acids)
            scat_base.set_offsets(sim.bases)
            return scat_acid, scat_base
        else:
            # Calculate density
            if len(sim.acids) > 0:
                # histogram2d returns H, xedges, yedges
                H, _, _ = np.histogram2d(sim.acids[:, 0], sim.acids[:, 1], bins=bins, range=[[0, sim.width], [0, sim.height]])
                # Transpose H because imshow expects (rows, cols) where rows are y
                density_img.set_data(H.T)
                # Auto-scale colorbar or keep fixed? Fixed is better for comparison, but dynamic is safer.
                # Let's stick to a reasonable fixed range or use the initial concentration to guess.
                # Max particles per bin ~ (total_acids / total_bins) * clustering_factor
                # For now, let's just let it be or update clim if needed. 
                # imshow auto-scales if we don't set vmin/vmax, but we set them.
                # Let's update clim dynamically for better visibility
                if H.max() > 0:
                    density_img.set_clim(vmin=0, vmax=H.max())
            else:
                density_img.set_data(np.zeros((bins, bins)))
                
            return density_img,
    
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=False)
    plt.show()

if __name__ == "__main__":
    run_animation()
