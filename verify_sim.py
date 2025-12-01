import numpy as np
from photoresist_sim import PhotoresistSimulation

def test_simulation_logic():
    print("Testing Simulation Logic...")
    
    # Initialize with high concentration to ensure reactions happen quickly
    sim = PhotoresistSimulation(
        width=1000, height=1000,
        acid_conc=0.005, base_conc=0.005,
        D_acid=100.0, D_base=100.0,
        reaction_radius=50.0, # Large radius to force reactions
        dt=0.1
    )
    
    initial_acids = len(sim.acids)
    initial_bases = len(sim.bases)
    print(f"Initial Acids: {initial_acids}, Initial Bases: {initial_bases}")
    
    # Run for 10 steps
    for i in range(10):
        sim.update()
        
    final_acids = len(sim.acids)
    final_bases = len(sim.bases)
    print(f"Final Acids: {final_acids}, Final Bases: {final_bases}")
    
    if final_acids < initial_acids and final_bases < initial_bases:
        print("SUCCESS: Particles reacted and were removed.")
    else:
        print("FAILURE: Particle counts did not decrease.")
        exit(1)

    # Test Boundary Conditions
    print("\nTesting Boundary Conditions...")
    # Place a particle outside and see if it comes back
    sim.acids = np.array([[-10.0, 500.0], [1010.0, 500.0]])
    sim.bases = np.zeros((0, 2))
    sim._apply_boundary_conditions(sim.acids)
    
    if np.all(sim.acids[:, 0] >= 0) and np.all(sim.acids[:, 0] <= sim.width):
        print("SUCCESS: Boundary conditions applied correctly.")
    else:
        print(f"FAILURE: Boundary conditions failed. Positions: {sim.acids}")
        exit(1)

if __name__ == "__main__":
    test_simulation_logic()
