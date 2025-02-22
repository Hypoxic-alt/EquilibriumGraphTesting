import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the generic reaction ODE for: aA + bB <-> cC + dD.
def generic_reaction(concentrations, t, k1, k2, a, b, c, d):
    A, B, C, D = concentrations
    # Use the convention: X^0 = 1.
    r_forward = k1 * (A ** a) * (B ** b)
    r_reverse = k2 * (C ** c) * (D ** d)
    r = r_forward - r_reverse
    dA_dt = -a * r
    dB_dt = -b * r
    dC_dt =  c * r
    dD_dt =  d * r
    return [dA_dt, dB_dt, dC_dt, dD_dt]

def simulate_reaction(a, b, c, d, reaction_type,
                      temp_effect, vol_effect,
                      A_perturb, B_perturb, C_perturb, D_perturb,
                      A_phase1, A_phase2, A_phase3, A_phase4,
                      B_phase1, B_phase2, B_phase3, B_phase4,
                      C_phase1, C_phase2, C_phase3, C_phase4,
                      D_phase1, D_phase2, D_phase3, D_phase4,
                      phase_changes, show_title):
    # Base rate constants.
    k1_base = 0.02
    k2_base = 0.01

    # Initialize current rate constants and state.
    k1_current = k1_base
    k2_current = k2_base
    init_state = [1.0, 1.0, 0.0, 0.0]

    # The full phase list: phase 1 is always "Base", and then the three user-selected boundaries.
    phases = ["Base"] + phase_changes  # total of 4 phases
    sols = []      # to store solution for each phase
    t_phases = []  # to store time arrays

    # Simulate each phase (each phase lasts 200 time units).
    for i, phase in enumerate(phases):
        t_phase = np.linspace(i * 200, (i + 1) * 200, 1000)
        sol = odeint(generic_reaction, init_state, t_phase, args=(k1_current, k2_current, a, b, c, d))
        sols.append(sol)
        t_phases.append(t_phase)
        
        # If not the last phase, apply the chosen change to update the state (and possibly rate constants)
        if i < len(phases) - 1:
            next_change = phases[i + 1]
            # Get final state from current phase.
            init_state = sol[-1].copy()
            if next_change == "Temperature":
                # Apply temperature change based on reaction type.
                if reaction_type == "Exothermic":
                    # For exothermic, increasing temperature shifts equilibrium toward reactants.
                    k2_current = k2_base * (1 + temp_effect)
                else:
                    # For endothermic, increasing temperature shifts equilibrium toward products.
                    k1_current = k1_base * (1 + temp_effect)
            elif next_change == "Volume/Pressure":
                # For a volume change, the concentrations change (assume dilution or compression).
                init_state = init_state / (1 + vol_effect)
            elif next_change == "Addition":
                # For an agent addition, adjust each species independently.
                init_state[0] *= (1 + A_perturb)
                init_state[1] *= (1 + B_perturb)
                init_state[2] *= (1 + C_perturb)
                init_state[3] *= (1 + D_perturb)

    # Create the plot.
    fig = plt.figure(figsize=(10, 6))

    # Plot each species for each phase if the corresponding show flag is True.
    # Species A:
    phases_labels = ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]
    for i, sol in enumerate(sols):
        if a != 0:
            if (i == 0 and A_phase1) or (i == 1 and A_phase2) or (i == 2 and A_phase3) or (i == 3 and A_phase4):
                plt.plot(t_phases[i], sol[:, 0], label='A ' + phases_labels[i], color='blue', linestyle='solid', linewidth=2)
    # Species B:
    for i, sol in enumerate(sols):
        if b != 0:
            if (i == 0 and B_phase1) or (i == 1 and B_phase2) or (i == 2 and B_phase3) or (i == 3 and B_phase4):
                plt.plot(t_phases[i], sol[:, 1], label='B ' + phases_labels[i], color='red', linestyle='solid', linewidth=2)
    # Species C:
    for i, sol in enumerate(sols):
        if c != 0:
            if (i == 0 and C_phase1) or (i == 1 and C_phase2) or (i == 2 and C_phase3) or (i == 3 and C_phase4):
                plt.plot(t_phases[i], sol[:, 2], label='C ' + phases_labels[i], color='green', linestyle='solid', linewidth=2)
    # Species D:
    for i, sol in enumerate(sols):
        if d != 0:
            if (i == 0 and D_phase1) or (i == 1 and D_phase2) or (i == 2 and D_phase3) or (i == 3 and D_phase4):
                plt.plot(t_phases[i], sol[:, 3], label='D ' + phases_labels[i], color='purple', linestyle='solid', linewidth=2)

    # Draw vertical dotted lines at the phase boundaries (at t = 200, 400, 600).
    for boundary in [200, 400, 600]:
        plt.axvline(x=boundary, color='grey', linestyle=':', linewidth=1)

    # Draw vertical connecting lines between phases for each species.
    # This connects the final value of one phase with the initial value of the next phase.
    # We do this only if the species is being shown in both phases.
    def draw_connection(sol_prev, sol_next, t_prev, t_next, color):
        plt.plot([t_prev[-1], t_prev[-1]], [sol_prev[-1], sol_next[0]], color=color, linestyle='solid', linewidth=2)

    # For species A.
    if a != 0:
        if A_phase1 and A_phase2:
            draw_connection(sols[0][:, 0], sols[1][:, 0], t_phases[0], t_phases[1], 'blue')
        if A_phase2 and A_phase3:
            draw_connection(sols[1][:, 0], sols[2][:, 0], t_phases[1], t_phases[2], 'blue')
        if A_phase3 and A_phase4:
            draw_connection(sols[2][:, 0], sols[3][:, 0], t_phases[2], t_phases[3], 'blue')
    # For species B.
    if b != 0:
        if B_phase1 and B_phase2:
            draw_connection(sols[0][:, 1], sols[1][:, 1], t_phases[0], t_phases[1], 'red')
        if B_phase2 and B_phase3:
            draw_connection(sols[1][:, 1], sols[2][:, 1], t_phases[1], t_phases[2], 'red')
        if B_phase3 and B_phase4:
            draw_connection(sols[2][:, 1], sols[3][:, 1], t_phases[2], t_phases[3], 'red')
    # For species C.
    if c != 0:
        if C_phase1 and C_phase2:
            draw_connection(sols[0][:, 2], sols[1][:, 2], t_phases[0], t_phases[1], 'green')
        if C_phase2 and C_phase3:
            draw_connection(sols[1][:, 2], sols[2][:, 2], t_phases[1], t_phases[2], 'green')
        if C_phase3 and C_phase4:
            draw_connection(sols[2][:, 2], sols[3][:, 2], t_phases[2], t_phases[3], 'green')
    # For species D.
    if d != 0:
        if D_phase1 and D_phase2:
            draw_connection(sols[0][:, 3], sols[1][:, 3], t_phases[0], t_phases[1], 'purple')
        if D_phase2 and D_phase3:
            draw_connection(sols[1][:, 3], sols[2][:, 3], t_phases[1], t_phases[2], 'purple')
        if D_phase3 and D_phase4:
            draw_connection(sols[2][:, 3], sols[3][:, 3], t_phases[2], t_phases[3], 'purple')

    plt.xlabel("Time")
    plt.ylabel("Concentration")
    if show_title:
        # In the title, we also show the chosen perturbations in order.
        title_str = ("Generic Reaction: {}A + {}B ↔ {}C + {}D  |  Reaction: {}  |  Boundaries: {}, {}, {}"
                     .format(a, b, c, d, reaction_type, phase_changes[0], phase_changes[1], phase_changes[2]))
        plt.title(title_str)
    plt.grid(False)
    plt.tight_layout()
    
    return fig

# ------------------- Streamlit UI -------------------

st.title("Generic Reaction Equilibrium Simulation")
st.markdown("Adjust the parameters in the sidebar to simulate the reaction.")

# Reaction parameters
st.sidebar.header("Reaction Parameters")
a = st.sidebar.slider("Coefficient a", min_value=0, max_value=5, value=1, step=1)
b = st.sidebar.slider("Coefficient b", min_value=0, max_value=5, value=1, step=1)
c = st.sidebar.slider("Coefficient c", min_value=0, max_value=5, value=1, step=1)
d = st.sidebar.slider("Coefficient d", min_value=0, max_value=5, value=1, step=1)

reaction_type = st.sidebar.selectbox("Reaction Type", options=["Exothermic", "Endothermic"])
temp_effect = st.sidebar.slider("Temp Effect", min_value=-1.0, max_value=1.0, value=0.0, step=0.05)
vol_effect = st.sidebar.slider("Vol/Pressure Effect", min_value=-0.5, max_value=0.5, value=0.0, step=0.05)

# Checkbox to show/hide the plot title.
show_title = st.sidebar.checkbox("Show Plot Title", value=True)

# Perturbation parameters (for addition events).
st.sidebar.header("Species Perturbations (Addition)")
A_perturb = st.sidebar.slider("A Perturb", min_value=-0.5, max_value=0.5, value=0.0, step=0.05)
B_perturb = st.sidebar.slider("B Perturb", min_value=-0.5, max_value=0.5, value=0.0, step=0.05)
C_perturb = st.sidebar.slider("C Perturb", min_value=-0.5, max_value=0.5, value=0.0, step=0.05)
D_perturb = st.sidebar.slider("D Perturb", min_value=-0.5, max_value=0.5, value=0.0, step=0.05)

# Boundary (phase transition) selections.
st.sidebar.header("Phase Boundary Changes")
phase_change1 = st.sidebar.selectbox("Boundary 1 change", options=["Temperature", "Volume/Pressure", "Addition"])
phase_change2 = st.sidebar.selectbox("Boundary 2 change", options=["Temperature", "Volume/Pressure", "Addition"])
phase_change3 = st.sidebar.selectbox("Boundary 3 change", options=["Temperature", "Volume/Pressure", "Addition"])
phase_changes = [phase_change1, phase_change2, phase_change3]

# Show/hide phase sections for species (choose which phases to plot per species).
st.sidebar.header("Show/Hide Phase Sections")

st.sidebar.subheader("Species A")
A_phase1 = st.sidebar.checkbox("A Phase 1", value=True)
A_phase2 = st.sidebar.checkbox("A Phase 2", value=True)
A_phase3 = st.sidebar.checkbox("A Phase 3", value=True)
A_phase4 = st.sidebar.checkbox("A Phase 4", value=True)

st.sidebar.subheader("Species B")
B_phase1 = st.sidebar.checkbox("B Phase 1", value=True)
B_phase2 = st.sidebar.checkbox("B Phase 2", value=True)
B_phase3 = st.sidebar.checkbox("B Phase 3", value=True)
B_phase4 = st.sidebar.checkbox("B Phase 4", value=True)

st.sidebar.subheader("Species C")
C_phase1 = st.sidebar.checkbox("C Phase 1", value=True)
C_phase2 = st.sidebar.checkbox("C Phase 2", value=True)
C_phase3 = st.sidebar.checkbox("C Phase 3", value=True)
C_phase4 = st.sidebar.checkbox("C Phase 4", value=True)

st.sidebar.subheader("Species D")
D_phase1 = st.sidebar.checkbox("D Phase 1", value=True)
D_phase2 = st.sidebar.checkbox("D Phase 2", value=True)
D_phase3 = st.sidebar.checkbox("D Phase 3", value=True)
D_phase4 = st.sidebar.checkbox("D Phase 4", value=True)

# Run simulation and display plot.
fig = simulate_reaction(a, b, c, d, reaction_type,
                        temp_effect, vol_effect,
                        A_perturb, B_perturb, C_perturb, D_perturb,
                        A_phase1, A_phase2, A_phase3, A_phase4,
                        B_phase1, B_phase2, B_phase3, B_phase4,
                        C_phase1, C_phase2, C_phase3, C_phase4,
                        D_phase1, D_phase2, D_phase3, D_phase4,
                        phase_changes, show_title)

st.pyplot(fig)
