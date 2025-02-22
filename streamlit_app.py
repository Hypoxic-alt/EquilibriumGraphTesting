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
                      show_title):
    # Base rate constants.
    k1_base = 0.02
    k2_base = 0.01

    # Define time arrays for each phase (each 200 time units long).
    t_phase1 = np.linspace(0, 200, 1000)
    t_phase2 = np.linspace(200, 400, 1000)
    t_phase3 = np.linspace(400, 600, 1000)
    t_phase4 = np.linspace(600, 800, 1000)

    # Phase 1: Base conditions.
    init_phase1 = [1.0, 1.0, 0.0, 0.0]  # [A], [B], [C], [D]
    sol1 = odeint(generic_reaction, init_phase1, t_phase1, args=(k1_base, k2_base, a, b, c, d))

    # Phase 2: Temperature perturbation.
    if reaction_type == "Exothermic":
        # Exothermic: increasing T shifts equilibrium toward reactants.
        # Keep k1 constant; adjust k2.
        k1_phase2 = k1_base
        k2_phase2 = k2_base * (1 + temp_effect)
    else:  # Endothermic
        # Endothermic: increasing T shifts equilibrium toward products.
        # Keep k2 constant; adjust k1.
        k1_phase2 = k1_base * (1 + temp_effect)
        k2_phase2 = k2_base
    init_phase2 = sol1[-1]  # Use final state from Phase 1.
    sol2 = odeint(generic_reaction, init_phase2, t_phase2, args=(k1_phase2, k2_phase2, a, b, c, d))

    # Phase 3: Volume perturbation.
    init_phase3 = sol2[-1].copy() / (1 + vol_effect)
    sol3 = odeint(generic_reaction, init_phase3, t_phase3, args=(k1_phase2, k2_phase2, a, b, c, d))

    # Phase 4: Independent species perturbation.
    init_phase4 = sol3[-1].copy()
    init_phase4[0] *= (1 + A_perturb)
    init_phase4[1] *= (1 + B_perturb)
    init_phase4[2] *= (1 + C_perturb)
    init_phase4[3] *= (1 + D_perturb)
    sol4 = odeint(generic_reaction, init_phase4, t_phase4, args=(k1_phase2, k2_phase2, a, b, c, d))

    # Create the plot
    fig = plt.figure(figsize=(10, 6))

    # Plot species A.
    if a != 0:
        if A_phase1:
            plt.plot(t_phase1, sol1[:, 0], label='A Phase 1', color='blue', linestyle='solid', linewidth=2)
        if A_phase2:
            plt.plot(t_phase2, sol2[:, 0], label='A Phase 2', color='blue', linestyle='solid', linewidth=2)
        if A_phase3:
            plt.plot(t_phase3, sol3[:, 0], label='A Phase 3', color='blue', linestyle='solid', linewidth=2)
        if A_phase4:
            plt.plot(t_phase4, sol4[:, 0], label='A Phase 4', color='blue', linestyle='solid', linewidth=2)
    # Plot species B.
    if b != 0:
        if B_phase1:
            plt.plot(t_phase1, sol1[:, 1], label='B Phase 1', color='red', linestyle='solid', linewidth=2)
        if B_phase2:
            plt.plot(t_phase2, sol2[:, 1], label='B Phase 2', color='red', linestyle='solid', linewidth=2)
        if B_phase3:
            plt.plot(t_phase3, sol3[:, 1], label='B Phase 3', color='red', linestyle='solid', linewidth=2)
        if B_phase4:
            plt.plot(t_phase4, sol4[:, 1], label='B Phase 4', color='red', linestyle='solid', linewidth=2)
    # Plot species C.
    if c != 0:
        if C_phase1:
            plt.plot(t_phase1, sol1[:, 2], label='C Phase 1', color='green', linestyle='solid', linewidth=2)
        if C_phase2:
            plt.plot(t_phase2, sol2[:, 2], label='C Phase 2', color='green', linestyle='solid', linewidth=2)
        if C_phase3:
            plt.plot(t_phase3, sol3[:, 2], label='C Phase 3', color='green', linestyle='solid', linewidth=2)
        if C_phase4:
            plt.plot(t_phase4, sol4[:, 2], label='C Phase 4', color='green', linestyle='solid', linewidth=2)
    # Plot species D.
    if d != 0:
        if D_phase1:
            plt.plot(t_phase1, sol1[:, 3], label='D Phase 1', color='purple', linestyle='solid', linewidth=2)
        if D_phase2:
            plt.plot(t_phase2, sol2[:, 3], label='D Phase 2', color='purple', linestyle='solid', linewidth=2)
        if D_phase3:
            plt.plot(t_phase3, sol3[:, 3], label='D Phase 3', color='purple', linestyle='solid', linewidth=2)
        if D_phase4:
            plt.plot(t_phase4, sol4[:, 3], label='D Phase 4', color='purple', linestyle='solid', linewidth=2)

    # Draw vertical dotted lines at phase boundaries.
    for boundary in [200, 400, 600]:
        plt.axvline(x=boundary, color='grey', linestyle=':', linewidth=1)

    # Draw vertical connecting lines between phases for each species.
    # For species A (blue).
    if a != 0:
        if A_phase1 and A_phase2:
            plt.plot([t_phase1[-1], t_phase1[-1]], [sol1[-1, 0], sol2[0, 0]], color='blue', linestyle='solid', linewidth=2)
        if A_phase2 and A_phase3:
            plt.plot([t_phase2[-1], t_phase2[-1]], [sol2[-1, 0], sol3[0, 0]], color='blue', linestyle='solid', linewidth=2)
        if A_phase3 and A_phase4:
            plt.plot([t_phase3[-1], t_phase3[-1]], [sol3[-1, 0], sol4[0, 0]], color='blue', linestyle='solid', linewidth=2)

    # For species B (red).
    if b != 0:
        if B_phase1 and B_phase2:
            plt.plot([t_phase1[-1], t_phase1[-1]], [sol1[-1, 1], sol2[0, 1]], color='red', linestyle='solid', linewidth=2)
        if B_phase2 and B_phase3:
            plt.plot([t_phase2[-1], t_phase2[-1]], [sol2[-1, 1], sol3[0, 1]], color='red', linestyle='solid', linewidth=2)
        if B_phase3 and B_phase4:
            plt.plot([t_phase3[-1], t_phase3[-1]], [sol3[-1, 1], sol4[0, 1]], color='red', linestyle='solid', linewidth=2)

    # For species C (green).
    if c != 0:
        if C_phase1 and C_phase2:
            plt.plot([t_phase1[-1], t_phase1[-1]], [sol1[-1, 2], sol2[0, 2]], color='green', linestyle='solid', linewidth=2)
        if C_phase2 and C_phase3:
            plt.plot([t_phase2[-1], t_phase2[-1]], [sol2[-1, 2], sol3[0, 2]], color='green', linestyle='solid', linewidth=2)
        if C_phase3 and C_phase4:
            plt.plot([t_phase3[-1], t_phase3[-1]], [sol3[-1, 2], sol4[0, 2]], color='green', linestyle='solid', linewidth=2)

    # For species D (purple).
    if d != 0:
        if D_phase1 and D_phase2:
            plt.plot([t_phase1[-1], t_phase1[-1]], [sol1[-1, 3], sol2[0, 3]], color='purple', linestyle='solid', linewidth=2)
        if D_phase2 and D_phase3:
            plt.plot([t_phase2[-1], t_phase2[-1]], [sol2[-1, 3], sol3[0, 3]], color='purple', linestyle='solid', linewidth=2)
        if D_phase3 and D_phase4:
            plt.plot([t_phase3[-1], t_phase3[-1]], [sol3[-1, 3], sol4[0, 3]], color='purple', linestyle='solid', linewidth=2)

    plt.xlabel("Time")
    plt.ylabel("Concentration")
    if show_title:
        plt.title("Generic Reaction: {}A + {}B ↔ {}C + {}D  |  Reaction: {}  |  Temp Effect: {:+.2f}  |  Vol Effect: {:+.2f}".format(
            a, b, c, d, reaction_type, temp_effect, vol_effect))
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
vol_effect = st.sidebar.slider("Vol Effect", min_value=-0.5, max_value=0.5, value=0.0, step=0.05)

# Checkbox to show/hide the plot title.
show_title = st.sidebar.checkbox("Show Plot Title", value=True)

# Species perturbations
st.sidebar.header("Species Perturbations")
A_perturb = st.sidebar.slider("A Perturb", min_value=-0.5, max_value=0.5, value=0.0, step=0.05)
B_perturb = st.sidebar.slider("B Perturb", min_value=-0.5, max_value=0.5, value=0.0, step=0.05)
C_perturb = st.sidebar.slider("C Perturb", min_value=-0.5, max_value=0.5, value=0.0, step=0.05)
D_perturb = st.sidebar.slider("D Perturb", min_value=-0.5, max_value=0.5, value=0.0, step=0.05)

# Show/hide phase sections
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

# Run simulation and display plot
fig = simulate_reaction(a, b, c, d, reaction_type,
                        temp_effect, vol_effect,
                        A_perturb, B_perturb, C_perturb, D_perturb,
                        A_phase1, A_phase2, A_phase3, A_phase4,
                        B_phase1, B_phase2, B_phase3, B_phase4,
                        C_phase1, C_phase2, C_phase3, C_phase4,
                        D_phase1, D_phase2, D_phase3, D_phase4,
                        show_title)

st.pyplot(fig)
