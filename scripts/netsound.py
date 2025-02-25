import streamlit as st
import numpy as np
import networkx as nx
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def generate_sound_and_visuals(G,audio_type,modulation_index,spectrum_type,pitchtime):
    """
    Generate sound and visuals based on the input graph and audio parameters.

    This function takes a graph, an audio type, and a modulation index as inputs. It computes 
    the adjacency matrix and eigenvalues for the graph, and then uses these to generate sound and 
    visuals. The function also employs Streamlit for layout and Matplotlib for plotting.

    Parameters
    ----------
    G : networkx.Graph
        Input graph for which sound and visuals are to be generated.
    audio_type : str
        Type of audio to be generated. E.g., 'sine', 'square', etc.
    modulation_index : float
        Modulation index for the sound generation.

    Returns
    -------
    None
        The function returns None but produces sound and visual outputs as side effects.

    Raises
    ------
    ValueError
        If the audio_type specified is not supported.
    LinAlgError
        If eigenvalue computation fails.

    Notes
    -----
    - The function uses the NetworkX library for graph analysis.
    - It uses the sounddevice library for sound generation.
    - Matplotlib is used for plotting the graph and additional visuals.
    - Streamlit is used for laying out the user interface.
    """
    if spectrum_type == 'Adjacency Matrix':
        adj_matrix = nx.adjacency_matrix(G).todense()
        eigenvalues, _ = np.linalg.eig(adj_matrix)
        eigenvalues = np.real(eigenvalues)
    elif spectrum_type == 'Laplacian':
        eigenvalues = nx.normalized_laplacian_spectrum(G)
    elif spectrum_type == 'Modularity':
        # Compute modularity matrix
        A = nx.to_numpy_array(G)
        k = np.sum(A, axis=0)
        m = np.sum(k)
        B = A - np.outer(k, k) / m
        # Compute eigenvalues
        eigenvalues, _ = np.linalg.eig(B)
        eigenvalues = np.real(eigenvalues)

    degrees = [d for n, d in G.degree()]
    col1, col2, col3 = st.columns(3)

    plt.style.use('dark_background')

    # Graph Plot with Nodes Colored by Degree
    fig, ax = plt.subplots()
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_edges(G, pos, edge_color='cyan', ax=ax)
    norm = plt.Normalize(min(degrees), max(degrees))
    nx.draw_networkx_nodes(G, pos, node_color=degrees, cmap=plt.cm.plasma, ax=ax)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,norm=norm)
    #sm.set_array([1, max(degrees)])
    sm._A = []
    fig.colorbar(sm, ax=ax, label='Node Degree' )
    col1.pyplot(fig)
    plt.close(fig)

    # Eigenvalue plot with pop colors
    plt.figure()
    plt.bar(range(len(eigenvalues)), eigenvalues, color='cyan')
    plt.xlabel('Eigenvalue Index', color='white')
    plt.ylabel('Eigenvalue', color='white')
    plt.title('Eigenvalues of the Graph', color='white')
    col2.pyplot(plt)
    plt.close()

    # Generate and play sound
    min_freq, max_freq = 200, 4000
    norm_eigenvalues = np.interp(eigenvalues, (eigenvalues.min(), eigenvalues.max()), (min_freq, max_freq))
    fs = 44100
    duration = pitchtime 
    t = np.linspace(0, duration, int(fs * duration), False)
    
    for norm_eigenvalue in norm_eigenvalues:
        modulating_frequency = norm_eigenvalue * modulation_index

        # Fundamental frequency component
        audio = np.sin(2 * np.pi * norm_eigenvalue * t)

        # Add first three harmonics
        for harmonic in range(2, 5):  # 2nd, 3rd, and 4th harmonics
            audio += 0.5 / harmonic * np.sin(2 * np.pi * harmonic * norm_eigenvalue * t)


        if audio_type == "Sine Wave":
            audio = np.sin((norm_eigenvalue + modulating_frequency * np.sin(2.0 * np.pi * modulating_frequency * t)) * 2 * np.pi * t)
        elif audio_type == "Square Wave":
            audio = np.sign(np.sin(2.0 * np.pi * norm_eigenvalue * t))
        elif audio_type == "Sawtooth Wave":
            audio = 0.5 * (1.0 - np.arctan(np.sin(2 * np.pi * norm_eigenvalue * t))/np.pi)
        elif audio_type == "FM Synthesis":
            carrier_freq = norm_eigenvalue
            modulating_freq = 2 * np.pi * modulating_frequency * t
            audio = np.sin(2.0 * np.pi * carrier_freq * t + np.sin(modulating_freq))
        elif audio_type=="Waveshaping Synthesis":
            audio = np.sin(2 * np.pi * norm_eigenvalue * t)
            audio = audio + 0.5 * np.sin(2 * np.pi * 2 * norm_eigenvalue * t)
            audio = np.sign(audio) * (1 - np.exp(-np.abs(audio)))

    
        
        audio *= 0.5 / np.max(np.abs(audio))
        sd.play(audio, samplerate=fs)
        sd.wait()
    
    frequencies, times, Sxx = spectrogram(audio, fs)
    plt.figure()
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), cmap='inferno', shading='gouraud')
    plt.ylabel('Frequency [Hz]', color='white')
    plt.xlabel('Time [sec]', color='white')
    plt.colorbar(label='Intensity [dB]')
    plt.title('Spectrogram', color='white')
    col3.pyplot(plt)
    plt.close()    

st.set_page_config(layout="wide")
st.title("Graph Eigenvalue Sound Generator")

# Sliders moved to sidebar
graph_choice = st.sidebar.selectbox("Choose a Graph Topology", ("Complete Graph", "Cycle Graph", "Random Graph", "Star Graph", "Wheel Graph", "Lollipop Graph",
                                                                 "Barabási–Albert Graph",
                                                                 "Ladder Graph", "Circular Ladder Graph", "Path Graph", "Binominal Tree",
                                                                 "Karate Club Graph", "Florentine Family Graph", "Les miserables Graph"
                                                                 ))
n_nodes = st.sidebar.slider("Number of Nodes", min_value=3, max_value=50, value=12)
spectrum_type = st.sidebar.selectbox("Choose a Spectrum Type", ("Adjacency Matrix", "Laplacian", "Modularity")) 
pitchtime = st.sidebar.slider("Pitch Time", min_value=0.03, max_value=1., value=0.03,step=0.01)  

audio_type = st.sidebar.selectbox("Choose an Audio Type", ("Sine Wave", "Square Wave", "Sawtooth Wave", "FM Synthesis", "Waveshaping Synthesis"))
modulation_index = st.sidebar.slider("Modulation Index", min_value=0.01, max_value=2.0, value=0.01, step=0.01)

if graph_choice == "Complete Graph":
    G = nx.complete_graph(n_nodes)
elif graph_choice == "Cycle Graph":
    G = nx.cycle_graph(n_nodes)
elif graph_choice == "Random Graph":
    G = nx.erdos_renyi_graph(n_nodes, 0.5)
elif graph_choice == "Star Graph":
    G = nx.star_graph(n_nodes - 1)
elif graph_choice == "Wheel Graph":
    G = nx.wheel_graph(n_nodes)
elif graph_choice == "Lollipop Graph":
    m = st.sidebar.slider("Length of the path", min_value=1, max_value=n_nodes-1, value=2)
    G = nx.lollipop_graph(n_nodes, m)
elif graph_choice == "Barabási–Albert Graph":
    m = st.sidebar.slider("Number of edges to attach from new node to existing nodes", min_value=1, max_value=n_nodes-1, value=2)
    G = nx.barabasi_albert_graph(n_nodes, m)
elif graph_choice == "Ladder Graph":
    G = nx.ladder_graph(n_nodes)
elif graph_choice == "Circular Ladder Graph":
    G = nx.circular_ladder_graph(n_nodes)
elif graph_choice == "Path Graph":
    G = nx.path_graph(n_nodes)
elif graph_choice == "Binominal Tree":
    G = nx.binomial_tree(n_nodes)
elif graph_choice == "Karate Club Graph":
    G = nx.karate_club_graph()
elif graph_choice == "Florentine Family Graph":
    G = nx.florentine_families_graph()
elif graph_choice == "Les miserables Graph":
    G = nx.les_miserables_graph()

if st.button("Generate Sound and Visuals"):
    generate_sound_and_visuals(G,audio_type,modulation_index,spectrum_type,pitchtime)
