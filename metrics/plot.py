import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

COLORS = {
        "whisper_large": "#55a868",
        "parakeet": "#4c72b0",
        "assemblyai": "#c44e52",
        "whisperx": "#f0e224"
    }

STYLE = "seaborn-v0_8-whitegrid"

def plot_single_episode(metric_results_df, models, metric_name):
    plt.style.use(STYLE)    

    fig, ax = plt.subplots(figsize=(8, 5))

    id = metric_results_df["Programma"] + "_" + metric_results_df["Data"]
    # Tracciamo una linea per ogni modello
    for model in models:        
        ax.plot(
            id,
            metric_results_df[model],
            marker="o",
            label=model.replace("_", " ").title(),
            color=COLORS.get(model, None)
        )
    
    #ax.set_title(f"{metric_name.upper()} DISTRIBUTION BY EPISODE AND MODEL", fontsize=14, weight="bold")
    ax.set_ylabel(f"{metric_name}", fontsize=18)
    ax.set_xlabel("Program", fontsize=18)
    ax.set_xticks(range(len(metric_results_df)))
    ax.set_xticklabels(id, rotation=90, ha="right")

    ax.legend(title="Model")
    plt.tight_layout()
    plt.show()

def plot_program(metric_results_df, metric_name, models):        

    df_long = metric_results_df.melt(id_vars=["Programma","Data","Tipologia"],
                    var_name="model", value_name=metric_name)
    
    typologies = df_long["Programma"].unique()   
    positions = []
    data_for_plot = []
    labels = []

    group_gap = 1.5  
    width = 0.2      

    for i, typology in enumerate(typologies):
        typology_data = df_long[df_long["Programma"] == typology]
        for j, model in enumerate(models):
            positions.append(i * (len(models) * width + group_gap) + j * width)
            data_for_plot.append(typology_data[typology_data["model"] == model][metric_name])
        labels.append(typology)

    plt.figure(figsize=(10, 6))

    box = plt.boxplot(
        data_for_plot,
        positions=positions,
        widths=width,
        patch_artist=True,
        medianprops=dict(color="black")
    )

    
    for i, patch in enumerate(box["boxes"]):
        model_index = i % len(models)
        patch.set_facecolor(COLORS[models[model_index]])

    tick_positions = []
    for i in range(len(typologies)):
        center = np.mean(positions[i*len(models):(i+1)*len(models)])
        tick_positions.append(center)

    plt.xticks(tick_positions, typologies, fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Program", fontsize=20)
    plt.ylabel(f"{metric_name.upper()}", fontsize=20)
    #plt.title(f"{metric_name.upper()} DISTRIBUTION BY PROGRAM AND MODEL", fontsize=14, weight="bold")
    plt.grid(alpha=0.8)

    # Legenda
    for model in models:
        plt.plot([], [], color=COLORS[model], label=model, linewidth=10)
    plt.legend(title="Model", fontsize=12, title_fontsize=12)

    plt.tight_layout()
    plt.show()

def plot_typology(metric_results_df, metric_name, models):
    df_long = metric_results_df.melt(id_vars=["Programma", "Data", "Tipologia"],
                            var_name="model", value_name=metric_name)

    typologies = df_long["Tipologia"].unique()       

    
    positions = []
    data_for_plot = []
    labels = []

    group_gap = 1.5
    width = 0.2      

    for i, typology in enumerate(typologies):
        typology_data = df_long[df_long["Tipologia"] == typology]
        for j, model in enumerate(models):
            positions.append(i * (len(models) * width + group_gap) + j * width)
            data_for_plot.append(typology_data[typology_data["model"] == model][metric_name])
        labels.append(typology)

    plt.figure(figsize=(10, 6))

    box = plt.boxplot(
        data_for_plot,
        positions=positions,
        widths=width,
        patch_artist=True,
        medianprops=dict(color="black")
    )

    for i, patch in enumerate(box["boxes"]):
        model_index = i % len(models)
        patch.set_facecolor(COLORS[models[model_index]])
    
    tick_positions = []
    for i in range(len(typologies)):
        center = np.mean(positions[i*len(models):(i+1)*len(models)])
        tick_positions.append(center)

    plt.xticks(tick_positions, typologies)
    plt.xlabel("Typology")
    plt.ylabel(f"{metric_name.upper()}")
    plt.title(f"{metric_name.upper()} DISTRIBUTION BY TYPOLOGY AND MODEL", fontsize=14, weight="bold")
    plt.grid(alpha=0.8)
    
    for model in models:
        plt.plot([], [], color=COLORS[model], label=model, linewidth=10)
    plt.legend(title="Model")

    plt.tight_layout()
    plt.show()
