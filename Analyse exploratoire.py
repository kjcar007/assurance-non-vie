# -*- coding: utf-8 -*-
"""
Analyse exploratoire - Base d'assurance auto P and V 
=====================================================

"""

# %% 0. Imports et paramètres
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
CSV_PATH = BASE / "Dataset.csv"
OUT = BASE / "sorties_rapport"
OUT.mkdir(exist_ok=True)
AFFICHER = True            # False = enregistrer les figures sans les afficher
BLEU, ORANGE, ROUGE = "#386CA6", "#F97D0B", "#8B0000"
SEUIL_CORR = 0.4           # seuil de multicolinéarité (valeur absolue)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7})


def finir(fig, nom):
    fig.tight_layout()
    fig.savefig(OUT / f"{nom}.png", dpi=150)
    if AFFICHER:
        plt.show()
    plt.close(fig)


def grille(n, ncols=2, h=3.3):
    """Crée une grille de n sous-graphiques ; les cases en trop sont masquées."""
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, h * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]


def lettre(i):
    return f"({chr(97 + i)}) "


# %% 1. Chargement et réparation des nombres corrompus
# Certains nombres du CSV ont perdu leur virgule et gagné des points de milliers
# (ex. '10.020.533.881' pour 1.0020533881). On lit tout en texte puis on répare.
brut = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8-sig", dtype=str,
                   na_values=["NA"], keep_default_na=False)


def repare_decimal(s, n_ent):
    if pd.isna(s) or s.count(".") <= 1:
        return float(s) if pd.notna(s) else np.nan
    d = s.replace(".", "")
    return float(d[:n_ent] + "." + d[n_ent:])


def repare_montant(s, ref):
    """Position de la virgule inconnue : on garde la valeur (1 à 6 chiffres entiers)
    la plus proche de la médiane 'ref' en échelle log. => APPROXIMATION."""
    if pd.isna(s) or s.count(".") <= 1:
        return float(s) if pd.notna(s) else np.nan
    d = s.replace(".", "").rstrip("0")
    cand = [round(float(d[:k] + "." + d[k:]), 2) for k in range(1, min(7, len(d)))]
    return min(cand, key=lambda v: abs(np.log10(v) - np.log10(ref)))


df = brut.copy()
corrompu = {c: df[c].fillna("").str.count(r"\.") > 1
            for c in ["claim_amt_cut_tpl_md", "claim_amt_cut_tpl_bi"]}
df["admi_risk_exposure"] = df["admi_risk_exposure"].map(lambda s: repare_decimal(s, 1))
df["geo_postcode_lng"] = df["geo_postcode_lng"].map(lambda s: repare_decimal(s, 1))
df["geo_postcode_lat"] = df["geo_postcode_lat"].map(lambda s: repare_decimal(s, 2))
for c, m in corrompu.items():
    ok = pd.to_numeric(df.loc[~m, c], errors="coerce")
    ref = ok[ok > 0].median()
    df[c] = df[c].map(lambda s: repare_montant(s, ref))

cat_txt = ["veh_fuel", "veh_make", "veh_type", "veh_use", "purpose",
           "geo_region_fr", "geo_province_fr", "geo_munty_fr"]
for c in df.columns:
    if c not in cat_txt:
        df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.rename(columns={"geo_postcode_2digits": "geo_postcode"})


# %% 2. Description de la base et pré-traitements
print("Dimensions :", df.shape)
print("Années :", sorted(df["admi_risk_year"].unique()))
print("Répartition train / game :\n", df["purpose"].value_counts())
print("\nValeurs manquantes (%) :")
print((df.isna().mean() * 100).round(1)[lambda s: s > 0])

# (a) Variable supprimée : veh_years_claim_free (71 % de NA, collectée progressivement)
print("\nNA de veh_years_claim_free par année (%) :")
print(df.groupby("admi_risk_year")["veh_years_claim_free"].apply(lambda s: s.isna().mean() * 100).round(1))
df = df.drop(columns=["veh_years_claim_free"])

# (b) Zéros = information non renseignée -> remplacés par la médiane (plus robuste que la moyenne).
#     veh_age == 0 est conservé : ce sont des véhicules neufs (4,5 % du portefeuille).
for c in ["veh_power", "veh_weight", "veh_value"]:
    n0 = (df[c] == 0).sum()
    if n0:
        df.loc[df[c] == 0, c] = df.loc[df[c] > 0, c].median()
    print(f"{c} : {n0} zéros remplacés par la médiane")

# (c) Regroupements : marques par origine, sièges en 3 classes
ORIGINE = {"bmw": "german", "audi": "german", "mercedes": "german", "volkswagen": "german",
           "opel": "german", "peugeot": "french", "renault": "french", "citroen": "french",
           "ford": "american", "fiat": "italian"}
df["veh_make"] = df["veh_make"].map(ORIGINE)
df["veh_seats"] = pd.cut(df["veh_seats"], [0, 2, 5, 9], labels=["small_1_2", "standard_3_5", "large_6_9"])

# (d) Sinistralité : disponible uniquement sur 'train' (les sinistres de 'game' sont masqués)
app = df[df["purpose"] == "train"].copy()
print("\nLignes train utilisées pour la sinistralité :", len(app))


def freq(data, col, cible="claim_nb_tpl_md"):
    """Fréquence moyenne de sinistres par année d'exposition, par modalité."""
    g = data.groupby(col, observed=True).agg(n=("id", "size"), expo=("admi_risk_exposure", "sum"),
                                              sin=(cible, "sum"))
    g["frequence"] = g["sin"] / g["expo"]
    return g


# %% 3. Analyse descriptive : variables continues
cont1 = [("veh_age", "Âge du véhicule"), ("driv_m_age", "Âge du conducteur principal"),
         ("driv_y_age", "Âge du conducteur secondaire"), ("cont_seniority", "Ancienneté du contrat")]
cont2 = [("admi_risk_exposure", "Exposition au risque"), ("veh_value", "Valeur du véhicule"),
         ("veh_weight", "Poids du véhicule"), ("veh_power", "Puissance du véhicule")]


def histos(liste, nom, titre):
    fig, axes = grille(len(liste))
    for i, (ax, (c, lib)) in enumerate(zip(axes, liste)):
        s = df[c].dropna()
        if c == "veh_weight":
            s = s[s <= 6000]          # 21 000 kg max : on tronque l'affichage
        ax.hist(s, bins=30, color=BLEU, edgecolor="black", linewidth=0.5)
        ax.set_title(lettre(i) + lib)
        ax.set_xlabel(c)
        ax.set_ylabel("Nombre d'assurés")
    fig.suptitle(titre, fontsize=10)
    finir(fig, nom)


histos(cont1, "fig01_histogrammes_1", "Histogrammes des variables du portefeuille (1/2)")
histos(cont2, "fig02_histogrammes_2", "Histogrammes des variables du portefeuille (2/2)")
print("\nPoids > 6000 kg exclus de l'affichage :", (df["veh_weight"] > 6000).sum())
print(df[["veh_age", "driv_m_age", "driv_y_age", "cont_seniority", "admi_risk_exposure",
          "veh_value", "veh_weight", "veh_power"]].describe().T.round(2))


# %% 4. Analyse descriptive : variables catégorielles
cat1 = [("veh_seats", "Catégories de sièges"), ("veh_make", "Marques de véhicules"),
        ("veh_fuel", "Types de carburant"), ("veh_type", "Types de véhicules")]
cat2 = [("veh_use", "Usage du véhicule"), ("geo_region_fr", "Répartition par région"),
        ("driv_y_add_flg", "Présence d'un conducteur secondaire")]


def barres(liste, nom, titre):
    fig, axes = grille(len(liste))
    for i, (ax, (c, lib)) in enumerate(zip(axes, liste)):
        vc = df[c].value_counts(dropna=True).sort_index()
        ax.bar(vc.index.astype(str), vc.values, color=ORANGE)
        ax.set_title(lettre(i) + lib)
        ax.set_ylabel("Nombre d'assurés")
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle(titre, fontsize=10)
    finir(fig, nom)


barres(cat1, "fig03_categorielles_1", "Répartition des variables catégorielles (1/2)")
barres(cat2, "fig04_categorielles_2", "Répartition des variables catégorielles (2/2)")
for c, _ in cat1 + cat2:
    print(f"\n{c} (%)\n", (df[c].value_counts(normalize=True, dropna=False) * 100).round(1))

# Nombre de sinistres (train)
fig, axes = grille(2)
for i, (ax, (c, lib)) in enumerate(zip(axes, [("claim_nb_tpl_md", "Nombre de sinistres matériels"),
                                              ("claim_nb_tpl_bi", "Nombre de sinistres corporels")])):
    vc = app[c].value_counts().sort_index()
    ax.bar(vc.index.astype(int).astype(str), vc.values, color=BLEU, edgecolor="black")
    ax.set_title(lettre(i) + lib)
    ax.set_xlabel("Nombre de sinistres")
    ax.set_ylabel("Nombre d'assurés")
    print(c, vc.to_dict(), f"-> {vc.get(1, 0) / vc.sum() * 100:.2f} % des polices avec sinistre")
fig.suptitle("Comparaison du nombre de sinistres matériels et corporels (train)", fontsize=10)
finir(fig, "fig05_nb_sinistres")


# %% 5. Effets marginaux (fréquence = sinistres / exposition, base train)
marg_cat = [("veh_fuel", "Type de carburant"), ("veh_make", "Type de marque"),
            ("veh_type", "Type de véhicule"), ("veh_use", "Usage du véhicule"),
            ("geo_region_fr", "Région"), ("veh_seats", "Nombre de sièges"),
            ("driv_y_add_flg", "Présence d'un conducteur secondaire")]
marg_cont = [("veh_age", "Âge du véhicule"), ("driv_m_age", "Âge du conducteur principal"),
             ("veh_value", "Valeur du véhicule"), ("cont_seniority", "Ancienneté du contrat"),
             ("veh_power", "Puissance du véhicule"), ("veh_weight", "Poids du véhicule"),
             ("driv_y_age", "Âge du conducteur secondaire")]


def effets(cible, suffixe, titre):
    # variables catégorielles : un point par modalité
    fig, axes = grille(len(marg_cat), ncols=3, h=3.0)
    for i, (ax, (c, lib)) in enumerate(zip(axes, marg_cat)):
        g = freq(app, c, cible)
        ax.plot(g.index.astype(str), g["frequence"], "o", color=BLEU, markersize=7)
        ax.set_title(lettre(i) + lib)
        ax.set_ylabel("Fréquence moyenne")
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle(f"Effets marginaux des variables catégorielles ({titre})", fontsize=10)
    finir(fig, f"fig{suffixe}a_effets_cat")
    # variables continues : déciles (classes d'effectifs comparables)
    fig, axes = grille(len(marg_cont), ncols=3, h=3.2)
    tab = {}
    for i, (ax, (c, lib)) in enumerate(zip(axes, marg_cont)):
        classes = pd.qcut(app[c], 10, duplicates="drop")
        g = freq(app.assign(classe=classes), "classe", cible)
        tab[c] = g
        ax.plot(range(len(g)), g["frequence"], "o-", color=ROUGE, linewidth=1)
        ax.set_xticks(range(len(g)), [f"{iv.right:.0f}" for iv in g.index], rotation=45)
        ax.set_title(lettre(i) + lib + " (borne sup. des classes)")
        ax.set_ylabel("Fréquence moyenne")
    fig.suptitle(f"Effets marginaux des variables continues ({titre})", fontsize=10)
    finir(fig, f"fig{suffixe}b_effets_cont")
    return tab


tab_md = effets("claim_nb_tpl_md", "06", "sinistres matériels")
tab_bi = effets("claim_nb_tpl_bi", "09", "sinistres corporels - annexe")
for c, _ in marg_cat:
    print(f"\n--- MD par {c}\n", freq(app, c).round(4))
    print(f"--- BI par {c}\n", freq(app, c, "claim_nb_tpl_bi").round(5))
for c, g in tab_md.items():
    print(f"\n--- MD par déciles de {c}\n", g.round(4))


# %% 6. Corrélations entre variables
# Les variables catégorielles sont codées numériquement (codes arbitraires) : la lecture de ces
# corrélations est donc indicative, comme dans les matrices de corrélation "tout-en-un".
ordre = ["admi_risk_exposure", "claim_nb_tpl_md", "claim_nb_tpl_bi", "veh_age", "veh_power",
         "veh_value", "veh_weight", "cont_seniority", "driv_m_age", "driv_y_age", "driv_y_add_flg",
         "veh_fuel", "veh_make", "veh_seats", "veh_type", "veh_use", "geo_postcode_lng",
         "geo_postcode_lat", "geo_region_fr"]
num = df[ordre].copy()
for c in ["veh_fuel", "veh_make", "veh_seats", "veh_type", "veh_use", "geo_region_fr"]:
    num[c] = num[c].astype("category").cat.codes
corr = num.corr()          # Pearson, observations deux à deux complètes

fig, ax = plt.subplots(figsize=(8.2, 7))
masque = np.triu(np.ones_like(corr, dtype=bool), k=1)
im = ax.imshow(np.ma.array(corr.values, mask=masque), cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(ordre)), ordre, rotation=60, ha="right", fontsize=7)
ax.set_yticks(range(len(ordre)), ordre, fontsize=7)
for i in range(len(ordre)):
    for j in range(i + 1):
        if i != j and abs(corr.iloc[i, j]) >= 0.2:
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
ax.grid(False)
fig.colorbar(im, shrink=0.75)
ax.set_title("Corrélation entre les variables")
finir(fig, "fig10_correlations")

fortes = (corr.where(~masque & ~np.eye(len(ordre), dtype=bool)).stack()
          .pipe(lambda s: s[s.abs() > SEUIL_CORR]).sort_values(key=abs, ascending=False))
print(f"\nCorrélations |r| > {SEUIL_CORR} :\n", fortes.round(2))
print("\nCorrélations des sinistres avec les autres variables (max |r|) :")
for c in ["claim_nb_tpl_md", "claim_nb_tpl_bi"]:
    s = corr[c].drop(c).abs().sort_values(ascending=False)
    print(c, s.head(3).round(3).to_dict())
corr.round(3).to_csv(OUT / "matrice_correlation.csv")

# Fin
print("\nTerminé. Sorties dans :", OUT)        
