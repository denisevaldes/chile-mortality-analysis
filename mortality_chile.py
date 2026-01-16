import numpy
import streamlit as st
import pandas as pd
import json
import plotly.express as px
import unicodedata
import re
import numpy as np
import os
import gdown
import plotly.graph_objects as go   

# Configuración de página para que se vea profesional
st.set_page_config(page_title="Chile Mortality Analysis", layout="wide")

def norm_key(s):
    if not s: return ""
    # 1. Quitar acentos y eñes para evitar el error UnicodeDecodeError
    s = str(s).lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    # 2. Limpieza de palabras ruidosas
    s = s.replace("region de ", "").replace("region del ", "")
    s = s.replace("de ", "").replace("del ", "")
    # 3. Quitar puntos y caracteres especiales que ensucian el merge
    s = s.replace(".", "").replace("'", "").replace("-", " ")
    return " ".join(s.split())

def normalize_col(col):
    col = str(col).strip()
    col = unicodedata.normalize("NFKD", col)
    col = "".join(c for c in col if not unicodedata.combining(c))
    return col.upper()

FILE_ID = "1WTUxtHFCXPSNgABeFdzfDpJDC1tyR4xm"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

@st.cache_data
def load_all_data():
    # ==== 1. DEFUNCIONES (CSV grande desde Drive) ====
    FILE_ID_DEF = "1WTUxtHFCXPSNgABeFdzfDpJDC1tyR4xm"
    URL_DEF = f"https://drive.google.com/uc?id={FILE_ID_DEF}"
    PATH_DEF = "defunciones.csv"

    if not os.path.exists(PATH_DEF):
        gdown.download(URL_DEF, PATH_DEF, quiet=False)

    df = pd.read_csv(
        PATH_DEF,
        sep=";",
        encoding="latin1",
        low_memory=False
    )

    # Normalizar columnas (CRÍTICO)
    df.columns = [normalize_col(c) for c in df.columns]
    
    # 2. Carga de Censo (Excel)
    df_pobl = pd.read_excel(
        "D1_Poblacion-censada-por-sexo-y-edad-en-grupos-quinquenales.xlsx",
        sheet_name=1,
        header=3
    )

    # --- Limpieza de Defunciones ---
    cols_to_drop = df.columns[df.isna().sum() > 1000]
    df = df.drop(columns=cols_to_drop)
    df["ANO"] = pd.to_numeric(df["ANO"], errors="coerce")
    df = df[df["ANO"].notna() & (df["ANO"] <= 2025)]
    df = df[df["NOMBRE_REGION"] != "Ignorada"]
    
    # --- Limpieza de Censo ---
    df_pobl = df_pobl[~df_pobl["Región"].isin(["País", "Censo", "NAN"])].dropna(subset=["Región"])
    df_pobl = df_pobl.rename(columns={"Región": "NOMBRE_REGION_POBL", "Población censada": "POBLACION"})

    # ✅ CREAR REGION_KEY AQUÍ (esto arregla tu error)
    df_pobl["REGION_KEY"] = df_pobl["NOMBRE_REGION_POBL"].apply(norm_key)
    
    # --- Procesamiento de Edad y Género ---
    def calcular_edad(row):
        tipo, cant = row['EDAD_TIPO'], row['EDAD_CANT']
        if tipo == 1: return cant
        elif tipo == 2: return cant / 12
        else: return cant / 365
    df['AGE_YEARS'] = df.apply(calcular_edad, axis=1)

    # --- Create age in years (EDAD_ANOS) and age group (TRAMO_ETARIO) ---
    df["EDAD_ANOS"] = df["AGE_YEARS"]  # reuse what you already computed

    bins = [0, 1, 15, 30, 45, 60, 75, 90, 120]
    labels = ['<1 year', '1-14', '15-29', '30-44', '45-59', '60-74', '75-89', '90+']

    df["TRAMO_ETARIO"] = pd.cut(df["EDAD_ANOS"], bins=bins, labels=labels, right=False)

    
    gender_map = {"Hombre": "Male", "Mujer": "Female", "Indeterminado": "Undetermined"}
    df['GENDER'] = df['SEXO_NOMBRE'].map(gender_map)
    
    return df, df_pobl

@st.cache_data
def load_geojson():
    with open("Regional.geojson", "r", encoding="utf-8") as f:
        return json.load(f)

# --- INICIO DE CARGA ---
df, df_pobl_raw = load_all_data()
chile_geojson = load_geojson()

# --- SIDEBAR ---
st.sidebar.header("Dashboard Filters")
selected_gender = st.sidebar.multiselect("Select Gender:", options=df["GENDER"].unique(), default=["Male", "Female"])
df_filtered = df[df["GENDER"].isin(selected_gender)]

TOP_N = st.sidebar.slider(
    "Top causes per place (Treemap)",
    min_value=3,
    max_value=15,
    value=8,
    step=1
)

st.sidebar.subheader("Age pyramid settings")

year_opt = st.sidebar.multiselect(
    "Select year(s)",
    options=sorted(df["ANO"].unique()),
    default=sorted(df["ANO"].unique())
)

sort_order = st.sidebar.selectbox(
    "Age order",
    options=["Youngest to oldest", "Oldest to youngest"],
    index=0
)

# Filter dataset by selected years for age pyramid
df_filtered_year = df_filtered[df_filtered["ANO"].isin(year_opt)]

# --- UI PRINCIPAL ---
st.title("Mortality Trends in Chile (2023-2025)")
st.markdown("Analysis based on Ministry of Health (DEIS) records and Census 2024 population data.")

st.subheader("Dataset summary")

col1, col2, col3 = st.columns(3)
col1.metric("Total records", f"{len(df_filtered):,}")
col2.metric("Years covered", "2023–2025")
col3.metric("Regions", df_filtered["NOMBRE_REGION"].nunique())

with st.expander("View sample of the dataset", expanded=False):
    st.dataframe(
        df_filtered.sample(500, random_state=42),
        use_container_width=True
    )

# --- MORTALITY TREND OVER TIME ---
st.subheader("Mortality trend over time")

# 0) Make sure the date column is datetime
df_trend = df_filtered.copy()
df_trend["FECHA_DEF"] = pd.to_datetime(df_trend["FECHA_DEF"], errors="coerce")

# Optional: drop rows with invalid dates
df_trend = df_trend.dropna(subset=["FECHA_DEF"])

# 1) Monthly aggregation
df_monthly = (
    df_trend
    .groupby([pd.Grouper(key="FECHA_DEF", freq="MS"), "GENDER"])  # use your English gender column
    .size()
    .reset_index(name="DEATHS")
)

# 2) Line chart
fig_trend = px.line(
    df_monthly,
    x="FECHA_DEF",
    y="DEATHS",
    color="GENDER",
    title="Monthly mortality trend in Chile (2023–2025)",
    labels={"FECHA_DEF": "Month", "DEATHS": "Number of deaths", "GENDER": "Gender"},
    markers=True,
    template="plotly_white"
)

fig_trend.update_xaxes(
    dtick="M1",
    tickformat="%b %Y",
    tickangle=-45
)

fig_trend.update_layout(
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=0, r=0, t=60, b=0)
)

st.plotly_chart(fig_trend, use_container_width=True)

# --- REGION NAME MAPPING: data -> exact GeoJSON names ---
REGION_MAP = {
    "De Arica y Parinacota": "Región de Arica y Parinacota",
    "De Tarapacá": "Región de Tarapacá",
    "De Antofagasta": "Región de Antofagasta",
    "De Atacama": "Región de Atacama",
    "De Coquimbo": "Región de Coquimbo",
    "De Valparaíso": "Región de Valparaíso",
    "Metropolitana de Santiago": "Región Metropolitana de Santiago",
    "Del Libertador B. O'Higgins": "Región del Libertador Bernardo O'Higgins",
    "Del Maule": "Región del Maule",
    "De Ñuble": "Región de Ñuble",
    "Del Bíobío": "Región del Bío-Bío",
    "De La Araucanía": "Región de La Araucanía",
    "De Los Ríos": "Región de Los Ríos",
    "De Los Lagos": "Región de Los Lagos",
    "De Aisén del Gral. C. Ibáñez del Campo": "Región de Aysén del Gral.Ibañez del Campo",
    "De Magallanes y de La Antártica Chilena": "Región de Magallanes y Antártica Chilena",
}

# --- REGION KEY ALIASES (only edge cases) ---
# ALIAS EXACTOS: Estos nombres deben coincidir con tu Excel del Censo y el DEIS
REGION_KEY_ALIASES = {
    "aisen gral c ibanez campo": "aisen general carlos ibanez campo",
    "libertador b ohiggins": "libertador general bernardo ohiggins",
    "metropolitana santiago": "metropolitana santiago",
    "magallanes antartica chilena": "magallanes antartica chilena"
}

def final_region_key(name: str) -> str:
    # Pasamos a minúsculas y quitamos acentos
    s = str(name).lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    
    # 1. Caso Aysén: Buscamos palabras clave sin importar el orden o puntos
    if "aisen" in s or "ibanez" in s:
        return "aisen general carlos ibanez del campo"
    
    # 2. Caso O'Higgins
    if "higgins" in s:
        return "libertador general bernardo ohiggins"
    
    # 3. Caso Metropolitana
    if "metropolitana" in s:
        return "metropolitana santiago"

    # Limpieza estándar para las demás
    for word in ["region de ", "region del ", "de ", "del "]:
        s = s.replace(word, "")
    return " ".join(s.split()).strip()

def build_region_table(df_in, df_pobl, chile_geojson):
    # 1. Conteo de defunciones
    df_region = df_in.groupby("NOMBRE_REGION").size().reset_index(name="DEATHS")
    df_region = df_region[df_region["NOMBRE_REGION"].str.lower() != "ignorada"]

    # 2. Generar llaves de unión
    df_region["REGION_KEY"] = df_region["NOMBRE_REGION"].apply(final_region_key)
    
    # Preparamos la tabla de población del Censo 2024
    df_pobl_clean = df_pobl.copy()
    df_pobl_clean["REGION_KEY"] = df_pobl_clean["NOMBRE_REGION_POBL"].apply(final_region_key)
    # Eliminamos duplicados en la tabla de población para evitar errores de merge
    df_pobl_clean = df_pobl_clean.drop_duplicates(subset=["REGION_KEY"])

    # 3. Merge (Unión de datos)
    df_merge = df_region.merge(
        df_pobl_clean[["REGION_KEY", "POBLACION"]],
        on="REGION_KEY",
        how="left"
    )

    # 4. Cálculo de Tasa por 100k habitantes
    df_merge["POBLACION"] = pd.to_numeric(df_merge["POBLACION"], errors="coerce")
    df_merge["RATE_100K"] = (df_merge["DEATHS"] / df_merge["POBLACION"]) * 100000
    df_merge["RATE_100K"] = df_merge["RATE_100K"].round(1)

    # 5. Mapeo para el GeoJSON (Asegúrate que coincida con el archivo .geojson)
    df_merge["REGION_GEO"] = df_merge["NOMBRE_REGION"].map(REGION_MAP)

    geo_regions = set(f["properties"]["Region"] for f in chile_geojson["features"])
    missing_geo = df_merge[df_merge["REGION_GEO"].isna()]
    not_in_geojson = set(df_merge["REGION_GEO"].dropna().unique()) - geo_regions
    missing_pop = df_merge[df_merge["POBLACION"].isna()]

    # Al final de la función build_region_table:
    return df_merge, missing_geo, not_in_geojson, missing_pop


def build_choropleth_figure(df_region, chile_geojson):
    """
    Plotly choropleth for Streamlit. Uses mapbox version (more compatible).
    """
    fig = px.choropleth_mapbox(
        df_region,
        geojson=chile_geojson,
        locations="REGION_GEO",
        featureidkey="properties.Region",
        color="RATE_100K",
        color_continuous_scale="PuBu",
        hover_name="NOMBRE_REGION",
        hover_data={
            "DEATHS": True,
            "POBLACION": True,
            "RATE_100K": ':.1f',
            "REGION_GEO": False,
            "REGION_KEY": False
        },
        center={"lat": -35, "lon": -71},
        zoom=3.3,
        mapbox_style="carto-positron",
        title="Death rate per 100,000 population by region (2023–2025)"
    )

    fig.update_layout(
    height=1000,              # Aumentamos el alto para estirar el país
    width=450,               # Reducimos el ancho para que sea una franja vertical
    margin=dict(l=10, r=10, t=50, b=10),
    mapbox=dict(
        center={"lat": -35.6, "lon": -71.5}, # Centrado en la zona central de Chile
        zoom=3.8,            # Zoom ideal para ver todo el territorio
        style="carto-positron"
    )
    )
    return fig

st.header("Geographic distribution")

df_region, missing_geo, not_in_geojson, missing_pop = build_region_table(
    df_filtered, df_pobl_raw, chile_geojson
)

# Plot
fig_map = build_choropleth_figure(df_region.dropna(subset=["REGION_GEO"]), chile_geojson)
#st.plotly_chart(fig_map, use_container_width=True)
st.plotly_chart(fig_map, use_container_width=True)

# --- CAUSES OF DEATH BY PLACE---

st.subheader("Causes of death by place of occurrence")

# Work on filtered dataset so it respects sidebar filters
df_lugar = df_filtered.copy()

# Basic cleaning (avoid NaNs)
df_lugar = df_lugar.dropna(subset=["LUGAR_DEFUNCION", "GLOSA_CAPITULO_DIAG1"]).copy()

# Optional: normalize place labels to English (edit if your values differ)
place_map = {
    "Casa": "Home",
    "Hospital o Clínica": "Hospital/Clinic",
    "Hospital/Clínica": "Hospital/Clinic",
    "Otro": "Other",
    "Otros": "Other"
}
df_lugar["PLACE"] = df_lugar["LUGAR_DEFUNCION"].replace(place_map)

# 1) Group by place and chapter
g = (
    df_lugar
    .groupby(["PLACE", "GLOSA_CAPITULO_DIAG1"])
    .size()
    .reset_index(name="DEATHS")
)

# 2) Rank within each place
g["RANK"] = g.groupby("PLACE")["DEATHS"].rank(method="first", ascending=False)

# 3) Top N keep name, rest go into "Other causes"
g["GROUP"] = np.where(
    g["RANK"] <= TOP_N,
    g["GLOSA_CAPITULO_DIAG1"],
    "Other causes"
)

# 4) Only "Other causes" can expand (so Top N doesn't split further)
g["DETAIL"] = np.where(
    g["GROUP"].eq("Other causes"),
    g["GLOSA_CAPITULO_DIAG1"],
    None
)

# 5) Treemap
fig = px.treemap(
    g,
    path=["PLACE", "GROUP", "DETAIL"],
    values="DEATHS",
    title=f"Causes of death by place of occurrence (Top {TOP_N} + expandable 'Other causes')"
)

fig.update_traces(
    textinfo="label+percent parent",
    textfont=dict(
        size=20,
        color="white"
    ),
    marker=dict(
        line=dict(color="black", width=0.6)
    )
)

fig.update_layout(
    height=950,
    margin=dict(t=70, l=10, r=10, b=10)
)

st.plotly_chart(fig, use_container_width=True)
# --- AGE PYRAMID (Place this after you create df_filtered_year and sidebar controls) ---
# Requirements:
# - df_filtered_year exists (already filtered by gender + year)
# - df_filtered_year has columns: TRAMO_ETARIO, GENDER, and optionally SEXO_NOMBRE if you use it
# - sidebar variables exist: sort_order (string) and show_undetermined (bool)
st.subheader("Mortality age pyramid (2023–2025)")

# 1) Start from the already-filtered dataframe (IMPORTANT)
df_pyr = df_filtered_year.copy()

# 3) Define the epidemiologically correct age order (DO NOT use sorted())
AGE_ORDER = [
    "<1 year", "1-14", "15-29", "30-44",
    "45-59", "60-74", "75-89", "90+"
]

# 4) Aggregate deaths by age group and gender
df_pyramid = (
    df_pyr
    .dropna(subset=["TRAMO_ETARIO", "GENDER"])
    .groupby(["TRAMO_ETARIO", "GENDER"], observed=False)
    .size()
    .reset_index(name="DEATHS")
)

# 5) Force categorical ordering (CRITICAL)
df_pyramid["TRAMO_ETARIO"] = pd.Categorical(
    df_pyramid["TRAMO_ETARIO"],
    categories=AGE_ORDER,
    ordered=True
)

# 6) Sidebar-controlled direction
age_order = AGE_ORDER[::-1] if sort_order == "Oldest to youngest" else AGE_ORDER

# 7) Split by gender (English labels)
male = df_pyramid[df_pyramid["GENDER"] == "Male"].sort_values("TRAMO_ETARIO")
female = df_pyramid[df_pyramid["GENDER"] == "Female"].sort_values("TRAMO_ETARIO")

# 8) Build the pyramid figure
fig = go.Figure()

fig.add_trace(go.Bar(
    y=male["TRAMO_ETARIO"],
    x=-male["DEATHS"],
    name="Male",
    orientation="h",
    customdata=male["DEATHS"],
    hovertemplate=(
        "<b>Age group:</b> %{y}<br>"
        "<b>Gender:</b> Male<br>"
        "<b>Deaths:</b> %{customdata:,}<extra></extra>"
    )
))

fig.add_trace(go.Bar(
    y=female["TRAMO_ETARIO"],
    x=female["DEATHS"],
    name="Female",
    orientation="h",
    customdata=female["DEATHS"],
    hovertemplate=(
        "<b>Age group:</b> %{y}<br>"
        "<b>Gender:</b> Female<br>"
        "<b>Deaths:</b> %{customdata:,}"
        "<extra></extra>"
    )
))

# 9) Make x-axis show absolute values
max_x = max(
    male["DEATHS"].max() if len(male) else 0,
    female["DEATHS"].max() if len(female) else 0
)

fig.update_layout(
    title="Mortality age pyramid by sex",
    barmode="relative",
    bargap=0.1,
    xaxis=dict(
        title="Number of deaths",
        range=[-max_x * 1.1, max_x * 1.1],
        tickvals=[-max_x, -max_x/2, 0, max_x/2, max_x],
        ticktext=[f"{int(max_x)}", f"{int(max_x/2)}", "0", f"{int(max_x/2)}", f"{int(max_x)}"]
    ),
    yaxis=dict(
        title="Age group",
        categoryorder="array",
        categoryarray=age_order   # CRITICAL: force the order you want
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=0, r=0, t=60, b=0),
    height=700
)

st.plotly_chart(fig, use_container_width=True)

# --- TOP CAUSES OF DEATH BY GENDER ---

st.subheader("Top causes of death by sex (Top 5)")

df_base = df_filtered_year.dropna(subset=["SEXO_NOMBRE", "GLOSA_SUBCATEGORIA_DIAG1"]).copy()

top_male = (
    df_base[df_base["SEXO_NOMBRE"] == "Hombre"]
    .groupby("GLOSA_SUBCATEGORIA_DIAG1")
    .size()
    .reset_index(name="DEATHS")
    .sort_values("DEATHS", ascending=False)
    .head(5)
    .sort_values("DEATHS", ascending=True)
)

top_female = (
    df_base[df_base["SEXO_NOMBRE"] == "Mujer"]
    .groupby("GLOSA_SUBCATEGORIA_DIAG1")
    .size()
    .reset_index(name="DEATHS")
    .sort_values("DEATHS", ascending=False)
    .head(5)
    .sort_values("DEATHS", ascending=True)
)

tab_male, tab_female = st.tabs(["Male", "Female"])

with tab_male:
    fig_m = px.bar(
        top_male,
        x="DEATHS",
        y="GLOSA_SUBCATEGORIA_DIAG1",
        orientation="h",
        title="Male — Top 5 causes",
        labels={"GLOSA_SUBCATEGORIA_DIAG1": "", "DEATHS": "Deaths"},
        template="plotly_white",
        height=600
    )
    fig_m.update_layout(
        title=dict(font=dict(size=22)),
        font=dict(size=16),
        margin=dict(l=340, r=20, t=80, b=40)
    )
    fig_m.update_yaxes(tickfont=dict(size=16))
    fig_m.update_xaxes(title_font=dict(size=16), tickfont=dict(size=14))

    st.plotly_chart(fig_m, use_container_width=True)

with tab_female:
    fig_f = px.bar(
        top_female,
        x="DEATHS",
        y="GLOSA_SUBCATEGORIA_DIAG1",
        orientation="h",
        title="Female — Top 5 causes",
        labels={"GLOSA_SUBCATEGORIA_DIAG1": "", "DEATHS": "Deaths"},
        template="plotly_white",
        height=600
    )
    fig_f.update_layout(
        title=dict(font=dict(size=22)),
        font=dict(size=16),
        margin=dict(l=340, r=20, t=80, b=40)
    )
    fig_f.update_yaxes(tickfont=dict(size=16))
    fig_f.update_xaxes(title_font=dict(size=16), tickfont=dict(size=14))

    st.plotly_chart(fig_f, use_container_width=True)
