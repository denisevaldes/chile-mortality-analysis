import numpy
import streamlit as st
import pandas as pd
import json
import plotly.express as px
import unicodedata
import re
import numpy as np

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

@st.cache_data
def load_all_data():
    # 1. Carga de Defunciones (Uso de ruta relativa para portabilidad)
    file_id = "1WTUxtHFCXPSNgABeFdzfDpJDC1tyR4xm"
    url = f"https://drive.google.com/uc?id={file_id}"

    df = pd.read_csv(
        url,
        sep=";",
        encoding="latin1",
        low_memory=False
    )
    
    # 2. Carga de Censo (Excel)
    df_pobl = pd.read_excel(
        "D1_Poblacion-censada-por-sexo-y-edad-en-grupos-quinquenales.xlsx",
        sheet_name=1,
        header=3
    )
    
    st.subheader("Columnas detectadas")
    st.write(list(df.columns))

    # --- Limpieza de Defunciones ---
    cols_to_drop = df.columns[df.isna().sum() > 1000]
    df = df.drop(columns=cols_to_drop)
    df = df[df["ANO"] <= 2025]
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

