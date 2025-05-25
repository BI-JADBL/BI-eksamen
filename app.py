import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, confusion_matrix, classification_report
from scipy.stats import f_oneway

# Sidebar Navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Vælg sektion:", (
    "Introduktion",
    "Visualiseringer",
    "Feature Importance",
    "Korrelationsmatrix",
    "Klyngeanalyse",
    "ANOVA",
    "Konklusioner"
))

# Load data
df = pd.read_csv("students_dropout_academic_success.csv")
df['target'] = df['target'].replace({'Graduate': 0, 'Dropout': 1, 'Enrolled': 2}).astype(float)
drop_columns = [
    "Application order", "Scholarship holder", "Gender", "Debtor", "Displaced",
    "Curricular units 1st sem (credited)", "Curricular units 2nd sem (credited)",
    "Previous qualification", "Curricular units 2nd sem (without evaluations)",
    "Curricular units 1st sem (without evaluations)", "Marital Status",
    "Daytime/evening attendance", "Nacionality", "International",
    "Educational special needs"
]
df.drop(columns=drop_columns, inplace=True, errors='ignore')

if menu == "Introduktion":
    st.title("Students Dropout and Success Analysis")
    st.write("Denne app præsenterer en omfattende analyse af data om frafald og succes blandt studerende.")



elif menu == "Visualiseringer":
    st.header("Boxplots og Histogrammer")
    filtered_df = df.copy()
    filtered_df['target'] = filtered_df['target'].replace({0: 'Graduate', 1: 'Dropout', 2: 'Enrolled'})

    st.subheader("1. Semester")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    sns.boxplot(data=filtered_df, x='target', y='Curricular units 1st sem (grade)', palette='pastel', ax=axs[0,0])
    sns.histplot(data=filtered_df, x='Curricular units 1st sem (grade)', hue='target', kde=True, palette='pastel', ax=axs[0,1])
    sns.boxplot(data=filtered_df, x='target', y='Curricular units 1st sem (approved)', palette='pastel', ax=axs[1,0])
    sns.histplot(data=filtered_df, x='Curricular units 1st sem (approved)', hue='target', kde=True, palette='pastel', ax=axs[1,1])
    st.pyplot(fig)

    st.subheader("2. Semester")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    sns.boxplot(data=filtered_df, x='target', y='Curricular units 2nd sem (grade)', palette='pastel', ax=axs[0,0])
    sns.histplot(data=filtered_df, x='Curricular units 2nd sem (grade)', hue='target', kde=True, palette='pastel', ax=axs[0,1])
    sns.boxplot(data=filtered_df, x='target', y='Curricular units 2nd sem (approved)', palette='pastel', ax=axs[1,0])
    sns.histplot(data=filtered_df, x='Curricular units 2nd sem (approved)', hue='target', kde=True, palette='pastel', ax=axs[1,1])
    st.pyplot(fig)

    st.subheader("Alder og Økonomi")
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    sns.boxplot(data=filtered_df, x='target', y='Age at enrollment', palette='coolwarm', ax=axs[0,0])
    sns.histplot(data=filtered_df, x='Age at enrollment', hue='target', kde=True, palette='coolwarm', ax=axs[0,1])
    sns.countplot(data=filtered_df, x='Tuition fees up to date', hue='target', palette='coolwarm', ax=axs[1,0])
    axs[1,1].axis('off')
    st.pyplot(fig)

elif menu == "Feature Importance":
    st.header("Feature Importance")
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5, stratify=y)
    rf = RandomForestClassifier(random_state=5)
    rf.fit(X_train, y_train)
    feat_importance = pd.Series(rf.feature_importances_, index=X.columns).nlargest(20) * 100
    feat_table = feat_importance.reset_index()
    feat_table.columns = ['Feature', 'Importance (%)']
    st.dataframe(feat_table.style.background_gradient(cmap='YlGn'))

elif menu == "Korrelationsmatrix":
    st.header("Korrelationsmatrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig)

elif menu == "Klyngeanalyse":
    st.header("KMeans Clustering")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.drop('target', axis=1))
    best_k, best_score, best_labels = None, -1, None
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=7, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    st.write(f"Bedste antal klynger: {best_k}")
    st.write(f"Bedste silhouette score: {best_score:.4f}")

    silhouette_vals = silhouette_samples(X_scaled, best_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10
    for i in range(best_k):
        ith_cluster_silhouette_vals = silhouette_vals[best_labels == i]
        ith_cluster_silhouette_vals.sort()
        size_cluster_i = ith_cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i
        color = plt.cm.viridis(float(i) / best_k)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_vals, facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    ax.axvline(best_score, color="red", linestyle="--")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")
    ax.set_title("Silhouette plot")
    st.pyplot(fig)

elif menu == "ANOVA":
    st.header("ANOVA: Komplet oversigt")
    anova_tests = {
        "Karakterer i 1. semester": ('Curricular units 1st sem (grade)',),
        "Beståede fag i 1. semester": ('Curricular units 1st sem (approved)',),
        "GDP": ('GDP',),
        "Unemployment rate": ('Unemployment rate',),
        "Inflation": ('Inflation rate',),
        "Alder ved studiestart": ('Age at enrollment',),
        "Tidligere karakterer": ('Previous qualification (grade)',),
        "Fars uddannelse": ("Father's qualification",)
    }
    for title, (col,) in anova_tests.items():
        st.subheader(title)
        dropout = df[df['target'] == 1][col].dropna()
        graduate = df[df['target'] == 0][col].dropna()
        enrolled = df[df['target'] == 2][col].dropna()
        f_stat, p_val = f_oneway(dropout, graduate, enrolled)
        st.write(f"F-statistic: {f_stat:.3f}, p-value: {p_val:.4f}")
        if p_val < 0.05:
            st.success("→ Signifikant forskel mellem mindst to grupper.")
        else:
            st.warning("→ Ingen signifikant forskel mellem grupperne.")

elif menu == "Konklusioner":
    st.header("Konklusioner")
    st.write("Analysen viser tydelige forskelle i karakterer og beståede fag mellem grupperne. Alder og økonomiske faktorer viser også interessante forskelle. ANOVA understøtter disse forskelle for nogle variable.")
