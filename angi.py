import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import plotly.graph_objects as go
import seaborn as sns
import plotly.express as px

# Configuración de estilo y tema personalizado
st.set_page_config(page_title="Predicción de Defectos en Productos", layout="wide")
st.markdown("""
    <style>
        .navbar {
            font-size: 24px;
            font-weight: bold;
            color: white;
            background-color: #007bff;
            padding: 10px;
            text-align: center;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Barra de navegación
st.markdown('<div class="navbar">🔍 Predicción de Defectos en Productos</div>', unsafe_allow_html=True)

# Pestañas interactivas
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Inicio", "Paso 1: Coeficientes", "Paso 2: Frontera de Decisión", 
    "Paso 3: Métricas de Evaluación", "Paso 4: Curva ROC y AUC", 
    "Paso 5: Análisis de Residuos", "Predicción Personalizada"
])

# Variables globales
df = None

# Tab: Inicio
with tab1:
    st.title("🔍 Predicción de Defectos en Productos")
    st.write("""
    Esta aplicación utiliza un modelo de regresión logística para predecir si un producto es defectuoso en función 
    de la cantidad en el lote y el tiempo de entrega.
    """)
    uploaded_file = st.file_uploader("Sube un archivo de datos (CSV o XLSX)", type=["csv", "xlsx"])
    
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        st.success("Archivo cargado con éxito. Ve a las pestañas para continuar.")
    else:
        st.info("Por favor, carga un archivo CSV o XLSX para comenzar.")

# Si el archivo está cargado, mostrar contenido en otras pestañas
if uploaded_file and df is not None:
    # Verificar que las columnas requeridas existen
    required_columns = ['Productos-Lote', 'Tiempo-Entrega', 'Defectuoso']
    if all(column in df.columns for column in required_columns):
        X1, X2, Y = 'Productos-Lote', 'Tiempo-Entrega', 'Defectuoso'
        X = df[[X1, X2]]
        y = df[Y]
        
        # Escalar datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # División de datos
        test_size = 0.3
        C_value = 1.0
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=0)
        
        # Entrenamiento del modelo
        model = LogisticRegression(solver='liblinear', C=C_value)
        model.fit(X_train, y_train)
        
         # Paso 1: Coeficientes del Modelo
        with tab2:
            st.subheader("📊 Paso 1: Coeficientes del Modelo")
            st.write(f"Intercepto (β₀): {model.intercept_[0]:.4f}")
            st.write(f"Coeficiente de {X1} (β₁): {model.coef_[0][0]:.4f}")
            st.write(f"Coeficiente de {X2} (β₂): {model.coef_[0][1]:.4f}")
            
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div class='card'>
                <strong>Interpretación:</strong> Los coeficientes muestran el impacto de cada variable en la probabilidad 
                de que el producto sea defectuoso. Un aumento en el valor de Productos-Lote o Tiempo-Entrega cambia 
                la probabilidad de que el producto sea defectuoso según el signo y la magnitud de sus coeficientes.
            </div>
            """, unsafe_allow_html=True)
            
            # Gráfico de coeficientes
            st.write("### Visualización de los Coeficientes")
            fig, ax = plt.subplots()
            coef_names = ["Intercepto", X1, X2]
            coef_values = [model.intercept_[0], model.coef_[0][0], model.coef_[0][1]]
            ax.barh(coef_names, coef_values)
            ax.set_title("Coeficientes del Modelo")
            ax.set_xlabel("Valor")
            st.pyplot(fig)
        
        # Paso 2: Visualización de la Frontera de Decisión
        with tab3:
            st.subheader("🌐 Paso 2: Visualización de la Frontera de Decisión")
            
            # Calcular los límites del gráfico
            x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
            y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
            h = (x_max - x_min) / 100  # Tamaño del paso para la cuadrícula
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            
            # Predecir sobre la cuadrícula
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Crear gráfico interactivo con Plotly
            fig = go.Figure(data=[
                go.Contour(
                    x=xx[0],
                    y=yy[:, 0],
                    z=Z,
                    showscale=False,
                    colorscale='RdBu',
                    opacity=0.5,
                    contours=dict(start=0, end=1, size=1, coloring='heatmap'),
                    name="Frontera de Decisión"
                ),
                go.Scatter(
                    x=X_scaled[y == 0, 0],
                    y=X_scaled[y == 0, 1],
                    mode='markers',
                    marker=dict(color='blue', size=8),
                    name='No Defectuoso'
                ),
                go.Scatter(
                    x=X_scaled[y == 1, 0],
                    y=X_scaled[y == 1, 1],
                    mode='markers',
                    marker=dict(color='red', size=8),
                    name='Defectuoso'
                )
            ])
            
            # Configuración del diseño del gráfico
            fig.update_layout(
                xaxis_title=X1,
                yaxis_title=X2,
                title="Regresión Logística con Frontera de Decisión",
                legend_title="Clases",
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            # Mostrar el gráfico interactivo
            st.plotly_chart(fig)
            
            # Divider y tarjeta de interpretación
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div class='card'>
                <strong>Interpretación:</strong> La frontera de decisión es la línea que separa las clases 
                'Defectuoso' y 'No Defectuoso' en el espacio de características. Esta línea muestra cómo el modelo 
                clasifica los productos basándose en sus características y permite ver las zonas de predicción de cada clase.
            </div>
            """, unsafe_allow_html=True)
        
        # Paso 3: Métricas de Evaluación
        with tab4:
            st.subheader("📉 Paso 3: Métricas de Evaluación - Matriz de Confusión")
            
            # Predicciones del modelo
            y_pred = model.predict(X_test)
            
            # Calcular matriz de confusión
            cm = confusion_matrix(y_test, y_pred)
            
            # Visualización interactiva con Plotly
            st.markdown("#### Matriz de Confusión (Interactividad con Plotly)")
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale="Blues",
                labels=dict(x="Predicción", y="Verdadero", color="Frecuencia"),
                title="Matriz de Confusión"
            )
            fig_cm.update_layout(
                xaxis_title="Predicción",
                yaxis_title="Verdadero",
                coloraxis_showscale=True
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Divider y tarjeta de interpretación
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div class='card'>
                <strong>Interpretación:</strong> La matriz de confusión muestra el rendimiento del modelo en términos 
                de verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos. Esto permite evaluar 
                la precisión y efectividad del modelo para clasificar productos defectuosos y no defectuosos.
            </div>
            """, unsafe_allow_html=True)
            
            # Reporte de clasificación adicional
            st.markdown("#### Reporte de Clasificación")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"}))


        # Paso 4: Curva ROC y AUC
        with tab5:
            st.subheader("🔄 Paso 4: Curva ROC y AUC")
            
            # Obtener probabilidades predichas
            y_scores = model.predict_proba(X_test)[:, 1]
            
            # Calcular puntos para la curva ROC
            fpr, tpr, thresholds = roc_curve(y_test, y_scores)
            roc_auc = auc(fpr, tpr)
            
            # Visualización de la Curva ROC con Plotly
            st.markdown("#### Curva ROC")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, 
                mode='lines', 
                name=f'AUC = {roc_auc:.2f}',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], 
                mode='lines', 
                line=dict(dash='dash', color='gray'),
                name='Azar'
            ))
            fig.update_layout(
                xaxis_title='Tasa de Falsos Positivos (FPR)',
                yaxis_title='Tasa de Verdaderos Positivos (TPR)',
                title=f'Curva ROC - AUC: {roc_auc:.2f}',
                showlegend=True
            )
            st.plotly_chart(fig)
            
            # Divider y explicación
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div class='card'>
                <strong>Interpretación:</strong> La curva ROC muestra la capacidad del modelo para distinguir entre 
                las clases defectuoso y no defectuoso. Un área bajo la curva (AUC) cercana a 1 indica un modelo que 
                clasifica correctamente; un AUC cercano a 0.5 sugiere que el modelo es casi aleatorio.
            </div>
            """, unsafe_allow_html=True)
            
            # Análisis adicional: Impacto de los umbrales
            st.markdown("#### Impacto de los Umbrales")
            threshold_df = pd.DataFrame({
                "Umbral": thresholds,
                "FPR (Tasa de Falsos Positivos)": fpr,
                "TPR (Tasa de Verdaderos Positivos)": tpr
            })
            st.dataframe(
                threshold_df.style.format({
                    "Umbral": "{:.2f}",
                    "FPR (Tasa de Falsos Positivos)": "{:.2f}",
                    "TPR (Tasa de Verdaderos Positivos)": "{:.2f}"
                })
            )

        # Paso 5: Análisis de Residuos
        with tab6:
            st.subheader("📌 Paso 5: Análisis de Residuos")
            
            # Calcular residuos
            y_pred_prob = model.predict_proba(X_test)[:, 1]
            residuals = y_test - y_pred_prob
            
            # Visualización 1: Histograma de residuos con seaborn
            st.markdown("#### Distribución de Residuos (Histograma)")
            fig, ax = plt.subplots()
            sns.histplot(residuals, kde=True, color='blue', ax=ax)
            ax.set_title("Distribución de Residuos")
            ax.set_xlabel("Residuo")
            ax.set_ylabel("Frecuencia")
            st.pyplot(fig)
            
            # Divider
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            
            # Visualización 2: Residuos vs Probabilidades Predichas
            st.markdown("#### Residuos vs Probabilidades Predichas")
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=y_pred_prob,
                y=residuals,
                mode='markers',
                marker=dict(size=7, color='blue', opacity=0.6),
                name="Residuos"
            ))
            fig_scatter.update_layout(
                xaxis_title="Probabilidades Predichas",
                yaxis_title="Residuo",
                title="Residuos vs Probabilidades Predichas",
                showlegend=False
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Divider y explicación
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div class='card'>
                <strong>Interpretación:</strong> 
                - Histograma de Residuos: Una distribución simétrica alrededor de cero sugiere que el modelo captura bien los patrones de los datos. Una distribución sesgada podría indicar problemas de ajuste.
                - Residuos vs Probabilidades Predichas: Este gráfico ayuda a detectar patrones en los residuos. Idealmente, los residuos deben distribuirse aleatoriamente alrededor de cero, sin tendencias claras.
            </div>
            """, unsafe_allow_html=True)
            
            # Resumen estadístico de los residuos
            st.markdown("#### Resumen Estadístico de los Residuos")
            residuals_summary = pd.DataFrame({
                "Métricas": ["Media", "Mediana", "Desviación Estándar", "Mínimo", "Máximo"],
                "Valores": [
                    residuals.mean(),
                    np.median(residuals),
                    residuals.std(),
                    residuals.min(),
                    residuals.max()
                ]
            })
            st.table(residuals_summary.style.format({"Valores": "{:.4f}"}))

        
        # Predicción Personalizada
        with tab7:
            st.subheader("🔮 Predicción de Nuevo Producto")
            
            # Entrada del usuario para características personalizadas
            Variable_x1 = st.number_input("Cantidad de Productos en el Lote:", min_value=0, value=50, step=1)
            Variable_x2 = st.number_input("Tiempo de Entrega (en minutos):", min_value=0, value=80000, step=10)
            
            if st.button("Predecir"):
                # Crear un DataFrame con los valores ingresados
                new_example = pd.DataFrame([[Variable_x1, Variable_x2]], columns=[X1, X2])
                
                # Escalar los datos ingresados
                new_example_scaled = scaler.transform(new_example)
                
                # Realizar predicción
                probability = model.predict_proba(new_example_scaled)[0][1]
                prediction = model.predict(new_example_scaled)
                result_phrase = "DEFECTUOSO" if prediction[0] == 1 else "NO DEFECTUOSO"
                
                # Mostrar el resultado
                st.success(f"Un producto en un lote de {Variable_x1} unidades y con un tiempo de entrega de {Variable_x2} minutos es probable que esté en estado: {result_phrase}.")
                st.write(f"Probabilidad estimada de que sea defectuoso: {probability:.4f}")
                
                # Visualización interactiva de la probabilidad
                st.markdown("#### Visualización de Probabilidad")
                fig_probability = go.Figure()
                fig_probability.add_trace(go.Bar(
                    x=["Probabilidad de No Defectuoso", "Probabilidad de Defectuoso"],
                    y=[1 - probability, probability],
                    marker=dict(color=["blue", "red"]),
                    text=[f"{(1 - probability) * 100:.2f}%", f"{probability * 100:.2f}%"],
                    textposition="auto"
                ))
                fig_probability.update_layout(
                    title="Distribución de Probabilidad de Predicción",
                    xaxis_title="Clases",
                    yaxis_title="Probabilidad",
                    showlegend=False
                )
                st.plotly_chart(fig_probability, use_container_width=True)
                
                # Divider y explicación
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='card'>
                    <strong>Interpretación:</strong> 
                    Este resultado predice si un nuevo producto es defectuoso o no, basado en los valores ingresados 
                    para el tamaño del lote y el tiempo de entrega. La probabilidad estimada muestra la confianza del modelo en esta clasificación:
                    - NO DEFECTUOSO: Probabilidad de {(1 - probability) * 100:.2f}%.
                    - DEFECTUOSO: Probabilidad de {probability * 100:.2f}%.
                    Usa esta predicción para tomar decisiones informadas sobre la calidad del producto.
                </div>
                """, unsafe_allow_html=True)

else:
    st.warning("Carga un archivo para habilitar las pestañas adicionales.")
