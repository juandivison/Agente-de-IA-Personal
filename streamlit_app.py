import os
import sqlite3
import json
import requests
import streamlit as st
import time
import chromadb
from google import genai

from google.genai import types
from google.genai.errors import ServerError, APIError

# ---------------------------------------------------------------------------
# 1. Configuración Inicial y Secrets
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Agente IA Carrera Personal - Juan Ramon Divison. RAG + GitHub", page_icon="🤖")
st.title("🤖 Asistente Virtual Personal con RAG (CV + LinkedIn + GitHub) & SQL")

# Obtener API Keys y configuraciones
try:
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    github_user = st.secrets.get("GITHUB_USERNAME") or os.getenv("GITHUB_USERNAME", "juandivison")
    github_token = st.secrets.get("GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN")
except Exception:
    api_key = os.getenv("GEMINI_API_KEY")
    github_user = os.getenv("GITHUB_USERNAME", "juandivison")
    github_token = os.getenv("GITHUB_TOKEN")

if not api_key:
    st.error("⚠️ Configura la variable GEMINI_API_KEY en los secretos de Streamlit o variables de entorno.")
    st.stop()

client = genai.Client(api_key=api_key)

def obtener_embedding(texto: str) -> list[float]:
    """Genera el vector de embeddings usando el modelo actual activo."""
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texto
    )
    if hasattr(response, 'embeddings') and response.embeddings:
        return response.embeddings[0].values
    elif hasattr(response, 'embedding') and response.embedding:
        return response.embedding.values
    return []

# ---------------------------------------------------------------------------
# 2. Función para Obtener Repositorios de GitHub y Archivos
# ---------------------------------------------------------------------------
def obtener_repositorios_github(usuario: str, token: str = None) -> list[str]:
    """Obtiene y formatea los proyectos públicos y privados de un usuario en GitHub usando token."""
    if not usuario or usuario == "tu_usuario_github":
        return []
    
    url = "https://api.github.com/user/repos?per_page=100&sort=updated&type=all"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    if token:
        headers["Authorization"] = f"token {token}"
    else:
        url = f"https://api.github.com/users/{usuario}/repos?per_page=100&sort=updated"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            repos = response.json()
            documentos_repos = []
            for repo in repos:
                nombre = repo.get("name", "")
                descripcion = repo.get("description") or "Sin descripción"
                lenguaje = repo.get("language") or "No especificado"
                es_privado = "Privado" if repo.get("private") else "Público"
                topics = ", ".join(repo.get("topics", []))
                
                texto = f"Proyecto de GitHub [{es_privado}] '{nombre}': {descripcion}. Lenguaje principal: {lenguaje}."
                if topics:
                    texto += f" Tecnologías/Temas: {topics}."
                
                documentos_repos.append(texto)
            return documentos_repos
        else:
            st.warning(f"No se pudieron obtener repositorios de GitHub (Código HTTP {response.status_code}).")
            return []
    except Exception as e:
        st.warning(f"Error al conectar con la API de GitHub: {e}")
        return []

def cargar_y_fragmentar_cv(ruta_archivo: str = "cv.md", tamano_bloque: int = 500) -> list[str]:
    """Lee el CV de un archivo de texto o Markdown y lo divide en párrafos/bloques."""
    if not os.path.exists(ruta_archivo):
        return []
    
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        contenido = f.read()
    
    parrafos = [p.strip() for p in contenido.split("\n\n") if p.strip()]
    return parrafos

# ---------------------------------------------------------------------------
# 3. Inicialización y Poblado de la BD Vectorial (RAG)
# ---------------------------------------------------------------------------
@st.cache_resource
def init_rag_vector_db(usuario_gh: str):
    """Inicializa ChromaDB e indexa tanto el CV como los proyectos de GitHub."""
    chroma_client = chromadb.PersistentClient(path="./cv_vector_db")
    collection = chroma_client.get_or_create_collection(name="cv_knowledge_base")

    if collection.count() == 0:
        datos_cv = cargar_y_fragmentar_cv("cv.md")
        datos_linkedin = cargar_y_fragmentar_cv("linkedin.md")
        datos_github = obtener_repositorios_github(usuario_gh, github_token)
        
        documentos_totales = datos_cv + datos_linkedin + datos_github

        for i, texto in enumerate(documentos_totales):
            embedding_vector = obtener_embedding(texto)
            collection.add(
                documents=[texto],
                embeddings=[embedding_vector],
                ids=[f"rag_doc_{i}"]
            )
            
    return collection

collection = init_rag_vector_db(github_user)

# ---------------------------------------------------------------------------
# 4. Base de Datos SQL (Herramienta Estructurada)
# ---------------------------------------------------------------------------
def init_sql_db():
    conn = sqlite3.connect("agente_stats.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS estadisticas_proyectos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            categoria TEXT,
            cantidad_proyectos INTEGER,
            nivel_experiencia TEXT
        )
    """)
    cursor.execute("SELECT COUNT(*) FROM estadisticas_proyectos")
    if cursor.fetchone()[0] == 0:
        cursor.executemany("""
            INSERT INTO estadisticas_proyectos (categoria, cantidad_proyectos, nivel_experiencia)
            VALUES (?, ?, ?)
        """, [
            ("ERP & POS (Delphi / Firebird)", 12, "Senior / Arquitecto"),
            ("Servicios Web & APIs (.NET / Node.js)", 8, "Senior"),
            ("Facturación Electrónica e-CF", 5, "Especialista"),
            ("Capacitación Técnica / Docencia", 15, "Facilitador / Tutor")
        ])
        conn.commit()
    conn.close()

init_sql_db()

# ---------------------------------------------------------------------------
# 5. Funciones del Agente RAG
# ---------------------------------------------------------------------------
def buscar_en_rag(pregunta: str) -> str:
    """Busca los fragmentos más relevantes para la consulta."""
    vector_pregunta = obtener_embedding(pregunta)
    resultados = collection.query(query_embeddings=[vector_pregunta], n_results=4)
    docs = resultados.get('documents', [[]])[0]
    return "\n".join(docs) if docs else "No se encontraron fragmentos relevantes."

def ejecutar_consulta_sql(query: str) -> str:
    try:
        conn = sqlite3.connect("agente_stats.db")
        cursor = conn.cursor()
        cursor.execute(query)
        filas = cursor.fetchall()
        columnas = [desc[0] for desc in cursor.description]
        conn.close()
        return json.dumps([dict(zip(columnas, fila)) for fila in filas], ensure_ascii=False)
    except Exception as e:
        return f"Error SQL: {e}"

def generar_con_reintento(prompt: str, max_reintentos: int = 3):
    """Intenta generar contenido con el modelo de IA.
    Si el servidor está ocupado (503), espera y reintenta hasta max_reintentos veces.
    """
    for intento in range(max_reintentos):
        try:
            return client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )
        except ServerError as e:
            if intento == max_reintentos - 1:
                raise e
            time.sleep(2 * (intento + 1))

def responder_usuario(pregunta_usuario: str) -> str:
    try:
        contexto_rag = buscar_en_rag(pregunta_usuario)
        
        contexto_sql = ""
        if any(p in pregunta_usuario.lower() for p in ["cuantos", "cuántos", "cantidad", "estadística", "proyectos", "experiencia"]):
            contexto_sql = ejecutar_consulta_sql("SELECT * FROM estadisticas_proyectos")

        prompt = f"""
        Eres el asistente virtual personal interactivo de Juan Ramón Divisón (Software Architect & Developer).
        Tu objetivo es responder de manera profesional y amigable a reclutadores, clientes y desarrolladores.

        REGLAS DE RESPUESTA:
        1. Si el usuario pregunta cómo contactar a Juan Ramón Divisón, proporcionar directamente sus canales oficiales de contacto:
           - 📧 Correo electrónico: divison@gmail.com
           - 📱 Teléfono / WhatsApp: +1 (809) 309-5001
           - 💼 LinkedIn: https://www.linkedin.com/in/juandivison/
           - 🌐 Sitio Web / Empresa: www.idesisa.com
        2. Para preguntas profesionales o técnicas, utiliza la información recuperada de la DB RAG (CV + LinkedIn + GitHub) y SQL.

        [INFORMACIÓN RECUPERADA DE LA DB RAG]:
        {contexto_rag}

        [ESTADÍSTICAS CONSULTADAS DE LA DB SQL]:
        {contexto_sql}

        Pregunta del usuario: {pregunta_usuario}
        """

        response = generar_con_reintento(prompt)
        return response.text

    except ServerError:
        return "⚠️ El servicio de inteligencia artificial está experimentando una alta demanda en este momento. Por favor, intenta de nuevo en unos segundos."
    except APIError:
        return "⚠️ Ocurrió un inconveniente temporal de comunicación con el servidor. Inténtalo de nuevo más tarde."
    except Exception as e:
        return f"⚠️ Ocurrió un error inesperado: {e}"

# ---------------------------------------------------------------------------
# 6. Interfaz de Usuario e Historial de Conversación
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuración")
    input_gh = st.text_input("Usuario de GitHub:", value=github_user)
    if input_gh != github_user:
        st.info("Para aplicar un nuevo usuario de GitHub, elimina la carpeta ./cv_vector_db y reinicia.")

# Inicialización de variables de estado
if "messages" not in st.session_state:
    st.session_state.messages = []

if "error_occurred" not in st.session_state:
    st.session_state.error_occurred = False

if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""

# Mostrar historial existente
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Función para ejecutar la consulta y almacenar la respuesta
def procesar_pregunta(prompt_texto: str):
    st.session_state.last_prompt = prompt_texto
    
    with st.chat_message("assistant"):
        with st.spinner("Consultando RAG (CV + LinkedIn + GitHub)..."):
            respuesta = responder_usuario(prompt_texto)
            
            # Verificar si se devolvió un error para activar el botón de reintento
            if respuesta.startswith("⚠️"):
                st.session_state.error_occurred = True
                st.error(respuesta)
            else:
                st.session_state.error_occurred = False
                st.markdown(respuesta)
                st.session_state.messages.append({"role": "assistant", "content": respuesta})

# Entrada de chat
if prompt_input := st.chat_input("Haz una pregunta sobre mi CV, proyectos o cómo contactarme..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)
    
    procesar_pregunta(prompt_input)

# Mostrar botón de reintento en la interfaz si falló la llamada anterior
if st.session_state.error_occurred and st.session_state.last_prompt:
    if st.button("🔄 Reintentar respuesta", key="btn_reintentar"):
        st.session_state.error_occurred = False
        procesar_pregunta(st.session_state.last_prompt)
        st.rerun()